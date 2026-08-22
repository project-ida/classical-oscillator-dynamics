
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st
from io import BytesIO
from pathlib import Path
from oscillator_model import (
    burst_duration,
    coupled_eigenfrequencies_khz,
    khz_to_omega,
    legacy_drive_frequency,
    steady_state_sweep,
    exact_rlc_eigenfrequencies_khz,
    exact_rlc_matrices,
    exact_rlc_sweep,
    experimental_energy_proxy,
    rk4_exact_rlc,
)

st.set_page_config(page_title="LC Transfer Analog Simulator", layout="wide")

# -----------------------------
# Fixed color scheme (consistent across modes/scenarios)
# -----------------------------
COLORS = {
    # Extracted/approximated from the attached schematic.
    "drive": "#0E7DAF",         # excitation/driver blue
    "donor": "#BD4E13",         # donor orange-brown
    "bus": "#000000",           # bus black
    "acceptor": "#348417",      # acceptor green

    # Auxiliary plotted quantity; not directly represented in the schematic.
    "undissipated": "#777777",  # gray
}

# -----------------------------
# Helper functions
# -----------------------------
def hz_to_omega(f_hz: float) -> float:
    return 2 * np.pi * f_hz

def make_drive(t, omega_ref, duration, envelope_shape="sin2"):
    """Return a selectable finite-burst envelope and physical sine carrier."""
    envelope = np.zeros_like(t)
    inside = (t >= 0) & (t <= duration)
    if np.any(inside):
        if envelope_shape == "rectangular":
            envelope[inside] = 1.0
        else:
            tau = t[inside] / duration
            envelope[inside] = np.sin(np.pi * tau) ** 2
    carrier = np.sin(omega_ref * t)
    return envelope, envelope * carrier

def rk4_coupled_modes(t, omegas, kappas, G, drive_vector, drive_envelope, omega_ref):
    """
    Rotating-frame coupled-mode model:

        da_i/dt = -(kappa_i/2 + i delta_i) a_i
                  - i sum_j G_ij a_j
                  + drive_i(t)

    with delta_i = omega_i - omega_ref.
    """
    omegas = np.array(omegas, dtype=float)
    kappas = np.array(kappas, dtype=float)
    G = np.array(G, dtype=float)
    drive_vector = np.array(drive_vector, dtype=float)

    deltas = omegas - omega_ref
    n_modes = len(omegas)
    a = np.zeros((len(t), n_modes), dtype=complex)

    def interp_env(t_now):
        return np.interp(t_now, t, drive_envelope)

    def deriv(t_now, y):
        env = interp_env(t_now)
        drive_now = drive_vector * env
        return -(kappas / 2 + 1j * deltas) * y - 1j * (G @ y) + drive_now

    for i in range(len(t) - 1):
        y = a[i].copy()
        h = t[i + 1] - t[i]

        k1 = deriv(t[i], y)
        k2 = deriv(t[i] + h / 2, y + h * k1 / 2)
        k3 = deriv(t[i] + h / 2, y + h * k2 / 2)
        k4 = deriv(t[i] + h, y + h * k3)

        a[i + 1] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return a

def energy_accounting(t, a, kappas, drive_vector, drive_envelope):
    """Return mode energies, cumulative drive work, and cumulative dissipated energy."""
    kappas = np.array(kappas, dtype=float)
    drive_vector = np.array(drive_vector, dtype=float)

    E_modes = np.abs(a) ** 2

    drive_now = drive_envelope[:, None] * drive_vector[None, :]
    P_drive = 2 * np.real(np.conj(a) * drive_now).sum(axis=1)
    W_drive = np.zeros_like(t)
    W_drive[1:] = np.cumsum(0.5 * (P_drive[1:] + P_drive[:-1]) * np.diff(t))

    P_loss = (kappas[None, :] * E_modes).sum(axis=1)
    E_loss = np.zeros_like(t)
    E_loss[1:] = np.cumsum(0.5 * (P_loss[1:] + P_loss[:-1]) * np.diff(t))

    start_idx = np.searchsorted(t, 0)
    W_drive -= W_drive[start_idx]
    E_loss -= E_loss[start_idx]

    return E_modes, W_drive, E_loss

def build_plot(
    t,
    a,
    amplitude_meta,
    energy_meta,
    kappas,
    drive_vector,
    drive_envelope,
    drive_signal_for_plot,
    omega_ref,
    duration,
    title,
    x_axis_max_us,
    plot_width_px,
    plot_height_px,
    font_scale,
    energy_norm_mode,
    observable_mode,
):
    """
    amplitude_meta: list of dicts with keys:
        {"label": ..., "indices": [...], "color_key": ...}

    energy_meta: list of dicts with keys:
        {"label": ..., "indices": [...], "color_key": ...}
    """
    # Build displayed physical signals
    displayed_signals = []
    amp_labels = []
    amp_colors = []
    for item in amplitude_meta:
        indices = item["indices"]
        amp = a[:, indices].sum(axis=1)
        signal = np.real(amp * np.exp(-1j * omega_ref * t))
        displayed_signals.append(signal)
        amp_labels.append(item["label"])
        amp_colors.append(COLORS[item["color_key"]])

    displayed_signals = np.column_stack(displayed_signals) if displayed_signals else np.zeros((len(t), 0))

    max_amp = np.max(np.abs(displayed_signals)) if displayed_signals.size else 1.0
    if max_amp <= 0:
        max_amp = 1.0
    displayed_signals_n = displayed_signals / max_amp

    # Energies
    E_modes, W_drive, E_loss = energy_accounting(t, a, kappas, drive_vector, drive_envelope)
    proxy_mode = observable_mode == "Experimental envelope-energy proxy"
    displayed_energy = (
        experimental_energy_proxy(t, displayed_signals, np.ones(displayed_signals.shape[1]))
        if proxy_mode else E_modes
    )

    if proxy_mode:
        norm = (np.max(displayed_energy[:, 0]) if energy_norm_mode == "Peak donor stored energy"
                else np.max(displayed_energy))
    else:
        norm = np.max(E_modes[:, 0]) if energy_norm_mode == "Peak donor stored energy" else np.max(W_drive)
    if norm <= 0:
        norm = max(np.max(E_modes), 1.0)

    W_drive_n = W_drive / norm
    E_undissipated_n = (W_drive - E_loss) / norm

    # Figure with shared x-axis.
    # Matplotlib uses inches, so convert requested pixel size using a fixed DPI.
    fig_dpi = 100
    fig_width_in = plot_width_px / fig_dpi
    fig_height_in = plot_height_px / fig_dpi

    fig, (ax_amp, ax_energy) = plt.subplots(
        2,
        1,
        figsize=(fig_width_in, fig_height_in),
        dpi=fig_dpi,
        sharex=True,
        height_ratios=[1, 1.1],
    )

    title_fs = 12 * font_scale
    label_fs = 10 * font_scale
    tick_fs = 9 * font_scale
    legend_fs = 9 * font_scale
    annotation_fs = 10 * font_scale

    # -----------------------------
    # Amplitude subplot (left axis: mode amplitudes)
    # -----------------------------
    for j, label in enumerate(amp_labels):
        ax_amp.plot(
            t * 1e6,
            displayed_signals_n[:, j],
            linewidth=1.8,
            label=label,
            color=amp_colors[j],
        )

    ax_amp.axvline(0, linestyle="--", linewidth=1, color="gray")
    ax_amp.axvline(duration * 1e6, linestyle="--", linewidth=1, color="gray")
    ax_amp.set_title(title, fontsize=title_fs)
    ax_amp.set_ylabel("Normalized mode amplitude", fontsize=label_fs)
    amplitude_ylim = (-1.15, 1.40)
    ax_amp.set_ylim(amplitude_ylim)
    amplitude_ticks = np.linspace(-1.0, 1.0, 5)
    ax_amp.set_yticks(amplitude_ticks)
    ax_amp.tick_params(axis="both", labelsize=tick_fs)
    ax_amp.legend(loc="upper left", fontsize=legend_fs)

    # Secondary y-axis for the drive
    ax_drive = ax_amp.twinx()
    drive_max = np.max(np.abs(drive_signal_for_plot))
    if drive_max <= 0:
        drive_max = 1.0
    drive_signal_n = drive_signal_for_plot / drive_max

    ax_drive.plot(
        t * 1e6,
        drive_signal_n,
        linestyle="--",
        linewidth=1.4,
        color=COLORS["drive"],
        label="Drive waveform",
    )
    ax_drive.set_ylabel("Normalized drive amplitude", color=COLORS["drive"], fontsize=label_fs)
    ax_drive.tick_params(axis='y', colors=COLORS["drive"], labelsize=tick_fs)
    ax_drive.set_yticks(amplitude_ticks)
    ax_drive.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_drive.set_ylim(amplitude_ylim)

    # Merge legends
    lines1, labels1 = ax_amp.get_legend_handles_labels()
    lines2, labels2 = ax_drive.get_legend_handles_labels()
    ax_amp.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=legend_fs)

    ax_amp.text(duration * 1e6 / 2, 1.15, "drive on", ha="center", fontsize=annotation_fs)
    ax_amp.text(
        duration * 1e6 + 0.12 * (t[-1] - t[0]) * 1e6,
        1.15,
        "free evolution",
        ha="center",
        fontsize=annotation_fs,
    )

    # -----------------------------
    # Energy subplot
    # -----------------------------
    if energy_norm_mode == "Delivered drive energy" and not proxy_mode:
        ax_energy.plot(
            t * 1e6, W_drive_n, linewidth=2.0,
            label="Cumulative energy delivered by drive", color=COLORS["drive"],
        )

    for item in energy_meta:
        if proxy_mode:
            curve_index = energy_meta.index(item)
            curve = displayed_energy[:, curve_index] / norm
            curve_label = f"Experimental-analysis energy proxy: {item['label'].replace('Stored ', '')}"
        else:
            curve = displayed_energy[:, item["indices"]].sum(axis=1) / norm
            curve_label = item["label"]
        ax_energy.plot(
            t * 1e6,
            curve,
            linewidth=2.0,
            label=curve_label,
            color=COLORS[item["color_key"]],
        )

    if energy_norm_mode == "Delivered drive energy" and not proxy_mode:
        ax_energy.plot(
            t * 1e6, E_undissipated_n, linewidth=2.0, linestyle="--",
            label="Total undissipated delivered energy", color=COLORS["undissipated"],
        )

    ax_energy.axvline(0, linestyle="--", linewidth=1, color="gray")
    ax_energy.axvline(duration * 1e6, linestyle="--", linewidth=1, color="gray")
    ax_energy.set_xlabel("Time (µs)", fontsize=label_fs)
    energy_ylabel = ("Experimental-analysis energy proxy / donor peak" if proxy_mode else
        "Energy / peak donor stored energy"
        if energy_norm_mode == "Peak donor stored energy"
        else "Energy / peak delivered drive energy"
    )
    ax_energy.set_ylabel(energy_ylabel, fontsize=label_fs)
    ax_energy.set_xlim(t[0] * 1e6, x_axis_max_us)
    ax_energy.set_ylim(-0.03, 1.22)
    ax_energy.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_energy.tick_params(axis="both", labelsize=tick_fs)
    ax_energy.legend(loc="upper right", fontsize=legend_fs)
    ax_energy.text(duration * 1e6 / 2, 1.10, "drive on", ha="center", fontsize=annotation_fs)
    ax_energy.text(
        duration * 1e6 + 0.12 * (t[-1] - t[0]) * 1e6,
        1.10,
        "drive off",
        ha="center",
        fontsize=annotation_fs,
    )

    fig.align_ylabels([ax_amp, ax_energy])
    plt.tight_layout()
    return fig

def build_exact_rlc_plot(t, q, currents, C_matrix, L_matrix, drive_waveform, drive_strength,
                         duration, title, x_axis_max_us, energy_norm_mode,
                         observable_mode, plot_width_px, plot_height_px):
    capacitances = np.diag(C_matrix)
    voltages = q / capacitances[None, :]
    amplitude_scale = max(float(np.max(np.abs(voltages))), 1e-30)
    fig, (ax_v, ax_e) = plt.subplots(
        2, 1, sharex=True, figsize=(plot_width_px / 100, plot_height_px / 100), dpi=100
    )
    labels, keys = ["Donor", "Bus", "Acceptor"], ["donor", "bus", "acceptor"]
    for index, (label, key) in enumerate(zip(labels, keys)):
        ax_v.plot(t * 1e6, voltages[:, index] / amplitude_scale,
                  label=f"{label} capacitor voltage", color=COLORS[key], linewidth=1.5)
    ax_v.plot(t * 1e6, drive_waveform, "--", color=COLORS["drive"], label="Drive waveform")
    ax_v.set(title=title, ylabel="Normalized voltage")
    ax_v.legend(loc="upper right")

    if observable_mode == "Experimental envelope-energy proxy":
        curves = experimental_energy_proxy(t, voltages, capacitances)
        curve_prefix = "Experimental-analysis energy proxy"
        donor_norm = max(float(np.max(curves[:, 0])), 1e-30)
        norm = donor_norm if energy_norm_mode == "Peak donor stored energy" else max(float(np.max(curves)), 1e-30)
        for index, (label, key) in enumerate(zip(labels, keys)):
            ax_e.plot(t * 1e6, curves[:, index] / norm,
                      label=f"{curve_prefix}: {label}", color=COLORS[key], linewidth=2)
        ax_e.set_ylabel("Envelope-energy proxy / donor peak" if energy_norm_mode == "Peak donor stored energy"
                        else "Normalized envelope-energy proxy")
    else:
        self_energy = 0.5 * q**2 / capacitances[None, :] + 0.5 * currents**2 * np.diag(L_matrix)[None, :]
        electric = 0.5 * np.einsum("ti,ij,tj->t", q, np.diag(1/capacitances), q)
        magnetic = 0.5 * np.einsum("ti,ij,tj->t", currents, L_matrix, currents)
        mutual = magnetic - 0.5 * np.sum(currents**2 * np.diag(L_matrix)[None, :], axis=1)
        total = electric + magnetic
        drive_power = drive_strength * drive_waveform * currents[:, 0]
        drive_work = np.zeros(len(t))
        drive_work[1:] = np.cumsum(0.5 * (drive_power[1:] + drive_power[:-1]) * np.diff(t))
        norm = max(float(np.max(self_energy[:, 0] if energy_norm_mode == "Peak donor stored energy" else drive_work)), 1e-30)
        for index, (label, key) in enumerate(zip(labels, keys)):
            ax_e.plot(t * 1e6, self_energy[:, index] / norm,
                      label=f"{label} local self-energy", color=COLORS[key], linewidth=1.7)
        ax_e.plot(t * 1e6, mutual / norm, label="Mutual coupling energy", color="purple", linestyle="--")
        ax_e.plot(t * 1e6, total / norm, label="Total physical energy", color=COLORS["undissipated"], linewidth=2)
        if energy_norm_mode == "Delivered drive energy":
            ax_e.plot(t * 1e6, drive_work / norm, label="Cumulative delivered drive energy",
                      color=COLORS["drive"], linewidth=1.5)
        ax_e.set_ylabel("Energy / donor peak" if energy_norm_mode == "Peak donor stored energy" else "Energy / peak delivered drive energy")
    for axis in (ax_v, ax_e):
        axis.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        axis.axvline(duration * 1e6, color="gray", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.15)
    ax_e.set(xlabel="Time (µs)", xlim=(t[0] * 1e6, x_axis_max_us))
    ax_e.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# -----------------------------
# URL query-parameter helpers
# -----------------------------
def _query_value(name):
    """Return the first query-param value as a string, or None."""
    try:
        value = st.query_params.get(name, None)
    except Exception:
        return None

    if isinstance(value, list):
        return value[0] if value else None
    return value

def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def query_float(name, default, min_value=None, max_value=None):
    raw = st.session_state.get(name, _query_value(name))
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)

    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value

def query_int(name, default, min_value=None, max_value=None):
    raw = st.session_state.get(name, _query_value(name))
    try:
        value = int(round(float(raw))) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)

    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value

def query_choice(name, options, default):
    raw = st.session_state.get(name, _query_value(name))
    return raw if raw in options else default

def query_float_option(name, options, default):
    raw = st.session_state.get(name, _query_value(name))
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)

    return min(options, key=lambda option: abs(float(option) - value))

def format_query_value(value):
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)

def sync_query_params(settings):
    """
    Write the current UI settings into the URL query parameters.
    This makes copied/bookmarked URLs reproduce the same configuration.
    """
    formatted_settings = {key: format_query_value(value) for key, value in settings.items()}
    if any(_query_value(key) != value for key, value in formatted_settings.items()):
        try:
            st.query_params.from_dict(formatted_settings)
        except AttributeError:
            st.query_params.update(formatted_settings)

# -----------------------------
# App UI
# -----------------------------
st.title("LC Transfer Analog Simulator")
st.caption(
    "Interactive classical LC analogs of donor preparation, direct transfer, resonant "
    "bus-mediated transfer, off-resonant bus-mediated transfer, and collective bright-mode enhancement."
)

scenario_options = [
    "Donor preparation",
    "a) Direct coupling",
    "b) Indirect coupling via resonant bus",
    "c) Indirect coupling via off-resonant bus",
    "d) Indirect coupling via off-resonant bus with collective enhancement",
]

scenario_default = query_choice("scenario", scenario_options, scenario_options[0])
scenario = st.sidebar.selectbox(
    "Scenario",
    scenario_options,
    index=scenario_options.index(scenario_default),
    key="scenario",
)

model_options = ["Coupled-mode approximation", "Exact mutually coupled RLC circuit"]
model = st.sidebar.selectbox(
    "Model", model_options,
    index=model_options.index(query_choice("model", model_options, model_options[0])),
    key="model",
)
exact_model = model == model_options[1]
st.sidebar.caption(
    "Coupled-mode: conceptual J_eff and bright/dark-mode intuition. "
    "Exact mutual RLC: quantitative coil/capacitor comparison."
)
if exact_model and scenario != "c) Indirect coupling via off-resonant bus":
    st.sidebar.warning("The exact D–B–A backend is currently available in scenario c; using coupled-mode here.")
    exact_model = False

if scenario == "c) Indirect coupling via off-resonant bus":
    def load_aug21_preset():
        preset = ({
            "fD_khz": 138.58, "fA_khz": 138.58, "fB_khz": 98.62,
            "C_D_nf": 100.0, "C_B_nf": 100.0, "C_A_nf": 103.2,
            "k_DB": 0.160, "k_BA": 0.160, "f_drive_khz": 142.13,
            "N_cycles": 3, "Q_D": 100, "Q_A": 100, "Q_B": 70,
            "drive_envelope": "rectangular", "observable_mode": "Experimental envelope-energy proxy",
            "energy_norm_mode": "Peak donor stored energy",
        } if exact_model else {
            "fD_khz": 138.61, "fA_khz": 138.61, "fB_khz": 103.5,
            "g_D_khz": 12.3, "g_A_khz": 12.3, "f_drive_khz": 142.1,
            "N_cycles": 3, "Q_D": 80, "Q_A": 80, "Q_B": 70,
            "energy_norm_mode": "Peak donor stored energy",
        })
        for key, value in preset.items():
            st.query_params[key] = format_query_value(value)
            st.session_state.pop(key, None)

    st.sidebar.button(
        ("Load Aug 21 exact mutual-L fit" if exact_model else "Load Aug 21 experimental fit"),
        on_click=load_aug21_preset,
        help=("Loads a simple three-mode spectral-fit starting point. The bare-frequency "
              "inputs are fitted model parameters, not isolated-resonator measurements."),
        width="stretch",
    )
    st.sidebar.caption(
        "Simple three-mode spectral-fit starting point; bare inputs are not independently "
        "measured isolated resonances."
    )

settings_to_sync = {"scenario": scenario, "model": model}

st.sidebar.header("Shared parameters")
fD_khz = st.sidebar.slider(
    "Bare donor frequency f_D (kHz)",
    20.0,
    300.0,
    query_float("fD_khz", 100.0, 20.0, 300.0),
    0.1,
    key="fD_khz",
)
fA_khz = st.sidebar.slider(
    "Bare acceptor frequency f_A (kHz)",
    20.0,
    300.0,
    query_float("fA_khz", 100.0, 20.0, 300.0),
    0.1,
    key="fA_khz",
)
f_drive_default = legacy_drive_frequency(
    _query_value("f_drive_khz"), fD_khz
)
f_drive_default = _clamp(f_drive_default, 20.0, 300.0)
f_drive_khz = st.sidebar.slider(
    "Drive carrier frequency f_drive (kHz)", 20.0, 300.0,
    f_drive_default, 0.1, key="f_drive_khz",
)
drive_envelope_values = ["rectangular", "sin2"]
drive_envelope_labels = {
    "rectangular": "Rectangular sine burst",
    "sin2": "sin² tapered burst",
}
drive_envelope = st.sidebar.selectbox(
    "Drive burst envelope", drive_envelope_values,
    index=drive_envelope_values.index(query_choice(
        "drive_envelope", drive_envelope_values, "sin2"
    )), format_func=lambda value: drive_envelope_labels[value], key="drive_envelope",
)
N_cycles = st.sidebar.slider(
    "Drive burst length (cycles)",
    1,
    20,
    query_int("N_cycles", 5, 1, 20),
    1,
    key="N_cycles",
)
Q_D = st.sidebar.slider(
    "Donor Q",
    10,
    1000,
    query_int("Q_D", 220, 10, 1000),
    10,
    key="Q_D",
)
Q_A = st.sidebar.slider(
    "Acceptor Q",
    10,
    1000,
    query_int("Q_A", 220, 10, 1000),
    10,
    key="Q_A",
)
drive_strength = st.sidebar.slider(
    "Drive strength (arb.)",
    1e4,
    5e5,
    query_float("drive_strength", 1.2e5, 1e4, 5e5),
    1e4,
    format="%.0f",
    key="drive_strength",
)
t_pre_us = st.sidebar.slider(
    "Time before burst (µs)",
    0.0,
    50.0,
    query_float("t_pre_us", 10.0, 0.0, 50.0),
    1.0,
    key="t_pre_us",
)
t_post_us = st.sidebar.slider(
    "Time after burst (µs)",
    100.0,
    3000.0,
    query_float("t_post_us", 850.0, 100.0, 3000.0),
    50.0,
    key="t_post_us",
)
dt_options = [0.01, 0.02, 0.05, 0.1, 0.2]
dt_us = st.sidebar.select_slider(
    "Time step (µs)",
    options=dt_options,
    value=query_float_option("dt_us", dt_options, 0.05),
    key="dt_us",
)
energy_norm_options = ["Delivered drive energy", "Peak donor stored energy"]
energy_norm_mode = st.sidebar.selectbox(
    "Energy normalization", energy_norm_options,
    index=energy_norm_options.index(query_choice(
        "energy_norm_mode", energy_norm_options, energy_norm_options[0]
    )), key="energy_norm_mode",
)
observable_options = ["Physical model energy", "Experimental envelope-energy proxy"]
observable_mode = st.sidebar.selectbox(
    "Energy / observable display", observable_options,
    index=observable_options.index(query_choice(
        "observable_mode", observable_options, observable_options[0]
    )), key="observable_mode",
)

settings_to_sync.update(
    {
        "fD_khz": fD_khz,
        "fA_khz": fA_khz,
        "f_drive_khz": f_drive_khz,
        "drive_envelope": drive_envelope,
        "N_cycles": N_cycles,
        "Q_D": Q_D,
        "Q_A": Q_A,
        "drive_strength": drive_strength,
        "t_pre_us": t_pre_us,
        "t_post_us": t_post_us,
        "dt_us": dt_us,
        "energy_norm_mode": energy_norm_mode,
        "observable_mode": observable_mode,
    }
)

omegaD = khz_to_omega(fD_khz)
omegaA = khz_to_omega(fA_khz)
omega_drive = khz_to_omega(f_drive_khz)
omega_ref = omega_drive

duration = burst_duration(N_cycles, f_drive_khz)
t = np.arange(-t_pre_us * 1e-6, duration + t_post_us * 1e-6, dt_us * 1e-6)
drive_envelope_values_t, drive_carrier_unit = make_drive(
    t, omega_ref, duration, drive_envelope
)
drive_envelope_key = drive_envelope
drive_envelope = drive_envelope_values_t

kappa_D = omegaD / Q_D
kappa_A = omegaA / Q_A

scenario_warning = None
scenario_notes = []
N_D = 1
N_A = 1

if scenario == "Donor preparation":
    scenario_notes = [
        ("markdown", "### Donor preparation"),
        (
            "markdown",
            "The drive prepares the donor LC excitation. The stored donor energy is the closest "
            "classical analog of a quantum-state occupation probability.",
        ),
    ]

    omegas = [omegaD]
    kappas = [kappa_D]
    G = [[0.0]]
    drive_vector = [drive_strength]

    amplitude_meta = [
        {"label": "Donor amplitude", "indices": [0], "color_key": "donor"},
    ]
    energy_meta = [
        {"label": "Stored donor energy", "indices": [0], "color_key": "donor"},
    ]

    drive_signal_for_plot = drive_strength * drive_carrier_unit

    title = f"Donor preparation after {N_cycles}-cycle {f_drive_khz:.1f} kHz burst"

elif scenario == "a) Direct coupling":
    # Direct coupling in kHz. For backward compatibility, old URLs using
    # J_hz are still read and converted if J_khz is not present.
    J_khz_default = query_float("J_hz", 3000.0, 0.0, 20000.0) / 1000.0
    J_khz = st.sidebar.slider(
        "Direct coupling J_a / 2π (kHz)",
        0.0,
        20.0,
        query_float("J_khz", J_khz_default, 0.0, 20.0),
        0.1,
        key="J_khz",
    )
    settings_to_sync["J_khz"] = J_khz

    J = khz_to_omega(J_khz)

    scenario_notes = [
        ("markdown", "### a) Direct coupling"),
        ("latex", r"J_a = g_{DA}(x)"),
        ("markdown", "In the weak/lossy transfer-rate limit,"),
        ("latex", r"\Gamma_a \sim \frac{|g_{DA}(x)|^2}{\kappa}"),
        (
            "markdown",
            "This simulation keeps the coherent dynamics explicitly, so visible energy sloshing "
            "appears when the coupling is strong enough.",
        ),
    ]

    omegas = [omegaD, omegaA]
    kappas = [kappa_D, kappa_A]
    G = [
        [0.0, J],
        [J, 0.0],
    ]
    drive_vector = [drive_strength, 0.0]

    amplitude_meta = [
        {"label": "Donor amplitude", "indices": [0], "color_key": "donor"},
        {"label": "Acceptor amplitude", "indices": [1], "color_key": "acceptor"},
    ]
    energy_meta = [
        {"label": "Stored donor energy", "indices": [0], "color_key": "donor"},
        {"label": "Stored acceptor energy", "indices": [1], "color_key": "acceptor"},
    ]

    drive_signal_for_plot = drive_strength * drive_carrier_unit

    title = (
        f"Direct donor-acceptor transfer after {N_cycles}-cycle {f_drive_khz:.1f} kHz burst\n"
        f"J_a/2π = {J_khz:.1f} kHz"
    )

elif scenario in (
    "c) Indirect coupling via off-resonant bus",
    "d) Indirect coupling via off-resonant bus with collective enhancement",
):
    collective_mode = scenario.startswith("d)")
    if collective_mode:
        scenario_notes = [
            ("markdown", "### d) Indirect coupling via off-resonant bus with collective enhancement"),
            (
                "markdown",
                "The simulation represents donor and acceptor bright modes coupled through an "
                "off-resonant bus.",
            ),
            (
                "latex",
                r"J_d \simeq \sqrt{N_D N_A}\frac{g_D g_A}{\Delta_B}",
            ),
        ]
    else:
        scenario_notes = [
            ("markdown", "### c) Indirect coupling via off-resonant bus"),
            (
                "markdown",
                "This is the single-donor/single-acceptor off-resonant bus case.",
            ),
            ("latex", r"J_c \simeq \frac{g_D g_A}{\Delta_B}"),
        ]

    fB_khz = st.sidebar.slider(
        "Bare bus frequency f_B (kHz)",
        5.0,
        300.0,
        query_float("fB_khz", 50.0, 5.0, 300.0),
        0.1,
        key="fB_khz",
    )
    Q_B = st.sidebar.slider(
        "Bus Q",
        10,
        1000,
        query_int("Q_B", 160, 10, 1000),
        10,
        key="Q_B",
    )
    if exact_model:
        st.sidebar.subheader("Exact circuit parameters")
        C_D_nf = st.sidebar.number_input(
            "Donor capacitance C_D (nF)", 1.0, 1000.0,
            query_float("C_D_nf", 100.0, 1.0, 1000.0), 0.1, key="C_D_nf")
        C_B_nf = st.sidebar.number_input(
            "Bus capacitance C_B (nF)", 1.0, 1000.0,
            query_float("C_B_nf", 100.0, 1.0, 1000.0), 0.1, key="C_B_nf")
        C_A_nf = st.sidebar.number_input(
            "Acceptor capacitance C_A (nF)", 1.0, 1000.0,
            query_float("C_A_nf", 103.2, 1.0, 1000.0), 0.1, key="C_A_nf")
        k_DB = st.sidebar.slider(
            "Coupling coefficient k_DB", -0.45, 0.45,
            query_float("k_DB", 0.160, -0.45, 0.45), 0.001, key="k_DB")
        k_BA = st.sidebar.slider(
            "Coupling coefficient k_BA", -0.45, 0.45,
            query_float("k_BA", 0.160, -0.45, 0.45), 0.001, key="k_BA")
        settings_to_sync.update({
            "C_D_nf": C_D_nf, "C_B_nf": C_B_nf, "C_A_nf": C_A_nf,
            "k_DB": k_DB, "k_BA": k_BA,
        })
    if collective_mode:
        N_D = st.sidebar.slider(
            "Number of coherent donors N_D",
            1,
            20,
            query_int("N_D", 2, 1, 20),
            1,
            key="N_D",
        )
        N_A = st.sidebar.slider(
            "Number of coherent acceptors N_A",
            1,
            20,
            query_int("N_A", 2, 1, 20),
            1,
            key="N_A",
        )
    else:
        N_D = 1
        N_A = 1
    g_D_khz = st.sidebar.slider(
        "Single-donor g_D / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_D_khz", 8.0, 0.0, 40.0),
        0.5,
        key="g_D_khz",
    )
    g_A_khz = st.sidebar.slider(
        "Single-acceptor g_A / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_A_khz", 8.0, 0.0, 40.0),
        0.5,
        key="g_A_khz",
    )

    settings_to_sync.update(
        {
            "fB_khz": fB_khz,
            "Q_B": Q_B,
            "g_D_khz": g_D_khz,
            "g_A_khz": g_A_khz,
        }
    )
    if collective_mode:
        settings_to_sync.update({"N_D": N_D, "N_A": N_A})

    omegaB = khz_to_omega(fB_khz)
    kappa_B = omegaB / Q_B
    g_D = khz_to_omega(g_D_khz)
    g_A = khz_to_omega(g_A_khz)
    Delta_B = omegaD - omegaB

    g_DB_bright = np.sqrt(N_D) * g_D
    g_BA_bright = np.sqrt(N_A) * g_A

    omegas = [omegaD, omegaB, omegaA]
    kappas = [kappa_D, kappa_B, kappa_A]
    G = [
        [0.0, g_DB_bright, 0.0],
        [g_DB_bright, 0.0, g_BA_bright],
        [0.0, g_BA_bright, 0.0],
    ]
    drive_vector = [np.sqrt(N_D) * drive_strength, 0.0, 0.0]

    if N_D == 1:
        donor_amp_label = "Donor amplitude"
        donor_energy_label = "Stored donor energy"
    else:
        donor_amp_label = "Donor bright-mode amplitude"
        donor_energy_label = "Total stored donor energy"

    if N_A == 1:
        acceptor_amp_label = "Acceptor amplitude"
        acceptor_energy_label = "Stored acceptor energy"
    else:
        acceptor_amp_label = "Acceptor bright-mode amplitude"
        acceptor_energy_label = "Total stored acceptor energy"

    amplitude_meta = [
        {"label": donor_amp_label, "indices": [0], "color_key": "donor"},
        {"label": "Bus amplitude", "indices": [1], "color_key": "bus"},
        {"label": acceptor_amp_label, "indices": [2], "color_key": "acceptor"},
    ]
    energy_meta = [
        {"label": donor_energy_label, "indices": [0], "color_key": "donor"},
        {"label": "Stored bus energy", "indices": [1], "color_key": "bus"},
        {"label": acceptor_energy_label, "indices": [2], "color_key": "acceptor"},
    ]

    drive_signal_for_plot = np.sqrt(N_D) * drive_strength * drive_carrier_unit

    detuning_ratio = np.inf
    if abs(Delta_B) > 1e-12:
        J_eff_hz = (g_DB_bright * g_BA_bright / Delta_B) / (2 * np.pi)
        detuning_ratio = max(abs(g_DB_bright), abs(g_BA_bright)) / abs(Delta_B)
        j_text = f"Large-detuning estimate: J_eff/2π ≈ {J_eff_hz / 1e3:.3f} kHz"
    else:
        J_eff_hz = np.nan
        j_text = "J estimate undefined at zero detuning"

    coupling_text = (
        f"g_DB/2π = {g_DB_bright / (2 * np.pi * 1e3):.1f} kHz, "
        f"g_BA/2π = {g_BA_bright / (2 * np.pi * 1e3):.1f} kHz, "
        f"{j_text}"
    )

    title = (
        f"Off-resonant bus-mediated transfer after {N_cycles}-cycle {f_drive_khz:.1f} kHz burst\n"
        f"N_D = {N_D}, N_A = {N_A}, f_B = {fB_khz:.0f} kHz, {coupling_text}"
    )

    if detuning_ratio > 0.1:
        scenario_warning = (
            f"g/|Δ_B| = {detuning_ratio:.3f}. The perturbative g²/Δ estimate may not be "
            "quantitatively reliable; use the exact three-mode eigenfrequencies."
        )


elif scenario == "b) Indirect coupling via resonant bus":
    scenario_notes = [
        ("markdown", "### b) Indirect coupling via resonant bus"),
        (
            "markdown",
            "This case keeps the bus as an explicit dynamical resonator that is on or near resonance "
            "with the donor and acceptor.",
        ),
        ("latex", r"D \leftrightarrow B \leftrightarrow A,\qquad g_{DB},\; g_{BA}"),
        ("latex", r"J_b = \frac{1}{2}\sqrt{g_D^2 + g_A^2}"),
    ]

    fB_khz = st.sidebar.slider(
        "Bare bus frequency f_B (kHz)",
        20.0,
        300.0,
        query_float("fB_res_khz", fD_khz, 20.0, 300.0),
        0.1,
        key="fB_res_khz",
    )
    Q_B = st.sidebar.slider(
        "Bus Q",
        10,
        1000,
        query_int("Q_B_res", 220, 10, 1000),
        10,
        key="Q_B_res",
    )
    g_DB_khz = st.sidebar.slider(
        "Donor-bus coupling g_DB / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_DB_res_khz", 3.0, 0.0, 40.0),
        0.1,
        key="g_DB_res_khz",
    )
    g_BA_khz = st.sidebar.slider(
        "Bus-acceptor coupling g_BA / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_BA_res_khz", 3.0, 0.0, 40.0),
        0.1,
        key="g_BA_res_khz",
    )

    settings_to_sync.update(
        {
            "fB_res_khz": fB_khz,
            "Q_B_res": Q_B,
            "g_DB_res_khz": g_DB_khz,
            "g_BA_res_khz": g_BA_khz,
        }
    )

    omegaB = khz_to_omega(fB_khz)
    kappa_B = omegaB / Q_B
    g_DB = khz_to_omega(g_DB_khz)
    g_BA = khz_to_omega(g_BA_khz)

    Delta_DB_hz = (omegaD - omegaB) / (2 * np.pi)
    Delta_AB_hz = (omegaA - omegaB) / (2 * np.pi)

    omegas = [omegaD, omegaB, omegaA]
    kappas = [kappa_D, kappa_B, kappa_A]
    G = [
        [0.0, g_DB, 0.0],
        [g_DB, 0.0, g_BA],
        [0.0, g_BA, 0.0],
    ]
    drive_vector = [drive_strength, 0.0, 0.0]

    amplitude_meta = [
        {"label": "Donor amplitude", "indices": [0], "color_key": "donor"},
        {"label": "Bus amplitude", "indices": [1], "color_key": "bus"},
        {"label": "Acceptor amplitude", "indices": [2], "color_key": "acceptor"},
    ]
    energy_meta = [
        {"label": "Stored donor energy", "indices": [0], "color_key": "donor"},
        {"label": "Stored bus energy", "indices": [1], "color_key": "bus"},
        {"label": "Stored acceptor energy", "indices": [2], "color_key": "acceptor"},
    ]

    drive_signal_for_plot = drive_strength * drive_carrier_unit

    J_res_khz = 0.5 * np.sqrt(g_DB_khz ** 2 + g_BA_khz ** 2)

    title = (
        f"Resonant/near-resonant bus transfer after {N_cycles}-cycle {f_drive_khz:.1f} kHz burst\n"
        f"f_B = {fB_khz:.0f} kHz, g_DB/2π = {g_DB_khz:.1f} kHz, "
        f"g_BA/2π = {g_BA_khz:.1f} kHz, J/2π = {J_res_khz:.1f} kHz"
    )

# Spectrum for the selected model.
if exact_model:
    L_matrix, C_matrix, C_inv, R_matrix, K_matrix = exact_rlc_matrices(
        [fD_khz, fB_khz, fA_khz], [C_D_nf, C_B_nf, C_A_nf],
        [Q_D, Q_B, Q_A], k_DB, k_BA,
    )
    eig_freqs_khz = exact_rlc_eigenfrequencies_khz(L_matrix, C_inv)
else:
    eig_freqs_khz = coupled_eigenfrequencies_khz(omegas, G)

if len(omegas) > 1:
    st.sidebar.header("Frequency sweep")
    auto_margin = max(5.0, 0.12 * (eig_freqs_khz[-1] - eig_freqs_khz[0]))
    sweep_start_khz = st.sidebar.number_input(
        "Sweep start frequency (kHz)", 1.0, 500.0,
        query_float("sweep_start_khz", max(1.0, eig_freqs_khz[0] - auto_margin), 1.0, 500.0),
        0.1, key="sweep_start_khz",
    )
    sweep_stop_khz = st.sidebar.number_input(
        "Sweep stop frequency (kHz)", 1.0, 500.0,
        query_float("sweep_stop_khz", min(500.0, eig_freqs_khz[-1] + auto_margin), 1.0, 500.0),
        0.1, key="sweep_stop_khz",
    )
    sweep_points = st.sidebar.slider(
        "Number of sweep points", 100, 4000,
        query_int("sweep_points", 1000, 100, 4000), 100, key="sweep_points",
    )
    settings_to_sync.update({
        "sweep_start_khz": sweep_start_khz, "sweep_stop_khz": sweep_stop_khz,
        "sweep_points": sweep_points,
    })

st.sidebar.header("Plot settings")
total_time_max_us = float(t[-1] * 1e6)
min_plot_max_us = max(float(duration * 1e6), 10.0)
default_plot_max_us = min(total_time_max_us, 900.0)
if default_plot_max_us < min_plot_max_us:
    default_plot_max_us = min_plot_max_us

x_axis_max_us = st.sidebar.slider(
    "x-axis time maximum (µs)",
    min_value=min_plot_max_us,
    max_value=total_time_max_us,
    value=query_float("x_axis_max_us", default_plot_max_us, min_plot_max_us, total_time_max_us),
    step=10.0,
    key="x_axis_max_us",
)
plot_width_px = st.sidebar.slider(
    "Plot width (px)",
    600,
    2000,
    query_int("plot_width_px", 1100, 600, 2000),
    50,
    key="plot_width_px",
)
plot_height_px = st.sidebar.slider(
    "Plot height (px)",
    400,
    1400,
    query_int("plot_height_px", 760, 400, 1400),
    20,
    key="plot_height_px",
)
font_scale = st.sidebar.slider(
    "Plot font size scaling factor",
    0.5,
    2.5,
    query_float("font_scale", 1.0, 0.5, 2.5),
    0.05,
    key="font_scale",
)

settings_to_sync.update(
    {
        "x_axis_max_us": x_axis_max_us,
        "plot_width_px": plot_width_px,
        "plot_height_px": plot_height_px,
        "font_scale": font_scale,
    }
)

# Run the selected time-domain backend.
if exact_model:
    q_exact, i_exact = rk4_exact_rlc(
        t, L_matrix, C_inv, R_matrix, drive_carrier_unit, drive_strength
    )
else:
    a = rk4_coupled_modes(
        t=t, omegas=omegas, kappas=kappas, G=G, drive_vector=drive_vector,
        drive_envelope=drive_envelope, omega_ref=omega_ref,
    )

scenario_image_paths = {
    "Donor preparation": Path("assets/scenario_0.png"),
    "a) Direct coupling": Path("assets/scenario_a.png"),
    "b) Indirect coupling via resonant bus": Path("assets/scenario_b.png"),
    "c) Indirect coupling via off-resonant bus": Path("assets/scenario_c.png"),
    "d) Indirect coupling via off-resonant bus with collective enhancement": Path("assets/scenario_d.png"),
}
st.image(str(scenario_image_paths[scenario]))

st.caption(
    f"Dynamic parameters: f_drive = {f_drive_khz:.1f} kHz; "
    f"bare f_D = {fD_khz:.2f} kHz; bare f_A = {fA_khz:.2f} kHz"
    + (f"; bare f_B = {fB_khz:.2f} kHz" if len(omegas) == 3 else "")
)

if len(omegas) > 1:
    modes_text = " | ".join(f"{frequency:.3f} kHz" for frequency in eig_freqs_khz)
    st.subheader("Exact mutually coupled circuit eigenfrequencies" if exact_model else "Coupled normal modes")
    st.info(modes_text)

    if sweep_stop_khz <= sweep_start_khz:
        st.warning("Sweep stop must be above sweep start; the plot uses a 0.1 kHz span.")
        sweep_stop_for_plot = sweep_start_khz + 0.1
    else:
        sweep_stop_for_plot = sweep_stop_khz
    sweep_frequencies = np.linspace(sweep_start_khz, sweep_stop_for_plot, sweep_points)
    if exact_model:
        sweep_response = exact_rlc_sweep(
            sweep_frequencies, L_matrix, C_matrix, C_inv, R_matrix,
            [drive_strength, 0.0, 0.0],
        )
    else:
        sweep_response = steady_state_sweep(
            sweep_frequencies, omegas, kappas, G, drive_vector
        )
    sweep_magnitudes = np.abs(sweep_response)
    sweep_norm = max(float(np.max(sweep_magnitudes)), 1e-30)
    sweep_magnitudes /= sweep_norm

    sweep_fig, sweep_ax = plt.subplots(figsize=(plot_width_px / 100, 3.8), dpi=100)
    sweep_ax.plot(sweep_frequencies, sweep_magnitudes[:, 0],
                  label="|V_D|" if exact_model else "|a_D|", color=COLORS["donor"], linewidth=2)
    if len(omegas) == 3:
        sweep_ax.plot(sweep_frequencies, sweep_magnitudes[:, 1],
                      label="|V_B|" if exact_model else "|a_B|", color=COLORS["bus"], linewidth=1.5)
        acceptor_index = 2
    else:
        acceptor_index = 1
    sweep_ax.plot(sweep_frequencies, sweep_magnitudes[:, acceptor_index],
                  label="|V_A|" if exact_model else "|a_A|", color=COLORS["acceptor"], linewidth=2)
    for mode_frequency in eig_freqs_khz:
        sweep_ax.axvline(mode_frequency, color="gray", linestyle=":", linewidth=0.9)
    sweep_ax.set(title="Normalized steady-state frequency response",
                 xlabel="Drive frequency (kHz)", ylabel="Relative amplitude")
    sweep_ax.grid(alpha=0.2)
    sweep_ax.legend()
    sweep_fig.tight_layout()
    st.pyplot(sweep_fig, width="content")
    plt.close(sweep_fig)

if exact_model:
    st.subheader("Exact circuit matrices")
    matrix_labels = ["D", "B", "A"]
    col_l, col_c = st.columns(2)
    with col_l:
        st.markdown("**L [µH]**")
        st.dataframe(pd.DataFrame(L_matrix * 1e6, index=matrix_labels, columns=matrix_labels).style.format("{:.3f}"))
        st.markdown("**R [Ω]**")
        st.dataframe(pd.DataFrame(R_matrix, index=matrix_labels, columns=matrix_labels).style.format("{:.4f}"))
    with col_c:
        st.markdown("**C [nF]**")
        st.dataframe(pd.DataFrame(C_matrix * 1e9, index=matrix_labels, columns=matrix_labels).style.format("{:.3f}"))
        st.markdown("**K (dimensionless)**")
        st.dataframe(pd.DataFrame(K_matrix, index=matrix_labels, columns=matrix_labels).style.format("{:.4f}"))
    heat_fig, heat_ax = plt.subplots(figsize=(5.2, 4.2), dpi=100)
    L_microhenry = L_matrix * 1e6
    color_limit = max(float(np.max(np.abs(L_microhenry))), 1e-12)
    image_handle = heat_ax.imshow(L_microhenry, cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    heat_ax.set_xticks(range(3), matrix_labels)
    heat_ax.set_yticks(range(3), matrix_labels)
    heat_ax.set_title("Inductance matrix L [µH]")
    for row in range(3):
        for column in range(3):
            heat_ax.text(column, row, f"{L_microhenry[row, column]:.2f}",
                         ha="center", va="center", color="black")
    heat_fig.colorbar(image_handle, ax=heat_ax, label="µH (zero-centered)")
    heat_fig.tight_layout()
    st.pyplot(heat_fig, width="content")
    plt.close(heat_fig)
    st.caption(
        "Local self-energy curves exclude mutual coupling energy, so they are diagnostics rather "
        "than a unique decomposition of total physical energy. Experimental bus amplitude should "
        "be compared by timing and waveform shape, not by absolute amplitude, unless the pickup is calibrated."
    )
    pair_db = exact_rlc_eigenfrequencies_khz(
        L_matrix[np.ix_([0, 1], [0, 1])], C_inv[np.ix_([0, 1], [0, 1])]
    )
    pair_ba = exact_rlc_eigenfrequencies_khz(
        L_matrix[np.ix_([1, 2], [1, 2])], C_inv[np.ix_([1, 2], [1, 2])]
    )
    st.markdown(
        f"**Derived pairwise lossless modes:** D–B = {pair_db[0]:.3f} / {pair_db[1]:.3f} kHz; "
        f"B–A = {pair_ba[0]:.3f} / {pair_ba[1]:.3f} kHz."
    )

if scenario == "c) Indirect coupling via off-resonant bus" and not exact_model:
    endpoint_splitting_khz = eig_freqs_khz[-1] - eig_freqs_khz[-2]
    beat_period_us = 1000.0 / endpoint_splitting_khz
    half_swap_us = 500.0 / endpoint_splitting_khz
    st.subheader("Off-resonant-bus diagnostics")
    st.markdown(
        f"""
| Quantity | Value |
|---|---:|
| Bare frequencies (D / B / A) | {fD_khz:.3f} / {fB_khz:.3f} / {fA_khz:.3f} kHz |
| Drive carrier; burst | {f_drive_khz:.3f} kHz; {N_cycles} cycles; {duration * 1e6:.3f} µs |
| Couplings g_DB/2π; g_BA/2π | {g_DB_bright / (2*np.pi*1e3):.3f}; {g_BA_bright / (2*np.pi*1e3):.3f} kHz |
| Coupled eigenfrequencies | {modes_text} |
| Endpoint-like splitting f₃ − f₂ | {endpoint_splitting_khz:.3f} kHz |
| Spectral beat period 1/Δf | {beat_period_us:.3f} µs |
| Spectral half-swap timescale 1/(2Δf) | {half_swap_us:.3f} µs |
| Large-detuning J_eff/2π estimate | {J_eff_hz / 1e3:.3f} kHz |
| Perturbative diagnostic g / abs(Δ_B) | {detuning_ratio:.3f} |
"""
    )
    st.caption(
        "The half-swap value is a spectral timescale, not a prediction that must equal the "
        "exact donor-minimum time in this lossy three-mode simulation. Experimental pickup-bus "
        "amplitude should be compared by timing and shape, not absolute scale."
    )

if exact_model:
    fig = build_exact_rlc_plot(
        t, q_exact, i_exact, C_matrix, L_matrix, drive_carrier_unit, drive_strength, duration,
        f"Exact mutual-RLC response after {N_cycles}-cycle {f_drive_khz:.2f} kHz burst",
        x_axis_max_us, energy_norm_mode, observable_mode, plot_width_px, plot_height_px,
    )
else:
    fig = build_plot(
        t=t, a=a, amplitude_meta=amplitude_meta, energy_meta=energy_meta,
        kappas=kappas, drive_vector=drive_vector, drive_envelope=drive_envelope,
        drive_signal_for_plot=drive_signal_for_plot, omega_ref=omega_ref,
        duration=duration, title=title, x_axis_max_us=x_axis_max_us,
        plot_width_px=plot_width_px, plot_height_px=plot_height_px,
        font_scale=font_scale, energy_norm_mode=energy_norm_mode,
        observable_mode=observable_mode,
    )

# Render to an image and display with an explicit pixel width.
# This avoids Streamlit stretching the plot to fill the full main-area width.
buf = BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
buf.seek(0)
st.image(buf, width=plot_width_px)
plt.close(fig)

if scenario_warning:
    st.warning(scenario_warning)

for note_type, note_body in scenario_notes:
    if note_type == "latex":
        st.latex(note_body)
    else:
        st.markdown(note_body)

# Keep the URL synchronized with the current UI state.
sync_query_params(settings_to_sync)

with st.expander("Model equations and notes"):
    st.write("The app uses rotating-frame coupled-mode equations:")

    st.latex(r"""
    \dot a_i =
    -\left(\frac{\kappa_i}{2}+i\delta_i\right)a_i
    -i\sum_j G_{ij}a_j
    +s_i(t)
    """)

    st.write("where")

    st.latex(r"""
    \delta_i=\omega_i-\omega_{\mathrm{ref}},
    \qquad
    \kappa_i=\frac{\omega_i}{Q_i}.
    """)

    st.write("The displayed mode-amplitude traces reconstruct a physical oscillatory signal using")

    st.latex(r"""
    x_i(t)\propto
    \mathrm{Re}\!\left[a_i(t)e^{-i\omega_{\mathrm{ref}}t}\right].
    """)

    st.write(
        "The drive waveform is shown on a secondary y-axis in the amplitude plot. "
        "The drive trace is normalized to its own peak amplitude, so it uses the full range "
        "of that secondary axis without being forced onto the same scale as the mode amplitudes."
    )

    st.write("The energy-like quantities are")

    st.latex(r"""
    E_i(t)=|a_i(t)|^2.
    """)

    st.write("The cumulative drive work is")

    st.latex(r"""
    W_{\mathrm{drive}}(t)
    =
    \int
    2\,\mathrm{Re}\!\left[a_i^*(t)s_i(t)\right]\,dt,
    """)

    st.write(
        "summed over driven modes. The cumulative dissipated energy is computed internally as"
    )

    st.latex(r"""
    E_{\mathrm{loss}}(t)
    =
    \int
    \sum_i \kappa_i |a_i(t)|^2\,dt.
    """)

    st.write("Instead of plotting the dissipated energy, the app plots the undissipated delivered energy:")

    st.latex(r"""
    E_{\mathrm{undissipated}}(t)
    =
    W_{\mathrm{drive}}(t)-E_{\mathrm{loss}}(t).
    """)

    st.write(
        "Up to numerical integration error, this equals the sum of the stored energies "
        "in the displayed modes."
    )

    st.write("Color convention:")
    st.markdown(
        """
        - drive/excitation: blue from the schematic
        - donor: orange-brown from the schematic
        - bus: black from the schematic
        - acceptor: green from the schematic
        - total undissipated delivered energy: gray auxiliary curve
        """
    )

    st.write("Number of time steps:", len(t))
    st.write("Drive duration (µs):", duration * 1e6)
    st.write("Displayed x-axis maximum (µs):", x_axis_max_us)
    st.write("Plot size (px):", f"{plot_width_px} × {plot_height_px}")
    st.write("Plot font scale:", font_scale)
    st.write("URL sync:", "All active page settings are written into the URL query parameters.")
