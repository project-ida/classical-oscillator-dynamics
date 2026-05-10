
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from io import BytesIO

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
def khz_to_omega(f_khz: float) -> float:
    return 2 * np.pi * f_khz * 1e3

def hz_to_omega(f_hz: float) -> float:
    return 2 * np.pi * f_hz

def make_drive(t, omega_ref, duration):
    """Smooth sin^2-envelope burst."""
    envelope = np.zeros_like(t)
    inside = (t >= 0) & (t <= duration)
    if np.any(inside):
        tau = t[inside] / duration
        envelope[inside] = np.sin(np.pi * tau) ** 2
    carrier = np.cos(omega_ref * t)
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

    norm = np.max(W_drive)
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
    ax_amp.set_ylim(-1.15, 1.15)
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
    ax_drive.set_ylim(-1.05, 1.05)

    # Merge legends
    lines1, labels1 = ax_amp.get_legend_handles_labels()
    lines2, labels2 = ax_drive.get_legend_handles_labels()
    ax_amp.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=legend_fs)

    ax_amp.text(duration * 1e6 / 2, 1.03, "drive on", ha="center", fontsize=annotation_fs)
    ax_amp.text(
        duration * 1e6 + 0.12 * (t[-1] - t[0]) * 1e6,
        1.03,
        "free evolution",
        ha="center",
        fontsize=annotation_fs,
    )

    # -----------------------------
    # Energy subplot
    # -----------------------------
    ax_energy.plot(
        t * 1e6,
        W_drive_n,
        linewidth=2.0,
        label="Cumulative energy delivered by drive",
        color=COLORS["drive"],
    )

    for item in energy_meta:
        curve = E_modes[:, item["indices"]].sum(axis=1) / norm
        ax_energy.plot(
            t * 1e6,
            curve,
            linewidth=2.0,
            label=item["label"],
            color=COLORS[item["color_key"]],
        )

    ax_energy.plot(
        t * 1e6,
        E_undissipated_n,
        linewidth=2.0,
        linestyle="--",
        label="Total undissipated delivered energy",
        color=COLORS["undissipated"],
    )

    ax_energy.axvline(0, linestyle="--", linewidth=1, color="gray")
    ax_energy.axvline(duration * 1e6, linestyle="--", linewidth=1, color="gray")
    ax_energy.set_xlabel("Time (µs)", fontsize=label_fs)
    ax_energy.set_ylabel("Energy / final delivered drive energy", fontsize=label_fs)
    ax_energy.set_xlim(t[0] * 1e6, x_axis_max_us)
    ax_energy.set_ylim(-0.03, 1.08)
    ax_energy.tick_params(axis="both", labelsize=tick_fs)
    ax_energy.legend(loc="upper right", fontsize=legend_fs)
    ax_energy.text(duration * 1e6 / 2, 1.02, "drive on", ha="center", fontsize=annotation_fs)
    ax_energy.text(
        duration * 1e6 + 0.12 * (t[-1] - t[0]) * 1e6,
        1.02,
        "drive off",
        ha="center",
        fontsize=annotation_fs,
    )

    plt.tight_layout()
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
    raw = _query_value(name)
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
    raw = _query_value(name)
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
    raw = _query_value(name)
    return raw if raw in options else default

def query_float_option(name, options, default):
    raw = _query_value(name)
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
    for key, value in settings.items():
        formatted = format_query_value(value)
        if _query_value(key) != formatted:
            st.query_params[key] = formatted

# -----------------------------
# App UI
# -----------------------------
st.title("LC Transfer Analog Simulator")
st.caption(
    "Interactive classical LC analogs of donor excitation, direct transfer, off-resonant bus-mediated "
    "transfer, collective bright-mode enhancement, and resonant-bus transfer."
)

scenario_options = [
    "1. Excite donor only",
    "2. Direct donor → acceptor transfer",
    "3. Donor → acceptor via off-resonant bus (single or collective)",
    "4. Donor → acceptor via resonant/near-resonant bus",
]

scenario_default = query_choice("scenario", scenario_options, scenario_options[0])
scenario = st.sidebar.selectbox(
    "Scenario",
    scenario_options,
    index=scenario_options.index(scenario_default),
)

settings_to_sync = {"scenario": scenario}

st.sidebar.header("Shared parameters")
fD_khz = st.sidebar.slider(
    "Donor frequency f_D (kHz)",
    20.0,
    300.0,
    query_float("fD_khz", 100.0, 20.0, 300.0),
    1.0,
)
fA_khz = st.sidebar.slider(
    "Acceptor frequency f_A (kHz)",
    20.0,
    300.0,
    query_float("fA_khz", 100.0, 20.0, 300.0),
    1.0,
)
N_cycles = st.sidebar.slider(
    "Drive burst length (cycles)",
    1,
    20,
    query_int("N_cycles", 5, 1, 20),
    1,
)
Q_D = st.sidebar.slider(
    "Donor Q",
    10,
    1000,
    query_int("Q_D", 220, 10, 1000),
    10,
)
Q_A = st.sidebar.slider(
    "Acceptor Q",
    10,
    1000,
    query_int("Q_A", 220, 10, 1000),
    10,
)
drive_strength = st.sidebar.slider(
    "Drive strength (arb.)",
    1e4,
    5e5,
    query_float("drive_strength", 1.2e5, 1e4, 5e5),
    1e4,
    format="%.0f",
)
t_pre_us = st.sidebar.slider(
    "Time before burst (µs)",
    0.0,
    50.0,
    query_float("t_pre_us", 10.0, 0.0, 50.0),
    1.0,
)
t_post_us = st.sidebar.slider(
    "Time after burst (µs)",
    100.0,
    3000.0,
    query_float("t_post_us", 850.0, 100.0, 3000.0),
    50.0,
)
dt_options = [0.01, 0.02, 0.05, 0.1, 0.2]
dt_us = st.sidebar.select_slider(
    "Time step (µs)",
    options=dt_options,
    value=query_float_option("dt_us", dt_options, 0.05),
)

settings_to_sync.update(
    {
        "fD_khz": fD_khz,
        "fA_khz": fA_khz,
        "N_cycles": N_cycles,
        "Q_D": Q_D,
        "Q_A": Q_A,
        "drive_strength": drive_strength,
        "t_pre_us": t_pre_us,
        "t_post_us": t_post_us,
        "dt_us": dt_us,
    }
)

omegaD = khz_to_omega(fD_khz)
omegaA = khz_to_omega(fA_khz)
omega_ref = omegaD

duration = N_cycles / (fD_khz * 1e3)
t = np.arange(-t_pre_us * 1e-6, duration + t_post_us * 1e-6, dt_us * 1e-6)
drive_envelope, drive_carrier_unit = make_drive(t, omega_ref, duration)

kappa_D = omegaD / Q_D
kappa_A = omegaA / Q_A

if scenario == "1. Excite donor only":
    st.markdown("### Scenario 1: donor excitation only")

    st.write(
        "The drive prepares the donor LC excitation. The stored donor energy is the closest "
        "classical analog of a quantum-state occupation probability."
    )

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

    title = f"Donor-only excitation after {N_cycles}-cycle {fD_khz:.0f} kHz burst"

elif scenario == "2. Direct donor → acceptor transfer":
    # Direct coupling in kHz. For backward compatibility, old URLs using
    # J_hz are still read and converted if J_khz is not present.
    J_khz_default = query_float("J_hz", 3000.0, 0.0, 20000.0) / 1000.0
    J_khz = st.sidebar.slider(
        "Direct coupling J_a / 2π (kHz)",
        0.0,
        20.0,
        query_float("J_khz", J_khz_default, 0.0, 20.0),
        0.1,
    )
    settings_to_sync["J_khz"] = J_khz

    J = khz_to_omega(J_khz)

    st.markdown("### Scenario 2: direct donor–acceptor transfer")

    st.latex(r"J_a = g_{DA}(x)")

    st.write("In the weak/lossy transfer-rate limit,")

    st.latex(r"\Gamma_a \sim \frac{|g_{DA}(x)|^2}{\kappa}")

    st.write(
        "This simulation keeps the coherent dynamics explicitly, so visible energy sloshing "
        "appears when the coupling is strong enough."
    )

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
        f"Direct donor-acceptor transfer after {N_cycles}-cycle {fD_khz:.0f} kHz burst\n"
        f"J_a/2π = {J_khz:.1f} kHz"
    )

elif scenario == "3. Donor → acceptor via off-resonant bus (single or collective)":
    st.markdown("### Scenario 3: donor–acceptor transfer through an off-resonant bus")

    st.write(
        "This combines the single-donor/single-acceptor off-resonant bus case and the "
        "collective-enhancement off-resonant bus case into one generalized scenario."
    )

    st.markdown(
        """
        - If **N_D = N_A = 1**, this reduces to the ordinary single-donor/single-acceptor bus case.
        - If either **N_D > 1** or **N_A > 1**, the simulation represents the corresponding
          donor and acceptor **bright modes** coupled through the bus.
        """
    )

    st.write("In the large-detuning limit,")

    st.latex(r"""
    J_{\rm single} \simeq \frac{g_D g_A}{\Delta_B},
    \qquad
    J_{\rm bright} \simeq \sqrt{N_D N_A}\frac{g_D g_A}{\Delta_B}.
    """)

    fB_khz = st.sidebar.slider(
        "Bus frequency f_B (kHz)",
        5.0,
        300.0,
        query_float("fB_khz", 50.0, 5.0, 300.0),
        1.0,
    )
    Q_B = st.sidebar.slider(
        "Bus Q",
        10,
        1000,
        query_int("Q_B", 160, 10, 1000),
        10,
    )
    N_D = st.sidebar.slider(
        "Number of coherent donors N_D",
        1,
        20,
        query_int("N_D", 1, 1, 20),
        1,
    )
    N_A = st.sidebar.slider(
        "Number of coherent acceptors N_A",
        1,
        20,
        query_int("N_A", 1, 1, 20),
        1,
    )
    g_D_khz = st.sidebar.slider(
        "Single-donor g_D / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_D_khz", 8.0, 0.0, 40.0),
        0.5,
    )
    g_A_khz = st.sidebar.slider(
        "Single-acceptor g_A / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_A_khz", 8.0, 0.0, 40.0),
        0.5,
    )

    settings_to_sync.update(
        {
            "fB_khz": fB_khz,
            "Q_B": Q_B,
            "N_D": N_D,
            "N_A": N_A,
            "g_D_khz": g_D_khz,
            "g_A_khz": g_A_khz,
        }
    )

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

    if abs(Delta_B) > 1e-12:
        J_single_hz = (g_D * g_A / Delta_B) / (2 * np.pi)
        J_bright_hz = (np.sqrt(N_D * N_A) * g_D * g_A / Delta_B) / (2 * np.pi)
        coupling_text = (
            f"J_single/2π ≈ {J_single_hz:.0f} Hz, "
            f"J_bright/2π ≈ {J_bright_hz:.0f} Hz"
        )
    else:
        coupling_text = "Effective-coupling estimate undefined at zero detuning"

    title = (
        f"Off-resonant bus-mediated donor-acceptor transfer after {N_cycles}-cycle {fD_khz:.0f} kHz burst\n"
        f"N_D = {N_D}, N_A = {N_A}, f_B = {fB_khz:.0f} kHz, {coupling_text}"
    )

    if abs(Delta_B / (2 * np.pi)) < 5e3:
        st.warning("The bus is close to resonance. The large-detuning estimate for J is not reliable here.")


else:
    st.markdown("### Scenario 4: donor–acceptor transfer through a resonant or near-resonant bus")

    st.write(
        "This case keeps the bus as an explicit dynamical resonator that is on or near resonance "
        "with the donor and/or acceptor. Unlike the off-resonant case, the bus can become "
        "substantially populated, so the simple virtual-bus estimate is not the right picture."
    )

    st.write("The relevant coherent couplings are")

    st.latex(r"""
    D \leftrightarrow B \leftrightarrow A,
    \qquad
    g_{DB},\; g_{BA},
    """)

    st.write("with detunings")

    st.latex(r"""
    \Delta_{DB} = \omega_D-\omega_B,
    \qquad
    \Delta_{AB} = \omega_A-\omega_B.
    """)

    fB_khz = st.sidebar.slider(
        "Bus frequency f_B (kHz)",
        20.0,
        300.0,
        query_float("fB_res_khz", fD_khz, 20.0, 300.0),
        1.0,
    )
    Q_B = st.sidebar.slider(
        "Bus Q",
        10,
        1000,
        query_int("Q_B_res", 220, 10, 1000),
        10,
    )
    g_DB_khz = st.sidebar.slider(
        "Donor-bus coupling g_DB / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_DB_res_khz", 3.0, 0.0, 40.0),
        0.1,
    )
    g_BA_khz = st.sidebar.slider(
        "Bus-acceptor coupling g_BA / 2π (kHz)",
        0.0,
        40.0,
        query_float("g_BA_res_khz", 3.0, 0.0, 40.0),
        0.1,
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

    title = (
        f"Resonant/near-resonant bus transfer after {N_cycles}-cycle {fD_khz:.0f} kHz burst\n"
        f"f_B = {fB_khz:.0f} kHz, Δ_DB/2π = {Delta_DB_hz:.0f} Hz, "
        f"Δ_AB/2π = {Delta_AB_hz:.0f} Hz"
    )

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
)
plot_width_px = st.sidebar.slider(
    "Plot width (px)",
    600,
    2000,
    query_int("plot_width_px", 1100, 600, 2000),
    50,
)
plot_height_px = st.sidebar.slider(
    "Plot height (px)",
    400,
    1400,
    query_int("plot_height_px", 760, 400, 1400),
    20,
)
font_scale = st.sidebar.slider(
    "Plot font size scaling factor",
    0.5,
    2.5,
    query_float("font_scale", 1.0, 0.5, 2.5),
    0.05,
)

settings_to_sync.update(
    {
        "x_axis_max_us": x_axis_max_us,
        "plot_width_px": plot_width_px,
        "plot_height_px": plot_height_px,
        "font_scale": font_scale,
    }
)

# Run simulation
a = rk4_coupled_modes(
    t=t,
    omegas=omegas,
    kappas=kappas,
    G=G,
    drive_vector=drive_vector,
    drive_envelope=drive_envelope,
    omega_ref=omega_ref,
)

fig = build_plot(
    t=t,
    a=a,
    amplitude_meta=amplitude_meta,
    energy_meta=energy_meta,
    kappas=kappas,
    drive_vector=drive_vector,
    drive_envelope=drive_envelope,
    drive_signal_for_plot=drive_signal_for_plot,
    omega_ref=omega_ref,
    duration=duration,
    title=title,
    x_axis_max_us=x_axis_max_us,
    plot_width_px=plot_width_px,
    plot_height_px=plot_height_px,
    font_scale=font_scale,
)

# Render to an image and display with an explicit pixel width.
# This avoids Streamlit stretching the plot to fill the full main-area width.
buf = BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
buf.seek(0)
st.image(buf, width=plot_width_px)
plt.close(fig)

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
