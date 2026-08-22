"""Numerical primitives shared by the Streamlit UI and lightweight tests."""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, hilbert, sosfiltfilt


def khz_to_omega(frequency_khz: float) -> float:
    return 2 * np.pi * frequency_khz * 1e3


def burst_duration(cycles: int, drive_frequency_khz: float) -> float:
    return cycles / (drive_frequency_khz * 1e3)


def legacy_drive_frequency(drive_query_value, donor_frequency_khz: float) -> float:
    """Use the donor carrier only for old URLs that omit the drive parameter."""
    return donor_frequency_khz if drive_query_value is None else float(drive_query_value)


def coupled_eigenfrequencies_khz(omegas, coupling_matrix):
    H = np.diag(np.asarray(omegas, dtype=float)) + np.asarray(coupling_matrix, dtype=float)
    return np.linalg.eigvalsh(H) / (2 * np.pi * 1e3)


def steady_state_sweep(frequencies_khz, omegas, kappas, coupling_matrix, drive_vector):
    omegas = np.asarray(omegas, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    coupling_matrix = np.asarray(coupling_matrix, dtype=float)
    drive_vector = np.asarray(drive_vector, dtype=complex)
    response = np.empty((len(frequencies_khz), len(omegas)), dtype=complex)
    for i, frequency_khz in enumerate(frequencies_khz):
        omega = khz_to_omega(frequency_khz)
        M = np.diag(kappas / 2 + 1j * (omegas - omega)) + 1j * coupling_matrix
        response[i] = np.linalg.solve(M, drive_vector)
    return response


def exact_rlc_matrices(frequencies_khz, capacitances_nf, quality_factors, k_db, k_ba, k_da=0.0):
    frequencies_hz = np.asarray(frequencies_khz, dtype=float) * 1e3
    C_values = np.asarray(capacitances_nf, dtype=float) * 1e-9
    omegas = 2 * np.pi * frequencies_hz
    L_values = 1 / (omegas**2 * C_values)
    L = np.diag(L_values)
    L[0, 1] = L[1, 0] = k_db * np.sqrt(L_values[0] * L_values[1])
    L[1, 2] = L[2, 1] = k_ba * np.sqrt(L_values[1] * L_values[2])
    L[0, 2] = L[2, 0] = k_da * np.sqrt(L_values[0] * L_values[2])
    C = np.diag(C_values)
    C_inv = np.diag(1 / C_values)
    R = np.diag(omegas * L_values / np.asarray(quality_factors, dtype=float))
    K = L / np.sqrt(np.outer(L_values, L_values))
    np.fill_diagonal(K, 0.0)
    if np.min(np.linalg.eigvalsh(L)) <= 0:
        raise ValueError("The inductance matrix must be positive definite; reduce coupling coefficients.")
    return L, C, C_inv, R, K


def exact_rlc_eigenfrequencies_khz(L, C_inv):
    omega_squared = eigh(C_inv, L, eigvals_only=True)
    return np.sqrt(np.maximum(omega_squared, 0)) / (2 * np.pi * 1e3)


def exact_rlc_sweep(frequencies_khz, L, C, C_inv, R, drive_vector):
    response = np.empty((len(frequencies_khz), 3), dtype=complex)
    drive_vector = np.asarray(drive_vector, dtype=complex)
    capacitances = np.diag(C)
    for index, frequency_khz in enumerate(frequencies_khz):
        omega = khz_to_omega(frequency_khz)
        Z = R + 1j * omega * L + C_inv / (1j * omega)
        currents = np.linalg.solve(Z, drive_vector)
        response[index] = currents / (1j * omega * capacitances)
    return response


def rk4_exact_rlc(t, L, C_inv, R, drive_waveform, drive_strength):
    """Integrate charge/current state for the exact mutual-inductance circuit."""
    n = L.shape[0]
    state = np.zeros((len(t), 2 * n), dtype=float)
    L_inv = np.linalg.inv(L)

    def derivative(time, y):
        q, current = y[:n], y[n:]
        drive = np.zeros(n)
        drive[0] = drive_strength * np.interp(time, t, drive_waveform)
        return np.concatenate((current, L_inv @ (drive - R @ current - C_inv @ q)))

    for index in range(len(t) - 1):
        y, h = state[index], t[index + 1] - t[index]
        k1 = derivative(t[index], y)
        k2 = derivative(t[index] + h / 2, y + h * k1 / 2)
        k3 = derivative(t[index] + h / 2, y + h * k2 / 2)
        k4 = derivative(t[index] + h, y + h * k3)
        state[index + 1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return state[:, :n], state[:, n:]


def experimental_energy_proxy(t, voltages, capacitances, low_khz=125.0, high_khz=160.0, smooth_khz=18.0):
    """Replicate the experimental band-pass, Hilbert-envelope energy analysis."""
    sample_rate = 1 / np.median(np.diff(t))
    nyquist = sample_rate / 2
    high_hz = min(high_khz * 1e3, 0.95 * nyquist)
    low_hz = min(low_khz * 1e3, 0.8 * high_hz)
    band = butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    smooth = butter(2, min(smooth_khz * 1e3, 0.9 * nyquist), btype="lowpass", fs=sample_rate, output="sos")
    proxies = np.empty_like(voltages)
    for mode in range(voltages.shape[1]):
        filtered = sosfiltfilt(band, voltages[:, mode])
        envelope = np.abs(hilbert(filtered))
        envelope = sosfiltfilt(smooth, envelope)
        proxies[:, mode] = capacitances[mode] * envelope**2
    return proxies
