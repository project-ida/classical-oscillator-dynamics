"""Numerical primitives shared by the Streamlit UI and lightweight tests."""

import numpy as np


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
