import numpy as np

from oscillator_model import (
    burst_duration,
    coupled_eigenfrequencies_khz,
    khz_to_omega,
    legacy_drive_frequency,
)


def test_independent_drive_carrier_controls_frame_and_duration():
    donor_khz, drive_khz = 138.0, 142.0
    assert khz_to_omega(donor_khz) != khz_to_omega(drive_khz)
    assert burst_duration(3, drive_khz) == 3 / 142_000


def test_eigenfrequencies_match_eigvalsh():
    bare = np.array([100.0, 130.0, 140.0])
    coupling = np.array([[0.0, 4.0, 0.0], [4.0, 0.0, 6.0], [0.0, 6.0, 0.0]])
    omegas = 2 * np.pi * 1e3 * bare
    G = 2 * np.pi * 1e3 * coupling
    expected = np.linalg.eigvalsh(np.diag(omegas) + G) / (2 * np.pi * 1e3)
    np.testing.assert_allclose(coupled_eigenfrequencies_khz(omegas, G), expected)


def test_no_coupling_returns_sorted_bare_frequencies():
    bare = np.array([138.61, 103.5, 138.61])
    result = coupled_eigenfrequencies_khz(2 * np.pi * 1e3 * bare, np.zeros((3, 3)))
    np.testing.assert_allclose(result, np.sort(bare))


def test_old_url_defaults_drive_to_donor():
    assert legacy_drive_frequency(None, 138.0) == 138.0
    assert legacy_drive_frequency("142.1", 138.0) == 142.1
