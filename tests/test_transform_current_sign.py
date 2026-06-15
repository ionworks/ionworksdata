"""Tests for ``set_positive_current_for_discharge`` sign handling.

Canonical convention (see ``validate_positive_current_is_discharge`` /
``positive_current_is_charge``): **positive current = discharge, negative =
charge**.

These tests cover the half-cell sign-inversion bug fix:

- The voltage-response heuristic assumes OCV *increases* with state-of-charge.
  That holds for full cells and positive-electrode (cathode) half-cells, but
  is FALSE for negative-electrode (anode) half-cells (graphite/Si vs Li),
  whose OCV *falls* on charge (lithiation). The heuristic therefore silently
  inverts the sign for anode half-cells.
- The fix adds a ``direction`` option so the caller can declare the known
  operation and bypass the ambiguous heuristic, and emits a warning when the
  heuristic is relied upon with low confidence.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import ionworksdata as iw


def _unsigned_current(n, magnitude=6.0):
    """Constant-magnitude unsigned current with a leading rest point."""
    current = np.full(n, magnitude, dtype=float)
    current[0] = 0.0
    return current


def _nonrest_mean(df, current_col="Current [A]"):
    col = df[current_col]
    return col.filter(col.abs() > 1e-9).mean()


def test_anode_half_cell_direction_override_makes_charge_negative():
    """Anode half-cell CHARGE (V falls) with ``direction='charge'`` override.

    The override must bypass the voltage heuristic and sign the (unsigned)
    current as charge -> negative, which the heuristic alone gets wrong for
    anode half-cells.
    """
    t = np.arange(0, 600, 10.0)
    current = _unsigned_current(t.size)
    # Anode lithiation (charge): terminal voltage decreases.
    voltage = 1.0 - 0.0015 * t

    df = pl.DataFrame(
        {"Time [s]": t, "Voltage [V]": voltage, "Current [A]": current}
    )

    out = iw.transform.set_positive_current_for_discharge(
        df, options={"direction": "charge"}
    )
    assert _nonrest_mean(out) < 0  # charge -> negative
    # Magnitude is preserved, only the sign is set.
    assert set(out["Current [A]"].to_list()) == {0.0, -6.0}


def test_anode_half_cell_no_override_low_confidence_warns():
    """Without an override, a low-confidence voltage guess emits a warning.

    Real anode half-cell relaxation/GITT segments have weak, noisy V-vs-Q
    trends, so the OCV-R heuristic is genuinely ambiguous (p_value near 1).
    The fix surfaces that rather than silently inverting the sign.
    """
    rng = np.random.default_rng(0)
    t = np.arange(0, 600, 10.0)
    current = _unsigned_current(t.size)
    # Near-flat, noisy voltage -> heuristic cannot resolve direction.
    voltage = 0.5 + rng.normal(0.0, 5e-4, size=t.size)

    df = pl.DataFrame(
        {"Time [s]": t, "Voltage [V]": voltage, "Current [A]": current}
    )

    with pytest.warns(UserWarning, match="voltage response"):
        iw.transform.set_positive_current_for_discharge(df)


def test_cathode_full_cell_default_path_classifies_correctly():
    """Regression guard: full/cathode CHARGE (V rises) on the default path.

    No override, no mode column -> voltage-response fallback. OCV rises with
    SOC here, so the heuristic correctly labels the (unsigned) current as
    charge -> negative, with no spurious warning.
    """
    t = np.arange(0, 600, 10.0)
    current = _unsigned_current(t.size)
    voltage = 3.5 + 0.0015 * t  # charge raises terminal voltage

    df = pl.DataFrame(
        {"Time [s]": t, "Voltage [V]": voltage, "Current [A]": current}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # confident classification -> no warn
        out = iw.transform.set_positive_current_for_discharge(df)

    assert _nonrest_mean(out) < 0  # charge -> negative


def test_explicit_discharge_override_yields_positive_current():
    """``direction='discharge'`` signs all non-rest current positive."""
    t = np.arange(0, 600, 10.0)
    current = _unsigned_current(t.size)
    # Voltage shape is irrelevant once the caller declares the direction.
    voltage = 1.0 - 0.0015 * t

    df = pl.DataFrame(
        {"Time [s]": t, "Voltage [V]": voltage, "Current [A]": current}
    )

    out = iw.transform.set_positive_current_for_discharge(
        df, options={"direction": "discharge"}
    )
    assert _nonrest_mean(out) > 0  # discharge -> positive
    assert set(out["Current [A]"].to_list()) == {0.0, 6.0}


def test_mixed_steps_with_direction_override_signs_whole_measurement():
    """Mixed charge+discharge-looking steps + a ``direction`` override.

    Documented semantics: an explicit ``direction`` declares that *all*
    non-rest current in the measurement is that single operation, so the
    whole measurement is signed accordingly regardless of per-step voltage
    behavior. This matches the common per-file delivery case (one file = one
    operation) and is intentional: the override is a deliberate declaration,
    not a hint. Callers with genuinely mixed operations in one file should
    use the default per-step / mode-column path instead.
    """
    t = np.arange(0, 1200, 10.0)
    current = _unsigned_current(t.size)
    current[60] = 0.0  # rest between the two segments

    voltage = np.empty_like(t)
    voltage[:60] = 3.0 + 0.002 * np.arange(60)  # rising segment
    voltage[60:] = 4.0 - 0.002 * np.arange(t.size - 60)  # falling segment

    df = pl.DataFrame(
        {"Time [s]": t, "Voltage [V]": voltage, "Current [A]": current}
    )

    out = iw.transform.set_positive_current_for_discharge(
        df, options={"direction": "discharge"}
    )
    nonrest = out["Current [A]"].filter(out["Current [A]"].abs() > 1e-9)
    assert bool((nonrest > 0).all())  # everything signed as discharge
    assert set(out["Current [A]"].to_list()) == {0.0, 6.0}


def test_invalid_direction_value_is_rejected():
    """An unknown ``direction`` value is rejected (allowed-values check)."""
    t = np.arange(0, 600, 10.0)
    df = pl.DataFrame(
        {
            "Time [s]": t,
            "Voltage [V]": 1.0 - 0.0015 * t,
            "Current [A]": _unsigned_current(t.size),
        }
    )
    with pytest.raises(ValueError):
        iw.transform.set_positive_current_for_discharge(
            df, options={"direction": "sideways"}
        )
