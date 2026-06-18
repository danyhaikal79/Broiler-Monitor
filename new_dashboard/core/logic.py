"""
Pure prediction logic for the broiler feed monitoring system.

No model inference, no UI. Just deterministic functions you can unit-test:
- THI calculation (temperature-humidity index)
- Wet-bulb temperature from RH + dry-bulb (Stull 2011)
- Daily feed intake per bird (Ross 308 as-hatched table, 2022)
- Heat-stress adjustment factor (Purswell et al. 2012)
- Final feed requirement + how much to add

References (all in ../paper/):
- Tao & Xin (2003): THI_broiler = 0.85*T_db + 0.15*T_wb
- Stull (2011): wet-bulb from T_db + RH, J. Appl. Meteor. Climatol.
- Aviagen Ross 308 Broiler Performance Objectives (2022): daily intake table
- Purswell et al. (2012) ILES12-0265: THI vs feed intake quadratic decline
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Ross 308 as-hatched daily feed intake per bird, grams.
# Source: Aviagen Ross 308 / 308 FF Broiler Performance Objectives 2022, p.3
# Index = age in days (0..56).  Day 0 = no intake.  Day 1 from cum. column.
# ----------------------------------------------------------------------------
ROSS308_DAILY_INTAKE_G: dict[int, int] = {
    0: 0,
    1: 12,   2: 16,   3: 20,   4: 24,   5: 27,   6: 31,   7: 35,
    8: 39,   9: 44,  10: 48,  11: 52,  12: 57,  13: 62,  14: 67,
   15: 72,  16: 77,  17: 83,  18: 88,  19: 94,  20: 100, 21: 105,
   22: 111, 23: 117, 24: 122, 25: 128, 26: 134, 27: 139, 28: 145,
   29: 150, 30: 156, 31: 161, 32: 166, 33: 171, 34: 176, 35: 180,
   36: 185, 37: 189, 38: 193, 39: 197, 40: 201, 41: 204, 42: 207,
   43: 211, 44: 213, 45: 216, 46: 219, 47: 221, 48: 223, 49: 225,
   50: 227, 51: 229, 52: 230, 53: 231, 54: 233, 55: 233, 56: 234,
}

# Max feed capacity per feeder type (kg). Used to convert YOLO fill ratio -> kg.
FEEDER_MAX_KG: dict[str, float] = {
    "pan3kg": 3.0,
    "pan7kg": 7.0,
    "tube7kg": 7.0,
}


# ----------------------------------------------------------------------------
# Wet-bulb temperature from dry-bulb + RH (Stull, 2011)
# ----------------------------------------------------------------------------
def wet_bulb_c(t_db_c: float, rh_pct: float) -> float:
    """Stull (2011) wet-bulb approximation. Valid for 5%..99% RH, -20..50°C."""
    T = t_db_c
    RH = max(5.0, min(99.0, rh_pct))   # clamp to valid range
    return (
        T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
        + math.atan(T + RH)
        - math.atan(RH - 1.676331)
        + 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH)
        - 4.686035
    )


# ----------------------------------------------------------------------------
# Temperature-humidity index for broilers (Tao & Xin, 2003)
# ----------------------------------------------------------------------------
def thi_broiler(t_db_c: float, rh_pct: float) -> float:
    """THI = 0.85 * dry-bulb + 0.15 * wet-bulb (Tao & Xin 2003)."""
    t_wb = wet_bulb_c(t_db_c, rh_pct)
    return 0.85 * t_db_c + 0.15 * t_wb


# ----------------------------------------------------------------------------
# Heat-stress adjustment factor for feed intake. Sources:
#   - ONSET 26°C: comfort ceiling from the Malaysian DVS broiler guide
#     (Buku Panduan Penternakan Ayam Pedaging Jilid Ke-4, 2025, Jadual 9:
#      "21-26°C = most suitable"; slight intake drop only above 26°C).
#   - SLOPE -1.4%/°C: feed-intake decline per °C above the comfort temperature,
#     from Andretta et al. (2021), Poultry Science 100(9):101338 (meta-analysis).
#     Applied to THI (not dry-bulb) so humidity still raises stress — at high RH,
#     THI ~= air temperature, so "THI 26" ~= the guide's 26°C.
#   - FLOOR 0.70 (max 30% reduction): a practical cap. Heat-stress studies span
#     ~12% on average (Liu et al. 2020, Poult. Sci. 99:6205) up to ~45% under
#     constant 35°C (Teyssier et al. 2022, Poult. Sci. 101:101963); 30% is an
#     engineering middle, and with the gentle slope it is essentially never hit.
# ----------------------------------------------------------------------------
def thi_stress_factor(thi_c: float) -> float:
    """1.0 = no stress (full intake). Lower = chickens eat less.
    Flat 1.0 up to THI 26, then -1.4% per THI degree, floored at 0.70 (-30%)."""
    if thi_c <= 26.0:
        return 1.0
    return max(0.70, 1.0 - 0.014 * (thi_c - 26.0))


# ----------------------------------------------------------------------------
# Daily intake per bird (Ross 308 baseline, adjusted by heat stress)
# ----------------------------------------------------------------------------
def daily_intake_per_bird_g(age_days: int, thi_c: float) -> float:
    """Baseline Ross 308 intake (g/bird/day) x THI stress factor."""
    if age_days < 0:
        return 0.0
    if age_days > 56:
        baseline = ROSS308_DAILY_INTAKE_G[56]   # clamp at day 56
    else:
        baseline = ROSS308_DAILY_INTAKE_G[age_days]
    return baseline * thi_stress_factor(thi_c)


# ----------------------------------------------------------------------------
# Top-level result for the dashboard
# ----------------------------------------------------------------------------
@dataclass
class Prediction:
    # Inputs (echoed back for the UI)
    temperature_c: float
    humidity_pct: float
    age_days: int
    chicken_count: int
    current_food_kg: float
    feeder_type: str | None        # "pan3kg" / "pan7kg" / "tube7kg" / None

    # Computed environment
    wet_bulb_c: float
    thi_c: float
    thi_stress_factor: float

    # Computed feed numbers
    per_bird_daily_g: float        # adjusted Ross 308 daily intake (g/bird)
    flock_daily_required_kg: float # per_bird * count, in kg
    feed_to_add_kg: float          # max(0, required - current)


def compute(
    *,
    temperature_c: float,
    humidity_pct: float,
    age_days: int,
    chicken_count: int,
    current_food_kg: float,
    feeder_type: str | None,
) -> Prediction:
    """Run the full prediction pipeline. All inputs are explicit/keyword-only."""
    t_wb = wet_bulb_c(temperature_c, humidity_pct)
    thi = thi_broiler(temperature_c, humidity_pct)
    sf = thi_stress_factor(thi)
    per_bird = daily_intake_per_bird_g(age_days, thi)
    flock_kg = per_bird * max(0, chicken_count) / 1000.0
    to_add = max(0.0, flock_kg - max(0.0, current_food_kg))

    return Prediction(
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        age_days=age_days,
        chicken_count=chicken_count,
        current_food_kg=current_food_kg,
        feeder_type=feeder_type,
        wet_bulb_c=t_wb,
        thi_c=thi,
        thi_stress_factor=sf,
        per_bird_daily_g=per_bird,
        flock_daily_required_kg=flock_kg,
        feed_to_add_kg=to_add,
    )
