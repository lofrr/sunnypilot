"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Acceleration Personality tuning tables. The controller shapes only what the longitudinal MPC CONSUMES
(the positive-accel ceiling + its open-rate, and an add-only follow-gap widen fed to the MPC's t_follow);
it never post-shapes the MPC's output accel. Disabled => every getter returns the upstream stock value,
so off == byte-stock.
"""

from cereal import custom

AccelerationPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ECO = AccelerationPersonality.eco
NORMAL = AccelerationPersonality.normal
SPORT = AccelerationPersonality.sport

PERSONALITY_MIN = min(AccelerationPersonality.schema.enumerants.values())
PERSONALITY_MAX = max(AccelerationPersonality.schema.enumerants.values())

# --- Positive-accel ceiling (launch/cruise) + its upward open-rate ---------------------------------------
# off == stock: get_max_accel/get_rise_rate fall back to the STOCK_* values (upstream get_max_accel table
# and +0.05 ceiling slew), independent of the NORMAL profile so NORMAL is free to differ.
# ACCEL_MAX (opendbc) hard-caps the ceiling at 2.0 m/s^2, so the launch knots are set at/below it.
A_CRUISE_MAX_BP = [0., 10., 25., 40.]              # m/s (matches upstream A_CRUISE_MAX_BP)
STOCK_A_CRUISE_MAX_V = [1.6, 1.2, 0.8, 0.6]        # upstream A_CRUISE_MAX_VALS -> off == byte-stock ceiling
STOCK_RISE_RATE = 0.05                             # upstream ceiling open-rate (m/s^2 per cycle)
A_CRUISE_MAX_V = {
  ECO:    [1.55, 0.75, 0.35, 0.20],   # responsive off the line, LAZY at highway speed (mileage)
  NORMAL: [2.00, 1.40, 0.95, 0.70],   # brisk launch, balanced cruise
  SPORT:  [2.00, 1.70, 1.20, 0.90],   # strong launch (ACCEL_MAX caps the 0 m/s knot), assertive cruise
}
# Ceiling open-rate: how fast the accel ceiling may rise per cycle, speed-dependent (decoupled from the
# ceiling magnitude above). Near a stop (v=0) the rate is set high enough to be non-binding within ~2 cycles
# (DT_MDL=0.05s @ 20Hz) so launch is never delayed waiting on the ceiling to open, regardless of personality.
# By the v=5 knot the rate settles to the telemetry-verified steady-state value (unchanged from before) so
# cruise/resume behavior at speed is preserved exactly. The MPC's own jerk/a_change cost still smooths the
# actual accel.
RISE_RATE_BP = [0., 5.]                            # m/s
RISE_RATE_V = {
  ECO:    [0.80, 0.07],
  NORMAL: [1.00, 0.16],
  SPORT:  [1.20, 0.24],
}

# --- Follow-gap widen (add-only, fed to the MPC t_follow) ------------------------------------------------
# Add a small speed-dependent widen to the stock t_follow (the driver's gap-button value). Wider gap ->
# MPC brakes earlier + gentler onto a slowing lead and settles a roomier cruise gap. Invariants:
#   * add-only         -> desired distance >= stock -> braking >= stock;
#   * zero below TF_WIDEN_V_BP[0] -> low-speed & standstill gap stay stock (stock stop distance preserved);
#   * slewed per cycle -> no rubber-band;  decel-hold -> gap won't shrink while braking (stays committed).
TF_WIDEN_V_BP = [14.0, 28.0]                       # m/s: widen ramps in across this band, flat above
TF_WIDEN_BASE_V = [0.0, 0.30]                      # s: base follow-time added at the band ends (pre-tier)
TF_WIDEN_TIER = {ECO: 1.30, NORMAL: 1.00, SPORT: 0.50}   # ECO roomiest/smoothest, SPORT tightest/snappiest
TF_WIDEN_MAX = 0.45                                # s: absolute cap on the added gap (never explodes)
TF_SLEW_PER_S = 0.50                               # s per second: max rate the widen may open/close
TF_DECEL_HOLD_A = -0.20                            # m/s^2: at/below this a_ego (braking) the widen won't shrink
