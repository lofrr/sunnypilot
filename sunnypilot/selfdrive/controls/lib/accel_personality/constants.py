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

# --- Launch jerk-cost relaxation (MPC INPUT: scales the core MPC's own jerk_factor) -----------------------
# The ceiling open-rate above stops binding once the ceiling is already open (the normal case while
# following a lead), so it can't pace the launch ramp. That ramp is paced by A_CHANGE_COST/J_EGO_COST via
# upstream get_jerk_factor(personality) -- stock 1.0 for relaxed/standard, 0.5 for aggressive. JERK_SCALE
# multiplies into that same upstream factor (same lever stock's aggressive tier already uses), bounded to
# near a stop and ramped back to 1.0 (stock) by cruise speed.
# The v=0 knot is NOT monotone with personality -- verified via a closed-loop MPC harness (dead stop +
# departing lead, 3 scenarios): 0.60/0.45 both measurably beat stock 1.0 (0.3-0.65s faster to cross the
# should_stop 0.1 m/s^2 gate, 0 solver resets), but pushing lower is NOT "more relaxed = faster" -- 0.30 came
# back SLOWER than stock in all 3 scenarios (the MPC back-loads the ramp instead of front-loading it once
# A_CHANGE_COST/J_EGO_COST get too cheap to bother avoiding), and below ~0.15 the solver itself destabilizes
# (46-68% QP resets). SPORT's knot is pinned to the verified-good value (tied with NORMAL) rather than pushed
# lower for tier-consistency -- lower is not safe or effective here, unlike every other tier table in this file.
JERK_SCALE_BP = [0., 5.]                           # m/s
JERK_SCALE_V = {
  ECO:    [0.60, 1.0],
  NORMAL: [0.45, 1.0],
  SPORT:  [0.45, 1.0],
}

# --- Onset jerk-cost relaxation (MPC INPUT: general accel<->decel-direction change, not just launch) ------
# The v_ego-indexed table above only relaxes near a stop. This relaxes on ANY fresh accel<->decel direction
# change (aEgo crossing the deadband in a new sign) regardless of speed: drop to a tier-scaled floor right
# away, then ease linearly back to 1.0 (stock) over ONSET_RAMP_S. Same shape as the launch ramp, general to
# any onset (e.g. releasing off a lead, starting to brake). Disabled -> 1.0.
ONSET_DEADBAND = 0.15          # m/s^2: ignore aEgo noise this small around a zero-crossing
ONSET_RAMP_S = 0.4             # s: ease back to stock over this long
ONSET_FLOOR = {ECO: 0.75, NORMAL: 0.65, SPORT: 0.50}

# --- Transient-relax ramp for the two LEVEL-triggered factors below (lead-braking, closing-rate) -----------
# Both factors are eased through _TransientRelax (accel_controller.py), not applied as a level-pinned floor:
# closed-loop-verified (selfdrive/test/longitudinal_maneuvers/plant.py-based) that holding jerk_scale at a
# fixed low value for the duration of a sustained closing/braking episode -- not just its first instant --
# destabilizes the MPC's real-time-iteration re-solve into an oscillation (30+ m/s^3 jerk vs 0 with the
# factor disabled, for an identical scenario). Same ramp duration as ONSET_RAMP_S -- same "soften the first
# jab, then get out of the way" shape -- kept as its own constant since the two mechanisms are conceptually
# distinct (direction-change vs a proactive level) and may need to diverge under future tuning.
RELAX_RAMP_S = 0.4             # s: ease back to stock over this long, regardless of whether the raw factor is still low

# --- Lead-braking jerk-cost relaxation (MPC INPUT: react faster to a hard-braking lead) --------------------
# When the tracked lead is itself decelerating hard, relax jerk cost so the MPC's reaction isn't paced by a
# jerk budget tuned for routine following. No lead, or lead not braking -> 1.0. Disabled -> 1.0.
LEAD_BRAKE_ALEAD_BP = [-3.0, -0.5]      # m/s^2, lead's own aLeadK (ascending, as np.interp requires)
LEAD_BRAKE_FACTOR_V = {
  ECO:    [0.75, 1.0],
  NORMAL: [0.60, 1.0],
  SPORT:  [0.45, 1.0],
}

# --- Closing-rate jerk-cost relaxation (MPC INPUT: react faster to a fast-closing gap, any cause) ----------
# Complements LEAD_BRAKE_FACTOR_V, which keys off the LEAD's own deceleration: a gap can close quickly for
# reasons aLeadK never reflects (a cut-in, or ego simply catching up faster than the lead is slowing). Onset
# relax (above) only reacts the cycle AFTER a_ego has already crossed its deadband -- reactive on a realized
# signal, so it structurally can't soften the very first jab into a fresh, fast-closing gap. vRel is an MPC
# INPUT (causal, known before any brake is commanded), so keying off it directly closes that gap. No lead, or
# not closing past the gate -> 1.0. Disabled -> 1.0.
CLOSING_VREL_BP = [-6.0, -1.5]          # m/s, closing rate (negative = closing), ascending for np.interp
CLOSING_FACTOR_V = {
  ECO:    [0.75, 1.0],
  NORMAL: [0.60, 1.0],
  SPORT:  [0.45, 1.0],
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
