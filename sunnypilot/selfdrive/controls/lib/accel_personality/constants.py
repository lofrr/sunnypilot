"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom

AccelerationPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ECO = AccelerationPersonality.eco
NORMAL = AccelerationPersonality.normal
SPORT = AccelerationPersonality.sport

PERSONALITY_MIN = min(AccelerationPersonality.schema.enumerants.values())
PERSONALITY_MAX = max(AccelerationPersonality.schema.enumerants.values())

# Positive-accel ceiling + its upward slew rate (launch/cruise side; independent of braking). off==stock is
# enforced in accel_controller (falls back to STOCK_* when disabled), so the tiers are free to differ.
A_CRUISE_MAX_BP = [0., 14., 25., 40.]
STOCK_A_CRUISE_MAX_V = [1.6, 0.7, 0.2, 0.08]
STOCK_RISE_RATE = 0.05
A_CRUISE_MAX_V = {
  ECO:    [1.70, 0.75, 0.25, 0.10],   # prompt launch, efficient cruise
  NORMAL: [2.10, 1.10, 0.50, 0.18],   # quick launch, balanced cruise
  SPORT:  [2.60, 1.55, 0.85, 0.35],   # fast launch, strong cruise
}
RISE_RATE = {ECO: 0.10, NORMAL: 0.15, SPORT: 0.22}   # ceiling open-rate: all >> stock 0.05 for fast take-off

# Anticipatory front-load: predicted brake need (m/s^2) -> early decel target (m/s^2). Starts a gentle
# decel early when a brake is predicted, so it arrives spread out, not as one late firm onset. The first
# knot sits AT the MIN_SMOOTH_BRAKE_NEED gate (0.00 there): below the gate there is no front-load, so there
# is no dead [0, gate) anchor and no step at the gate (the old [0.0 -> 0.00] knot was never evaluated).
SMOOTH_DECEL_BP = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
SMOOTH_DECEL_V = {
  ECO:    [0.00, -0.20, -0.35, -0.55, -0.78, -1.00],
  NORMAL: [0.00, -0.30, -0.55, -0.84, -1.12, -1.40],
  SPORT:  [0.00, -0.40, -0.72, -1.05, -1.35, -1.65],
}
BRAKE_DEEPENING_JERK = {ECO: 0.5, NORMAL: 0.8, SPORT: 1.0}
BRAKE_RELEASE_JERK = 2.0
ACCEL_RISE_JERK = {ECO: 1.0, NORMAL: 1.5, SPORT: 2.2}   # accel-onset jerk: higher = snappier take-off, stepped per tier

SMOOTH_DECEL_LOOKAHEAD_T = 3.0
MIN_SMOOTH_BRAKE_NEED = 0.4   # below this no front-load (kills the faint low-brake_need drag + the gate-crossing toggle)

# Cap how much DEEPER than the live plan the front-load may bite -> no abrupt over-bite on a cut-in
# brake_need spike (binds only when the plan still wants throttle; once it brakes, the table wins).
OVERBITE_CAP = 0.30   # m/s^2 max front-load depth below the live plan

# Hard brake: at/below this accel, or this predicted brake_need within the lookahead, the controller hands
# the plan straight through at full strength and rate (no front-load, no rate limit) -- a firm/closing-lead
# brake must never be delayed, softened or rate-limited.
HARD_BRAKE_TARGET_ACCEL = -1.5
HARD_BRAKE_NEED = 2.6

# Stop-imminent stand-down. When the plan predicts a near-stop within the lookahead, hand the plan straight
# through (stock decel) so the car stops at the proper gap with no front-load coast-in. Keyed on the
# PREDICTED speed reaching ~0 (covers lead AND light/sign stops), not raw ego speed.
STOP_IMMINENT_VEGO = 1.0          # m/s  plan-predicted speed below this within the lookahead == stop coming
STOP_IMMINENT_LOOKAHEAD_T = 3.0   # s

# Below this ego speed the brake side is stock passthrough (the comfort stop below adds the only low-speed
# shaping); the bounded onset-spread does not run here, so a stock stop is not rate-limited.
STOP_PASSTHROUGH_V = 5.0          # m/s

# Scoped onset-spread -- the ONLY place the output may be transiently WEAKER than the plan. On a NON-emergency
# brake the onset may arrive spread over a bounded ramp instead of stepping straight to the plan: the output
# may lag the plan by at most ONSET_SPREAD_MAX, deepening toward it at ONSET_SPREAD_JERK. A firm/closing brake
# (raw <= HARD_BRAKE_TARGET_ACCEL or brake_need >= HARD_BRAKE_NEED, FCW/crash, should_stop, blended/e2e) skips
# this entirely (raw passthrough), so a real hard brake is never softened or delayed.
ONSET_SPREAD_MAX = 0.25           # m/s^2: max the output may lag (be weaker than) the live plan, non-emergency only
ONSET_SPREAD_JERK = 2.5           # m/s^3: rate the spread output deepens back toward the plan

# Low-speed comfort stop = ANTI-CREEP HOLD (not a brake adder). In the final approach behind a (near-)stopped
# lead it HOLDS the deepest decel the PLAN itself has commanded (gentle-capped), so the brake does not ease
# off / creep in before the car is stopped (no roll, slightly roomier). It is NEVER firmer than the plan, so
# it can never add a hard bite -- the stop stays as gentle as the plan's own decel. Outside the final approach
# (cruising / gap opening as a creeping lead pulls away / lead moving / launch) the floor eases out at the
# release rate. min(plan, floor) keeps it never weaker than the plan. Replaces the old kinematic v^2/(2*gap)
# enforcer, which engaged late and demanded a firm ~-1.6 grab to hit a fixed gap. Off => no-op.
COMFORT_STOP_ENABLED = False      # gated off: final-approach stops pass through stock
COMFORT_STOP_V = 4.0              # m/s: only engage at/below this ego speed
COMFORT_STOP_LEAD_V = 1.0         # m/s: only behind a (near-)stopped lead
COMFORT_STOP_GAP = 5.0            # m: reference standstill gap (radar dRel) for the final-approach window
COMFORT_STOP_MAX_DECEL = -1.6     # m/s^2: backstop cap on the held decel (a brief plan spike is not held firmer than this)
COMFORT_STOP_RELEASE_V = 0.3      # m/s: below this, ease the floor out (release rate) -> smooth stock standstill handoff
COMFORT_STOP_HOLD_GAP = 2.0       # m: within this of the reference gap = final-approach window where the hold applies;
                                  # beyond it the floor eases out (a creeping lead opening the gap -> no phantom brake)

# Gas suppression near a lead: coast instead of accelerating toward a close lead, in two cases (OR) --
# T1 we braked for it within RECENT_T and it is still not pulling away (closing < VREL); T2 we are clearly
# gaining on it (closing < CLOSE). Only reduces accel, never a brake; opening/far lead keeps its gas.
GAS_SUPPRESS_ENABLED = False
GAS_SUPPRESS_DREL = 60.0          # m: lead within this distance
GAS_SUPPRESS_VREL = 0.5           # m/s: "not pulling away" bound for the rebound trigger (vLead - vEgo)
GAS_SUPPRESS_CLOSE = -1.5         # m/s: closing rate below which gas is suppressed outright
GAS_SUPPRESS_RECENT_T = 3.0       # s: a brake within this long counts as recent
GAS_SUPPRESS_BRAKE_THR = -0.30    # m/s^2: output below this is a "brake" for the recency latch
