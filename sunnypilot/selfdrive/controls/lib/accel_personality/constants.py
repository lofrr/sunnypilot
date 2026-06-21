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

# Accel ceiling + its upward slew rate (the POSITIVE-accel / launch + cruise-accel side; independent of
# braking, so tuning here does NOT touch the gentle-brake goals). off==stock is enforced in accel_controller
# (get_max_accel/get_rise_rate fall back to STOCK_* when disabled), so the NORMAL profile is free to differ
# from stock -- all three tiers are now distinct. Start-from-stop is FAST: launch peak (v=0) is firm and the
# rise rate (how fast the ceiling opens) is well above stock 0.05 in every tier, stepped ECO < NORMAL < SPORT.
A_CRUISE_MAX_BP = [0., 14., 25., 40.]
STOCK_A_CRUISE_MAX_V = [1.6, 0.7, 0.2, 0.08]
STOCK_RISE_RATE = 0.05
A_CRUISE_MAX_V = {
  ECO:    [1.70, 0.75, 0.25, 0.10],   # prompt launch, efficient cruise
  NORMAL: [2.10, 1.10, 0.50, 0.18],   # quick launch, balanced cruise
  SPORT:  [2.60, 1.55, 0.85, 0.35],   # fast launch, strong cruise
}
RISE_RATE = {ECO: 0.10, NORMAL: 0.15, SPORT: 0.22}   # ceiling open-rate: all >> stock 0.05 for fast take-off

# Early soft braking: predicted brake need (m/s^2) -> early decel target (m/s^2). Front-loads a gentle
# decel as soon as the 3s plan lookahead predicts a brake, so decel is spread out instead of arriving as
# one late firm onset. The old ECO row was near-flat (-0.07 at brake_need~1.0 vs an eventual ~-0.88 plan
# brake) so it barely front-loaded -> late, jerky onsets on route 00000456. Deepened toward (but kept
# gentler than) NORMAL. Hard brakes (brake_need>=HARD_BRAKE_NEED or raw<=HARD_BRAKE_TARGET_ACCEL) still
# bypass to stock, and min(.,raw) keeps it never weaker than the plan.
SMOOTH_DECEL_BP = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
SMOOTH_DECEL_V = {
  ECO:    [0.00, -0.08, -0.20, -0.35, -0.55, -0.78, -1.00],
  NORMAL: [0.00, -0.13, -0.30, -0.55, -0.84, -1.12, -1.40],
  SPORT:  [0.00, -0.17, -0.40, -0.72, -1.05, -1.35, -1.65],
}
BRAKE_DEEPENING_JERK = {ECO: 0.5, NORMAL: 0.8, SPORT: 1.0}
BRAKE_RELEASE_JERK = 2.0
ACCEL_RISE_JERK = {ECO: 1.0, NORMAL: 1.5, SPORT: 2.2}   # accel-onset jerk: higher = snappier take-off, stepped per tier

SMOOTH_DECEL_LOOKAHEAD_T = 3.0
MIN_SMOOTH_BRAKE_NEED = 0.2

# Front-load over-bite cap. The SMOOTH_DECEL front-load is allowed to brake at most this much DEEPER than
# the live raw plan. On a cut-in/merge the 3s brake_need spikes and the table would front-load a firm brake
# while the plan still wants throttle/coast -> an abrupt early bite (the "worse on merge" feel; routes
# 45e/460). This binds only in that contradictory case; once the plan itself brakes, raw-OVERBITE_CAP sits
# below the table value so the table wins and the anticipatory early brake (route 456 fix) is preserved.
OVERBITE_CAP = 0.30   # m/s^2 max front-load depth below the live plan
HARD_BRAKE_TARGET_ACCEL = -1.5
HARD_BRAKE_NEED = 2.6

# Hard-brake onset jerk cap. Firm/closing-lead brakes (raw<=HARD_BRAKE_TARGET_ACCEL or brake_need>=
# HARD_BRAKE_NEED) used to fully stand the shaper down -> raw stock MPC onset, which lands as a grab
# (felt jerk on routes 45c/45d, both user bookmarks). This caps ONLY the deepening RATE of such onsets;
# the full plan depth is always reached (one-sided), so the brake is never weaker or later in magnitude --
# only the rate of getting there is smoothed. Verified on the 45d@767 closing lead (ego 28.6 m/s, lead
# braking aLeadK -2.9): cap=2.0 adds <=65ms to reach 80% of target and <=0.22m extra gap, inside the
# 150ms / 2.0m safety budget. Do NOT lower below 2.0 (1.8 breaches the lag gate). True emergencies
# (mpc.crash_cnt>0, i.e. FCW) skip the cap and stay pure passthrough.
HARD_BRAKE_ONSET_JERK = 2.0   # m/s^3, deepening-only onset rate cap on firm (non-crash) hard brakes

# Stop-imminent stand-down. The shaper's gentle bite is softer than the plan, so on a STOP approach it
# coasts the car in -> halts too close / "stop-roll-stop" creep. When the plan predicts a near-stop
# within the lookahead, stand the shaper down (full stock decel) so it stops at the proper gap with no
# coast. Keyed on the PREDICTED speed reaching ~0 (covers lead AND light/sign stops), NOT raw ego speed
# -- so non-stop low-speed braking (slowing to a moving follow) keeps the gentle onset at every speed.
STOP_IMMINENT_VEGO = 1.0          # m/s  plan-predicted speed below this within the lookahead == stop coming
STOP_IMMINENT_LOOKAHEAD_T = 3.0   # s

# Stop/creep stop-neutrality. Below this ego speed, the brake side hands the plan straight through (stock),
# so the controller cannot soften the final crawl and let the car coast in closer than stock. Matches the
# radar_distance low-speed stop-neutrality so ON == OFF near stops. Positive-accel (launch) shaping is
# unaffected (the launch profiles still apply via the accel ceiling).
STOP_PASSTHROUGH_V = 5.0          # m/s ego speed below which braking is stock passthrough

# --- Convex brake-onset shaper (param-gated; ECO/SPORT only, NORMAL = stock passthrough) ---
# The grabby bite is the raw MPC plan: stock deepening uses a CONSTANT jerk (integrates to a LINEAR
# accel ramp) and min(slewed,raw) lets the deep raw plan win, so the bite passes through untouched.
# Fix: jerk-limit the deepening with a DEPTH-PROPORTIONAL jerk
#   jerk(a) = ONSET_JERK0 + ONSET_JERK_GAIN * abs(a_current),  capped at ONSET_JERK_MAX
# At the bite (a~0) the jerk is ONSET_JERK0 (gentle); it grows with decel depth, so the decel magnitude
# follows da/dt = j0 + k*a  =>  a(t) = (j0/k)*(exp(k*t) - 1) -- the exponential-growth reference. The
# output is never deeper than the plan (only ever softer-or-equal during the bite) and converges to it.
# No velocity-debt feedback: it carried stale state across closely-spaced stop-and-go brakes and
# over-braked the next onset (verified). NORMAL omitted -> shaper never runs.
ONSET_JERK0 = {ECO: 0.15, SPORT: 0.25}        # m/s^3  initial gentle jerk at the bite (target band 0.15-0.25)
ONSET_JERK_GAIN = {ECO: 0.9, SPORT: 1.5}      # 1/s    depth-proportional growth rate k (lowered: gentler jerk-build = smoother decel, less "jerky")

# Bounded softening: the gentle bite lags the plan (brakes shallower) at the very start. To keep the
# softening modest (so it never feels like "no brakes"), an INSTANTANEOUS-gap catch-up adds jerk when
# realized lags the plan by more than ONSET_GAP_SOFT, hard-capped at ONSET_JERK_MAX. This uses the
# current accel gap only (no integrated state) so nothing carries across closely-spaced brakes. Steady
# softening then settles near ONSET_GAP_SOFT; the hard cap keeps the catch from ever being a grab.
ONSET_GAP_SOFT = {ECO: 0.30, SPORT: 0.25}     # m/s^2  tolerated shallower-than-plan gap before catch-up
ONSET_GAP_GAIN = {ECO: 4.0, SPORT: 5.0}       # 1/s    extra jerk per m/s^2 of gap beyond ONSET_GAP_SOFT
ONSET_JERK_MAX = {ECO: 1.1, SPORT: 1.4}       # m/s^3  hard ceiling on convex-path jerk (lowered: smoother catch-up)
# Fast hand-back: once the plan leaves the gentle zone (no longer armed) but a soft gap is still open,
# close it at this FIRM jerk so the output catches the plan BEFORE braking gets firm -> no late-brake lag
# into the [-1.5,-1.0] band. Jerk-limited (not a snap), and never deeper than the plan, so not a grab.
ONSET_HANDBACK_JERK = {ECO: 2.2, SPORT: 3.0}  # m/s^3  gap-close rate (lowered: gentler hand-back = less jounce/jerk)

# Arm gates (conservative). Only shape genuinely gentle onsets; firm/deep onsets fall to the stock
# never-weaker slew (they SHOULD bite). Two independent safety layers against late braking: (1) the
# PREDICTIVE brake_need gate declines to start a gentle bite when a firmer brake is seen within 3s, so
# we don't soften ahead of one; (2) the fast hand-back (ONSET_HANDBACK_JERK) closes any open soft gap
# before the plan reaches firm braking. Together: 0 firm-band ([-1.5,-1.0]) lag on the verified windows.
SOFT_ONSET_MAX_BRAKE_NEED = 0.9               # do NOT soften if a firmer brake is predicted within 3s
SOFT_ONSET_MAX_INSTANT_ACCEL = -0.7           # m/s^2  stop softening (fast hand-back) once raw is this deep
# Sticky re-arm: once an onset goes firm (instantaneously too deep) it latches OFF; require this many
# consecutive released/flat frames before a NEW soft window may open, so lead/SnG jitter cannot re-arm
# the bite every few hundred ms (flicker guard). Controller runs at the model rate (DT_MDL = 0.05 s).
SOFT_ONSET_REARM_FRAMES = 10                  # frames (~0.5 s at 20 Hz model rate) of release before re-arm
