"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Acceleration personality: per-profile launch/cruise accel ceiling (ECO/NORMAL/SPORT), an anticipatory
brake front-load, and a low-speed comfort stop. SAFETY: a firm/closing brake -- emergency (raw <=
HARD_BRAKE_TARGET_ACCEL or brake_need >= HARD_BRAKE_NEED), FCW/crash, should_stop, or blended/e2e -- passes
the plan straight through at full strength and rate, never softened/delayed/rate-limited. Only on the
NON-emergency comfort path may the onset arrive spread by at most ONSET_SPREAD_MAX (a tightly bounded,
transient lag) so a gentle brake does not land as a step. The front-load and comfort stop only ever ADD
braking (min(., plan)). Disabled => byte-stock.
"""

from collections.abc import Sequence

import numpy as np

from cereal import messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  NORMAL, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, A_CRUISE_MAX_V, RISE_RATE, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, SMOOTH_DECEL_BP, SMOOTH_DECEL_V, BRAKE_DEEPENING_JERK, \
  BRAKE_RELEASE_JERK, ACCEL_RISE_JERK, SMOOTH_DECEL_LOOKAHEAD_T, MIN_SMOOTH_BRAKE_NEED, \
  HARD_BRAKE_TARGET_ACCEL, HARD_BRAKE_NEED, OVERBITE_CAP, STOP_PASSTHROUGH_V, \
  STOP_IMMINENT_VEGO, STOP_IMMINENT_LOOKAHEAD_T, ONSET_SPREAD_MAX, ONSET_SPREAD_JERK, \
  COMFORT_STOP_ENABLED, COMFORT_STOP_V, COMFORT_STOP_LEAD_V, COMFORT_STOP_GAP, \
  COMFORT_STOP_MAX_DECEL, COMFORT_STOP_RELEASE_V, COMFORT_STOP_HOLD_GAP, \
  GAS_SUPPRESS_ENABLED, GAS_SUPPRESS_DREL, GAS_SUPPRESS_VREL, GAS_SUPPRESS_CLOSE, \
  GAS_SUPPRESS_RECENT_T, GAS_SUPPRESS_BRAKE_THR, \
  PHYSICS_CAP_ENABLED, PHYS_CAP_MIN_TTC, PHYS_CAP_MIN_DREL, PHYS_CAP_TGAP, PHYS_CAP_MIN_GAP, PHYS_CAP_VREL_MARGIN, \
  PHYS_CAP_FORGET_T, PHYS_CAP_MIN_A

_ZERO_ACCEL_EPS = 1e-6


class AccelController:
  def __init__(self, CP: structs.CarParams, mpc, params=None):
    self._CP = CP
    self._mpc = mpc
    self._params = params or Params()
    self._frame = 0
    self._enabled = self._params.get_bool("AccelPersonalityEnabled")
    self._personality = NORMAL
    self._v_ego = 0.0
    self._last_target_accel = 0.0
    self._brake_need = 0.0
    self._decel_target = 0.0
    self._smooth_active = False
    self._bypassed = False
    self._lead_status = False
    self._lead_d = 0.0
    self._lead_vlead = 0.0
    self._stop_floor = 0.0       # comfort-stop floor latch (monotone within a stop episode, eased on release)
    self._comfort_stop_enabled = COMFORT_STOP_ENABLED
    self._gas_suppress_enabled = GAS_SUPPRESS_ENABLED
    self._physics_cap_enabled = PHYSICS_CAP_ENABLED
    self._cap_vrel = 0.0                                 # held worst-case (most-closing) lead for the physics cap
    self._cap_dRel = 1e9
    self._cap_vlead = 0.0
    self._since_brake_frames = 10 ** 6                   # frames since last brake output (gas-suppress recency)
    self._read_params()

  def _read_params(self) -> None:
    self._enabled = self._params.get_bool("AccelPersonalityEnabled")
    if not self._enabled:
      self._personality = NORMAL
      return
    self._personality = get_sanitize_int_param("AccelPersonality", PERSONALITY_MIN, PERSONALITY_MAX, self._params)

  def update(self, sm: messaging.SubMaster) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._read_params()
    self._v_ego = sm['carState'].vEgo
    lead = sm['radarState'].leadOne          # raw radard lead (== what the MPC sees at crawl, where the enforcer acts)
    self._lead_status = bool(lead.status)
    self._lead_d = float(lead.dRel)
    self._lead_vlead = float(lead.vLead)
    self._frame += 1

  def get_max_accel(self, v_ego: float) -> float:
    # Disabled -> stock ceiling (off == stock, independent of the NORMAL profile so NORMAL is free to differ).
    table = A_CRUISE_MAX_V[self._personality] if self._enabled else STOCK_A_CRUISE_MAX_V
    return float(np.interp(v_ego, A_CRUISE_MAX_BP, table))

  def get_rise_rate(self) -> float:
    # Disabled -> stock rise rate (off == stock, independent of the NORMAL profile).
    return RISE_RATE[self._personality] if self._enabled else STOCK_RISE_RATE

  def get_decel_target(self, brake_need: float) -> float:
    return float(np.interp(max(0.0, float(brake_need)), SMOOTH_DECEL_BP, SMOOTH_DECEL_V[self._personality]))

  def smooth_target_accel(self, raw_target_accel: float, accel_trajectory: Sequence[float], t_idxs: Sequence[float],
                          should_stop: bool, reset: bool = False, stock_brake: bool = False,
                          speed_trajectory: Sequence[float] | None = None) -> float:
    raw = float(raw_target_accel)
    self._brake_need = self._compute_brake_need(raw, accel_trajectory, t_idxs)
    self._decel_target = 0.0
    self._smooth_active = False
    self._bypassed = False

    out = self._shape(raw, should_stop, reset, speed_trajectory, t_idxs, stock_brake)
    out = self._comfort_stop(out, reset)   # low-speed monotone comfort decel-to-stop (replaces the self-releasing enforcer)
    out = self._physics_decel_cap(out, reset)   # don't over-brake a closing lead that has room (brakes < stock)
    return self._finalize(out)

  def _shape(self, raw: float, should_stop: bool, reset: bool, speed_trajectory, t_idxs, stock_brake: bool) -> float:
    # --- Full stock passthroughs (output is exactly the plan, no shaping) ---
    if reset or not self._enabled:
      return raw                                               # disabled / reset
    if self._v_ego < STOP_PASSTHROUGH_V and raw <= 0.0:
      return raw                                               # stop/creep regime: braking is stock (no coast-in)
    self._bypassed = self._emergency_bypass(raw, should_stop)
    if self._bypassed or self._stop_imminent(speed_trajectory, t_idxs):
      return raw                                               # emergency / coming stop: full strength, no delay

    # Anticipatory front-load, capped at OVERBITE_CAP below the live plan (avoids an abrupt over-bite on a
    # cut-in brake_need spike).
    target = raw
    if self._brake_need >= MIN_SMOOTH_BRAKE_NEED:
      self._smooth_active = True
      self._decel_target = max(self.get_decel_target(self._brake_need), raw - OVERBITE_CAP)
      target = min(raw, self._decel_target)
      if raw > 0.0:
        target = max(target, 0.0)                              # plan wants throttle -> ease the gas early, never fabricate a brake
    target = self._suppress_gas_near_lead(target, raw)
    slewed = self._slew(target)
    if raw >= 0.0:
      return slewed
    if stock_brake:
      return min(slewed, raw)                                  # blended/e2e: the model owns the brake -> strict never-weaker
    return self._onset_spread(slewed, raw)                     # non-emergency brake: bounded onset spread (<= ONSET_SPREAD_MAX weaker)

  def _physics_decel_cap(self, out: float, reset: bool) -> float:
    # On a closing lead with genuine room, cap the brake at the kinematic decel needed to settle at a comfortable
    # gap -- the stock MPC over-brakes a slower lead at speed. Only softens (max(out, a_phys)). Uses a HELD
    # worst-case closing (decaying ~PHYS_CAP_FORGET_T) so a benign lead-flicker frame cannot relax it, and only
    # acts when the closing itself warrants a real brake (a_phys <= PHYS_CAP_MIN_A) so it never softens a brake
    # meant for another cause (curve / vision / a closer lead). Guarded to TTC + distance, pessimistic vRel
    # margin, self-disengaging as room shrinks (full stock brake returns).
    if reset or not self._lead_status:
      self._cap_vrel, self._cap_dRel, self._cap_vlead = 0.0, 1e9, 0.0
    else:
      vrel = self._lead_vlead - self._v_ego
      if vrel < self._cap_vrel:                                # adopt a more-closing lead immediately
        self._cap_vrel, self._cap_dRel, self._cap_vlead = vrel, self._lead_d, self._lead_vlead
      else:                                                    # forget an old threat slowly
        f = DT_MDL / PHYS_CAP_FORGET_T
        self._cap_vrel += (vrel - self._cap_vrel) * f
        self._cap_dRel += (self._lead_d - self._cap_dRel) * f
        self._cap_vlead += (self._lead_vlead - self._cap_vlead) * f
    if not self._enabled or not self._physics_cap_enabled or out >= 0.0 or not self._lead_status:
      return out
    hv, hd, hl = self._cap_vrel, self._cap_dRel, self._cap_vlead
    if hv >= -0.5 or hd < PHYS_CAP_MIN_DREL or hd / -hv < PHYS_CAP_MIN_TTC:
      return out
    room = hd - max(PHYS_CAP_MIN_GAP, PHYS_CAP_TGAP * hl)
    if room <= 1.0:
      return out
    a_phys = -((hv - PHYS_CAP_VREL_MARGIN) ** 2) / (2.0 * room)
    if a_phys > PHYS_CAP_MIN_A:                                # lead-closing alone does not warrant a real brake
      return out
    return max(out, a_phys)                                    # only ever softens; never below the needed decel

  def _suppress_gas_near_lead(self, target: float, raw: float) -> float:
    # Coast instead of accelerating toward a close lead: T1 recent brake + lead not pulling away, or T2 clearly
    # closing. Only reduces accel, never a brake. Off => no-op.
    if not self._gas_suppress_enabled or raw <= 0.0 or not self._lead_status:
      return target
    if not 0.1 < self._lead_d < GAS_SUPPRESS_DREL:
      return target
    closing = self._lead_vlead - self._v_ego
    recent_brake = self._since_brake_frames * DT_MDL < GAS_SUPPRESS_RECENT_T
    if (recent_brake and closing < GAS_SUPPRESS_VREL) or closing < GAS_SUPPRESS_CLOSE:
      return min(target, 0.0)
    return target

  def _onset_spread(self, shaped: float, raw: float) -> float:
    # Scoped softening: on a NON-emergency brake the onset may arrive spread instead of stepping to the plan.
    # The output deepens toward the plan jerk-limited at ONSET_SPREAD_JERK and may lag it by at most
    # ONSET_SPREAD_MAX -- a tightly bounded, transient weaker-than-plan window that smooths the felt onset.
    # Emergency brakes never reach here (raw passthrough in _shape), so a genuine hard brake is never softened.
    # The front-load still wins when it is deeper (anticipation preserved).
    spread = max(raw, self._last_target_accel - ONSET_SPREAD_JERK * DT_MDL)   # deepen toward the plan, jerk-limited
    spread = min(spread, raw + ONSET_SPREAD_MAX)                              # never more than the bounded lag weaker
    return min(shaped, spread)

  def _comfort_stop(self, out: float, reset: bool) -> float:
    # Low-speed ANTI-CREEP HOLD behind a near-stopped lead. In the final-approach window it HOLDS the deepest
    # decel the PLAN itself commanded this episode (gentle-capped at COMFORT_STOP_MAX_DECEL), so the brake does
    # not ease off / creep in before the car is stopped (no roll, slightly roomier). It is NEVER firmer than the
    # plan -- it only stops the brake from WEAKENING -- so it can never add a hard bite (the old kinematic
    # enforcer demanded a firm ~-1.6 grab; this does not). Outside the window (gap opening as a creeping lead
    # pulls away / lead moving / launch / standstill) the floor eases out at the release rate. min(out, floor)
    # keeps it never weaker than the plan. Off => no-op (off==stock).
    if reset or not self._enabled or not self._comfort_stop_enabled:
      self._stop_floor = 0.0                                   # disabled/gated/reset: drop the latch, pure passthrough
      return out
    final_approach = (self._lead_status and self._lead_vlead < COMFORT_STOP_LEAD_V and self._lead_d > 0.1
                      and COMFORT_STOP_RELEASE_V <= self._v_ego < COMFORT_STOP_V
                      and self._lead_d - COMFORT_STOP_GAP <= COMFORT_STOP_HOLD_GAP)
    if final_approach:
      plan_hold = max(out, COMFORT_STOP_MAX_DECEL)             # the plan's own decel, gentle-capped (never firmer)
      self._stop_floor = min(plan_hold, self._stop_floor)      # latch the deepest -> hold through the plan's ease
    else:
      # Not final approach (cruise / gap opening / lead moving / launch / standstill): ease the floor toward 0 at
      # the release rate. Matches _shape's own _slew_up rate, so the floor decays in lockstep with the natural
      # output -> no launch drag, no release-direction snap, no phantom brake into an opening gap.
      self._stop_floor = min(0.0, self._stop_floor + BRAKE_RELEASE_JERK * DT_MDL)
    return min(out, self._stop_floor) if self._stop_floor < 0.0 else out

  def _stop_imminent(self, speed_trajectory: Sequence[float] | None, t_idxs: Sequence[float]) -> bool:
    # plan predicts a near-stop within the lookahead -> a stop is coming (lead or light/sign).
    if speed_trajectory is None:
      return False
    return any(float(s) < STOP_IMMINENT_VEGO
               for s, t in zip(speed_trajectory, t_idxs, strict=False) if float(t) <= STOP_IMMINENT_LOOKAHEAD_T)

  def _compute_brake_need(self, raw_target_accel: float, accel_trajectory: Sequence[float], t_idxs: Sequence[float]) -> float:
    min_accel = float(raw_target_accel)
    for accel, t in zip(accel_trajectory, t_idxs, strict=False):
      if float(t) <= SMOOTH_DECEL_LOOKAHEAD_T:
        min_accel = min(min_accel, float(accel))
    return max(0.0, -min_accel)

  def _emergency_bypass(self, raw_target_accel: float, should_stop: bool) -> bool:
    return (self._mpc.crash_cnt > 0 or should_stop or
            raw_target_accel <= HARD_BRAKE_TARGET_ACCEL or self._brake_need >= HARD_BRAKE_NEED)

  def _slew(self, target_accel: float) -> float:
    # Jerk-limit the brake DEEPENING (smooths the front-load's extra depth). On the brake side the caller
    # clamps with min(., raw), so this NEVER delays a real brake -- when the plan is deeper than the slewed
    # value, min(.) picks the plan and the brake passes through at full rate.
    target_accel = float(target_accel)
    if target_accel <= self._last_target_accel:
      jmax = BRAKE_DEEPENING_JERK[self._personality]
      return self._clean_accel(max(target_accel, self._last_target_accel - jmax * DT_MDL))
    return self._slew_up(target_accel)

  def _slew_up(self, target_accel: float) -> float:
    # Releasing the brake / accelerating: rate-limit the rise (release jerk on the brake side, the
    # personality accel-rise jerk on the throttle side).
    if self._last_target_accel < 0.0:
      released = min(target_accel, self._last_target_accel + BRAKE_RELEASE_JERK * DT_MDL)
      if released <= 0.0:
        return self._clean_accel(released)
      return self._clean_accel(min(target_accel, ACCEL_RISE_JERK[self._personality] * DT_MDL))
    step = ACCEL_RISE_JERK[self._personality] * DT_MDL
    return self._clean_accel(min(target_accel, self._last_target_accel + step))

  def _finalize(self, target_accel: float) -> float:
    target_accel = self._clean_accel(target_accel)
    self._last_target_accel = target_accel
    self._since_brake_frames = 0 if target_accel < GAS_SUPPRESS_BRAKE_THR else self._since_brake_frames + 1
    return target_accel

  @staticmethod
  def _clean_accel(accel: float) -> float:
    accel = float(accel)
    return 0.0 if abs(accel) < _ZERO_ACCEL_EPS else accel

  def enabled(self) -> bool:
    return self._enabled

  def personality(self):
    return self._personality

  def max_accel(self) -> float:
    return self.get_max_accel(self._v_ego)

  def brake_need(self) -> float:
    return self._brake_need

  def decel_target(self) -> float:
    return self._decel_target

  def smooth_active(self) -> bool:
    return self._smooth_active

  def bypassed(self) -> bool:
    return self._bypassed

  def comfort_stop_floor(self) -> float:
    return self._stop_floor

  def comfort_stop_active(self) -> bool:
    return self._stop_floor < 0.0
