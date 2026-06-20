"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Sequence

import numpy as np

from cereal import messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  NORMAL, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, A_CRUISE_MAX_V, RISE_RATE, SMOOTH_DECEL_BP, \
  STOCK_A_CRUISE_MAX_V, STOCK_RISE_RATE, \
  SMOOTH_DECEL_V, BRAKE_DEEPENING_JERK, BRAKE_RELEASE_JERK, ACCEL_RISE_JERK, SMOOTH_DECEL_LOOKAHEAD_T, \
  MIN_SMOOTH_BRAKE_NEED, HARD_BRAKE_TARGET_ACCEL, HARD_BRAKE_NEED, STOP_IMMINENT_VEGO, STOP_IMMINENT_LOOKAHEAD_T, \
  ONSET_JERK0, ONSET_JERK_GAIN, ONSET_GAP_SOFT, ONSET_GAP_GAIN, ONSET_JERK_MAX, ONSET_HANDBACK_JERK, \
  SOFT_ONSET_MAX_BRAKE_NEED, SOFT_ONSET_MAX_INSTANT_ACCEL, SOFT_ONSET_REARM_FRAMES

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
    # convex brake-onset shaper state
    self._onset_latched = False # sticky: True once an onset goes firm -> no re-soften until sustained release
    self._onset_release = 0     # consecutive non-deepening frames (sticky re-arm debounce)
    self._soft_active = False   # True iff the convex shaper governed this frame's output (bypasses min(.,raw))
    self._soft_episode = False  # True while a soft onset is open (incl. closing its gap) -> own deepening, no snap
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
    self._soft_active = False

    # The convex onset shaper runs ONLY for ECO/SPORT (NORMAL and disabled are stock). Reset its state
    # whenever it cannot run so nothing leaks across a personality toggle or a passthrough interlude.
    if not (self._enabled and self._personality != NORMAL):
      self._reset_onset()

    # Passthroughs (hand the plan straight through, no shaping):
    if reset or not self._enabled or (stock_brake and (raw < 0.0 or self._brake_need >= MIN_SMOOTH_BRAKE_NEED)):
      self._bypassed = False                                  # disabled / reset / blended-e2e braking
      return self._passthrough(raw)
    self._bypassed = self._emergency_bypass(raw, should_stop)
    if self._bypassed:                                        # a hard brake must never be softened
      return self._stand_down(raw)
    if self._stop_imminent(speed_trajectory, t_idxs):         # stop coming -> stock decel, no coast/creep
      return self._stand_down(raw)

    # Front-load a gentle early brake when a deeper brake is predicted ahead. The convex shaper owns the
    # output when it governed this frame (soft_active); otherwise never weaker than the plan.
    if self._brake_need >= MIN_SMOOTH_BRAKE_NEED:
      self._smooth_active = True
      self._decel_target = self.get_decel_target(self._brake_need)
      slewed = self._slew(min(raw, self._decel_target))
      return self._finalize(slewed if self._soft_active else min(slewed, raw))

    # Below the smooth-brake threshold: track the plan, never weaker than it while braking.
    slewed = self._slew(raw)
    if self._soft_active or raw >= 0.0:
      return self._finalize(slewed)
    return self._finalize(min(slewed, raw))

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
    target_accel = float(target_accel)
    p = self._personality
    jmax = BRAKE_DEEPENING_JERK[p]
    deepening = target_accel <= self._last_target_accel
    if not deepening:
      # genuine release / coast: close any soft episode, advance the re-arm debounce, unlatch after a
      # sustained release.
      self._soft_episode = False
      self._onset_release += 1
      if self._onset_release >= SOFT_ONSET_REARM_FRAMES:
        self._onset_latched = False
      return self._slew_up(target_accel)
    self._onset_release = 0
    # NORMAL (and disabled, forced to NORMAL) -> stock constant-jerk linear deepening, byte-exact.
    if p == NORMAL:
      return self._clean_accel(max(target_accel, self._last_target_accel - jmax * DT_MDL))
    return self._slew_convex(target_accel, jmax)

  def _onset_soft_armed(self, target_accel: float) -> bool:
    # Gentle non-emergency onset. Armed from the FIRST deepening tick (no brake_need lower gate, so the
    # gentle bite lands on the actual onset, not after the plan has already deepened). Two upper gates
    # keep it off firm/deep braking: the 3s-lookahead brake_need ceiling AND the instantaneous raw depth.
    return (self._enabled and self._personality != NORMAL and
            0.0 < self._brake_need < SOFT_ONSET_MAX_BRAKE_NEED and
            target_accel > SOFT_ONSET_MAX_INSTANT_ACCEL)

  def _slew_convex(self, target_accel: float, jmax: float) -> float:
    # target_accel is the effective plan to track (raw, or min(raw, decel_target) on the smooth branch).
    # Dispatch: armed -> gentle bite; firm zone with an open soft gap -> fast hand-back; else stock.
    last = self._last_target_accel
    gap = max(0.0, last - target_accel)   # m/s^2 currently shallower than the plan (last,target both <=0)
    soft_armed = self._onset_soft_armed(target_accel)
    if soft_armed and not self._onset_latched:
      return self._onset_bite(target_accel, last, gap)
    if soft_armed:                                  # firm/deep zone -> latch off further (re)arming
      self._onset_latched = True
    if self._soft_episode and gap > _ZERO_ACCEL_EPS:
      return self._onset_handback(target_accel)
    self._soft_episode = False                      # no open gap: NEVER soften a fresh firm brake
    return self._clean_accel(max(target_accel, last - jmax * DT_MDL))   # stock; caller does min(.,raw)

  def _onset_bite(self, target_accel: float, last: float, gap: float) -> float:
    # Gentle convex onset. Depth-proportional jerk: gentle ONSET_JERK0 at the bite (a~0), growing with
    # current decel depth -- da/dt = j0 + k*a integrates to a(t) = (j0/k)*(exp(k*t)-1), the exponential-
    # growth profile. A stateless instantaneous-gap catch-up adds bounded jerk once realized lags the plan
    # by more than ONSET_GAP_SOFT, hard-capped at ONSET_JERK_MAX so even the catch is never a grab.
    p = self._personality
    self._soft_episode = True
    jerk = ONSET_JERK0[p] + ONSET_JERK_GAIN[p] * abs(last)
    jerk = min(jerk + ONSET_GAP_GAIN[p] * max(0.0, gap - ONSET_GAP_SOFT[p]), ONSET_JERK_MAX[p])
    out = max(last - jerk * DT_MDL, target_accel)   # never deeper than the plan -> only softer-or-equal
    if out <= target_accel + _ZERO_ACCEL_EPS:       # gap closed -> episode complete
      self._soft_episode = False
    self._soft_active = True
    return self._clean_accel(out)

  def _onset_handback(self, target_accel: float) -> float:
    # Plan left the gentle zone but a soft gap is still open: close it FAST (firm, jerk-limited so it is
    # not a snap) so the output catches the plan before braking gets firm -> no late-brake lag.
    out = max(self._last_target_accel - ONSET_HANDBACK_JERK[self._personality] * DT_MDL, target_accel)
    if out <= target_accel + _ZERO_ACCEL_EPS:
      self._soft_episode = False
    self._soft_active = True
    return self._clean_accel(out)

  def _reset_onset(self) -> None:
    self._onset_latched = False
    self._onset_release = 0
    self._soft_active = False
    self._soft_episode = False

  def _slew_up(self, target_accel: float) -> float:
    if self._last_target_accel < 0.0:
      released = min(target_accel, self._last_target_accel + BRAKE_RELEASE_JERK * DT_MDL)
      if released <= 0.0:
        return self._clean_accel(released)
      return self._clean_accel(min(target_accel, ACCEL_RISE_JERK[self._personality] * DT_MDL))
    step = ACCEL_RISE_JERK[self._personality] * DT_MDL
    return self._clean_accel(min(target_accel, self._last_target_accel + step))

  def _passthrough(self, target_accel: float) -> float:
    self._smooth_active = False
    self._soft_active = False
    return self._finalize(target_accel)

  def _stand_down(self, target_accel: float) -> float:
    # clear shaper state and hand the plan straight through (emergency / stop-imminent)
    self._reset_onset()
    return self._passthrough(target_accel)

  def _finalize(self, target_accel: float) -> float:
    target_accel = self._clean_accel(target_accel)
    self._last_target_accel = target_accel
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
