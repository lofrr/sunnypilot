"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Acceleration Personality (ECO / NORMAL / SPORT). Tunes only MPC INPUTS, never the output:
  * positive-accel ceiling + speed-dependent per-cycle open-rate -> tier-scaled take-off from a stop
    (the open-rate is fast near v=0 so launch is never delayed, tapering to a steady-state rate at speed);
  * jerk-cost relaxation (scales the core MPC's jerk_factor) -> smooth accel/decel onset: near a stop, on
    any fresh accel<->decel direction change, when the tracked lead is itself braking hard, or when the gap
    is closing fast for any other reason (cut-in, ego overtaking a slower lead) -- the last one is the only
    proactive trigger keyed on an MPC INPUT (vRel) rather than a_ego's own realized sign flip, so it can
    soften the very first brake jab instead of only the recovery after it. The lead-keyed factors are
    TRANSIENT (dip on a fresh/escalating trigger, then ease back to 1.0 over a fixed window regardless of
    whether the raw trigger is still active) rather than a level-level pin -- a sustained low jerk_scale
    destabilizes the MPC's real-time-iteration re-solve into an oscillation (closed-loop-verified: a pinned
    0.6 factor produced a 30+ m/s^3 jerk whipsaw where stock stayed at 0), so only the onset gets softened,
    matching the "first jab" intent, never the sustained middle of a real braking episode;
  * add-only, speed-dependent follow-gap widen on the MPC t_follow -> earlier/gentler braking, roomier gap;
  * sticky should_stop hysteresis -> no stop-and-go gas-brake-gas-brake.
Add-only gap => desired distance >= stock => braking >= stock. Disabled => stock everywhere (byte-stock).
"""

import numpy as np

from cereal import messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.accel_personality.constants import \
  NORMAL, PERSONALITY_MIN, PERSONALITY_MAX, A_CRUISE_MAX_BP, A_CRUISE_MAX_V, STOCK_A_CRUISE_MAX_V, \
  RISE_RATE_BP, RISE_RATE_V, STOCK_RISE_RATE, JERK_SCALE_BP, JERK_SCALE_V, ONSET_DEADBAND, ONSET_RAMP_S, \
  ONSET_FLOOR, RELAX_RAMP_S, LEAD_BRAKE_ALEAD_BP, LEAD_BRAKE_FACTOR_V, CLOSING_VREL_BP, CLOSING_FACTOR_V, \
  TF_WIDEN_V_BP, TF_WIDEN_BASE_V, TF_WIDEN_TIER, TF_WIDEN_MAX, TF_SLEW_PER_S, TF_DECEL_HOLD_A


class _OnsetRelax:
  # Detects a fresh accel<->decel direction change on aEgo (a real, causal signal -- never the MPC's own
  # solved target) and relaxes toward a tier-scaled floor immediately, then eases linearly back to 1.0 over
  # ONSET_RAMP_S. Feeds the MPC's cost weights for the cycles that follow; never touches this cycle's output.
  def __init__(self):
    self._prev_sign = 0
    self._ramp = 1.0

  def reset(self) -> None:
    self._prev_sign = 0
    self._ramp = 1.0

  def update(self, a_ego: float, floor: float) -> float:
    sign = 0
    if a_ego > ONSET_DEADBAND:
      sign = 1
    elif a_ego < -ONSET_DEADBAND:
      sign = -1

    if sign != 0 and sign != self._prev_sign:
      self._ramp = floor
      self._prev_sign = sign
    else:
      self._ramp = min(1.0, self._ramp + (1.0 - floor) * (DT_MDL / ONSET_RAMP_S))
    return self._ramp


class _TransientRelax:
  # Same shape as _OnsetRelax's ramp, but triggered by a level (a raw tier-scaled factor computed fresh each
  # cycle) instead of a sign flip: on a fresh dip (factor newly below 1.0) or a further escalation (factor
  # drops below the lowest value already seen this episode), snap down to it; otherwise ease back toward 1.0
  # over RELAX_RAMP_S REGARDLESS of whether the raw factor is still low. Without this, a raw factor that
  # stays pinned at its floor for the duration of a real, sustained closing/braking episode (not just the
  # first instant) keeps jerk_scale low for seconds at a time -- closed-loop-verified to destabilize the
  # MPC's real-time-iteration re-solve into an oscillation (30+ m/s^3 jerk vs 0 for the identical scenario
  # with the factor disabled). Ramping back after the initial dip keeps the "soften the first jab" intent
  # without holding a low jerk cost through the sustained middle of a real episode.
  def __init__(self):
    self._was_active = False
    self._episode_floor = 1.0
    self._ramp = 1.0

  def reset(self) -> None:
    self._was_active = False
    self._episode_floor = 1.0
    self._ramp = 1.0

  def update(self, raw_factor: float, ramp_s: float) -> float:
    active = raw_factor < 1.0 - 1e-3
    if active and (not self._was_active or raw_factor < self._episode_floor):
      self._episode_floor = raw_factor
      self._ramp = raw_factor
    else:
      self._ramp = min(1.0, self._ramp + DT_MDL / ramp_s)
      if not active:
        self._episode_floor = 1.0
    self._was_active = active
    return self._ramp


class AccelController:
  def __init__(self, CP: structs.CarParams, mpc=None, params=None):
    # CP/mpc accepted for the planner's constructor signature; unused (shapes MPC inputs only).
    self._params = params or Params()
    self._frame = 0
    self._enabled = False
    self._personality = NORMAL
    self._v_ego = 0.0
    self._a_ego = 0.0
    self._widen = 0.0                     # current slewed follow-gap widen (s), add-only
    self._t_follow = 0.0                  # last t_follow handed to the MPC (telemetry)
    self._onset_relax = _OnsetRelax()
    self._lead_brake_relax = _TransientRelax()
    self._closing_relax = _TransientRelax()
    self._onset_factor = 1.0
    self._lead_brake_factor = 1.0
    self._closing_factor = 1.0
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
    self._v_ego = float(sm['carState'].vEgo)
    self._a_ego = float(sm['carState'].aEgo)

    if self._enabled:
      lead = sm['radarState'].leadOne
      self._onset_factor = self._onset_relax.update(self._a_ego, ONSET_FLOOR[self._personality])
      self._lead_brake_factor = self._lead_brake_relax.update(self._get_lead_brake_factor(lead), RELAX_RAMP_S)
      self._closing_factor = self._closing_relax.update(self._get_closing_factor(lead), RELAX_RAMP_S)
    else:
      self._onset_relax.reset()
      self._lead_brake_relax.reset()
      self._closing_relax.reset()
      self._onset_factor = 1.0
      self._lead_brake_factor = 1.0
      self._closing_factor = 1.0

    self._frame += 1

  def _get_lead_brake_factor(self, lead) -> float:
    # Raw, uneased tier-scaled factor for THIS cycle -- fed through _TransientRelax in update(), never used
    # directly as the published factor (see the class docstring for why: a level-pinned floor destabilizes
    # the MPC's re-solve over a sustained episode).
    if not lead.status:
      return 1.0
    return float(np.interp(lead.aLeadK, LEAD_BRAKE_ALEAD_BP, LEAD_BRAKE_FACTOR_V[self._personality]))

  def _get_closing_factor(self, lead) -> float:
    # Raw, uneased tier-scaled factor for THIS cycle -- see _get_lead_brake_factor's note.
    if not lead.status:
      return 1.0
    return float(np.interp(lead.vRel, CLOSING_VREL_BP, CLOSING_FACTOR_V[self._personality]))

  def reset(self) -> None:
    # Drop the accumulated widen (e.g. on disengage / standstill re-init) so it re-ramps cleanly.
    self._widen = 0.0
    self._onset_relax.reset()
    self._lead_brake_relax.reset()
    self._closing_relax.reset()
    self._onset_factor = 1.0
    self._lead_brake_factor = 1.0
    self._closing_factor = 1.0

  def get_max_accel(self, v_ego: float) -> float:
    # Disabled -> stock ceiling (off == stock, independent of the NORMAL profile so NORMAL is free to differ).
    table = A_CRUISE_MAX_V[self._personality] if self._enabled else STOCK_A_CRUISE_MAX_V
    return float(np.interp(v_ego, A_CRUISE_MAX_BP, table))

  def get_rise_rate(self, v_ego: float) -> float:
    # Disabled -> stock ceiling open-rate (off == stock, independent of the NORMAL profile).
    # Speed-dependent: fast near a stop (non-binding, no launch delay), tapering to the steady-state rate.
    if not self._enabled:
      return STOCK_RISE_RATE
    return float(np.interp(v_ego, RISE_RATE_BP, RISE_RATE_V[self._personality]))

  def get_jerk_scale(self, v_ego: float) -> float:
    # Disabled -> 1.0 -> byte-stock jerk cost. Enabled: takes the most-relaxed of four tier-scaled factors --
    # near a stop (v_ego), a fresh accel<->decel onset (any speed), a hard-braking lead, and a fast-closing
    # gap (any cause) -- each never exceeding 1.0 (stock), so this only ever relaxes jerk cost, never tightens
    # it beyond stock.
    if not self._enabled:
      return 1.0
    near_stop = float(np.interp(v_ego, JERK_SCALE_BP, JERK_SCALE_V[self._personality]))
    return min(near_stop, self._onset_factor, self._lead_brake_factor, self._closing_factor)

  def get_t_follow(self, t_follow: float, v_ego: float) -> float:
    # MPC t_follow hook. Adds a slewed, decel-held, speed-dependent comfort widen on top of the stock
    # t_follow. Identity when disabled => byte-stock. Add-only => desired distance >= stock => brake >= stock.
    t_follow = float(t_follow)
    if not self._enabled:
      self._widen = 0.0
      self._t_follow = t_follow
      return t_follow

    target = float(np.interp(v_ego, TF_WIDEN_V_BP, TF_WIDEN_BASE_V)) * TF_WIDEN_TIER[self._personality]
    target = min(target, TF_WIDEN_MAX)
    step = TF_SLEW_PER_S * DT_MDL

    if self._a_ego <= TF_DECEL_HOLD_A and target < self._widen:
      pass                                              # decel-hold: don't ease the gap in while braking
    elif target > self._widen:
      self._widen = min(target, self._widen + step)     # open the gap, slewed
    else:
      self._widen = max(target, self._widen - step)     # close the gap, slewed

    self._widen = max(0.0, self._widen)                 # add-only guard
    self._t_follow = t_follow + self._widen
    return self._t_follow

  # --- telemetry (published to cereal LongitudinalPlanSP.acceleration; no control effect) ---
  def enabled(self) -> bool:
    return self._enabled

  def personality(self):
    return self._personality

  def max_accel(self) -> float:
    return self.get_max_accel(self._v_ego)

  def t_follow(self) -> float:
    return self._t_follow

  def follow_widen(self) -> float:
    return self._widen

  def widen_active(self) -> bool:
    return self._enabled and self._widen > 0.005
