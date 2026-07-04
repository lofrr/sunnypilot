"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Acceleration Personality (ECO / NORMAL / SPORT). Tunes only MPC INPUTS, never the output:
  * positive-accel ceiling + speed-dependent per-cycle open-rate -> tier-scaled take-off from a stop
    (the open-rate is fast near v=0 so launch is never delayed, tapering to a steady-state rate at speed);
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
  RISE_RATE_BP, RISE_RATE_V, STOCK_RISE_RATE, JERK_SCALE_BP, JERK_SCALE_V, TF_WIDEN_V_BP, TF_WIDEN_BASE_V, \
  TF_WIDEN_TIER, TF_WIDEN_MAX, TF_SLEW_PER_S, TF_DECEL_HOLD_A


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
    self._frame += 1

  def reset(self) -> None:
    # Drop the accumulated widen (e.g. on disengage / standstill re-init) so it re-ramps cleanly.
    self._widen = 0.0

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
    # Disabled -> 1.0 -> byte-stock jerk cost. Enabled: relaxes the core MPC's jerk_factor near a stop
    # (tier-scaled), ramping back to 1.0 (stock) by the v=5 knot so cruise/follow jerk is unchanged.
    if not self._enabled:
      return 1.0
    return float(np.interp(v_ego, JERK_SCALE_BP, JERK_SCALE_V[self._personality]))

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
