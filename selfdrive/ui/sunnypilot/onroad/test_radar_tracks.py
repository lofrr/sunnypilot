from cereal import car

from openpilot.selfdrive.ui.sunnypilot.onroad import radar_tracks
from openpilot.selfdrive.ui.sunnypilot.onroad.radar_tracks import format_radar_tracks_onroad_columns, format_radar_tracks_onroad_status, \
                                                                  format_radar_tracks_status


def test_format_radar_tracks_status_none():
  live_tracks = car.RadarData.new_message()

  assert format_radar_tracks_status(live_tracks) == "none"
  assert format_radar_tracks_onroad_status(live_tracks) == "none"
  assert format_radar_tracks_onroad_columns(live_tracks) == ("", "none")


def test_format_radar_tracks_status_range_and_count():
  live_tracks = car.RadarData.new_message()
  live_tracks.trackSources = [{"startAddress": 0x500, "endAddress": 0x51F, "bus": 1, "trackCount": 2}]
  live_tracks.init("points", 2)

  assert format_radar_tracks_status(live_tracks) == "0x500-0x51F · 2 tracks"
  assert format_radar_tracks_onroad_status(live_tracks) == "0x500-0x51F - 2"
  assert format_radar_tracks_onroad_columns(live_tracks) == ("0x500-0x51F -", "2")


def test_format_radar_tracks_status_deduplicates_and_sorts_ranges():
  live_tracks = car.RadarData.new_message()
  live_tracks.trackSources = [
    {"startAddress": 0x500, "endAddress": 0x51F, "bus": 2, "trackCount": 3},
    {"startAddress": 0x210, "endAddress": 0x21F, "bus": 1, "trackCount": 1},
    {"startAddress": 0x500, "endAddress": 0x51F, "bus": 0, "trackCount": 2},
  ]
  live_tracks.init("points", 1)

  assert format_radar_tracks_status(live_tracks) == "0x210-0x21F, 0x500-0x51F · 1 track"
  assert format_radar_tracks_onroad_status(live_tracks) == "0x210-0x21F - 1\n0x500-0x51F - 2\n0x500-0x51F - 3"
  assert format_radar_tracks_onroad_columns(live_tracks) == (
    "0x210-0x21F -\n0x500-0x51F -\n0x500-0x51F -",
    "1\n2\n3",
  )


def test_draw_radar_tracks_applies_screen_offset(monkeypatch):
  live_tracks = car.RadarData.new_message()
  points = live_tracks.init("points", 1)
  points[0].dRel = 10
  points[0].yRel = 1
  points[0].vRel = 2
  points[0].aRel = 0
  drawn_circles = []
  monkeypatch.setattr(radar_tracks.rl, "draw_circle", lambda x, y, size, color: drawn_circles.append((x, y, size)))

  radar_tracks.RadarTracks().draw_radar_tracks(
    live_tracks,
    lambda d_rel, y_rel, z: (20, 30),
    path_offset_z=1.2,
    track_size=3,
    screen_offset=(100, 7),
  )

  assert drawn_circles == [(120, 37, 3)]
