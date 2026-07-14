from cereal import car

from openpilot.selfdrive.ui.sunnypilot.onroad.radar_tracks import format_radar_tracks_onroad_status, format_radar_tracks_status


def test_format_radar_tracks_status_none():
  live_tracks = car.RadarData.new_message()

  assert format_radar_tracks_status(live_tracks) == "none"
  assert format_radar_tracks_onroad_status(live_tracks) == "none"


def test_format_radar_tracks_status_range_and_count():
  live_tracks = car.RadarData.new_message()
  live_tracks.trackSources = [{"startAddress": 0x500, "endAddress": 0x51F, "bus": 1, "trackCount": 2}]
  live_tracks.init("points", 2)

  assert format_radar_tracks_status(live_tracks) == "0x500-0x51F · 2 tracks"
  assert format_radar_tracks_onroad_status(live_tracks) == "0x500-0x51F - 2"


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
