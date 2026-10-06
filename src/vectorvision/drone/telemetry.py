"""Pixhawk MAVLink telemetry — read only.

Reads GPS position, height above home and heading from the flight controller over a
SiK radio. It never sends a command, so it cannot affect the aircraft.

Why each field matters downstream:
    lat, lon   where the drone was when the frame was captured
    alt_m      height above ground, which sets the ground scale of each pixel
    hdg_deg    which way the camera was facing, needed to rotate a detection's
               position in the frame into a real compass direction

Without this, a detected pool can only be guessed at by offsetting from the survey
centre, which is why targets came out marked "approximate".
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TelemetryState:
    """Latest usable Pixhawk telemetry."""

    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: Optional[float] = None
    hdg_deg: Optional[float] = None
    timestamp: float = 0.0

    @property
    def has_fix(self) -> bool:
        """A usable GPS position and a known height, including on the ground.

        Altitude is allowed to be zero or slightly negative: the drone sits at 0 m
        before take-off, and the barometer drifts. Requiring alt > 0 would mean no
        telemetry at all until it lifts off, which hides connection problems until
        the worst moment to find them.
        """
        return (self.lat is not None and self.lon is not None
                and self.alt_m is not None)

    @property
    def usable_for_projection(self) -> bool:
        """High enough for a frame position to be projected onto the ground."""
        return self.has_fix and self.alt_m is not None and self.alt_m >= 2.0

    @property
    def age_s(self) -> float:
        return time.time() - self.timestamp if self.timestamp else float("inf")

    def as_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "alt_m": self.alt_m,
                "hdg_deg": self.hdg_deg, "timestamp": self.timestamp}


class PixhawkTelemetry:
    """Read-only MAVLink connection.

        COM5                  Windows serial telemetry radio
        /dev/ttyUSB0          Linux USB serial telemetry radio
        udp:127.0.0.1:14550   SITL or a simulator
    """

    def __init__(self, connection: str, baud: int = 57600, heartbeat_timeout: float = 10.0):
        self.connection_string = connection
        self.baud = int(baud)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.master = None
        self.state = TelemetryState()
        self.messages = 0

    def connect(self) -> bool:
        from pymavlink import mavutil
        print(f"telemetry: connecting to {self.connection_string} at {self.baud} baud")
        self.master = mavutil.mavlink_connection(self.connection_string, baud=self.baud)
        print("telemetry: waiting for heartbeat...")
        hb = self.master.wait_heartbeat(timeout=self.heartbeat_timeout)
        if hb is None:
            self.master = None
            raise TimeoutError(
                f"No heartbeat in {self.heartbeat_timeout:.0f}s. Check the radio is paired, "
                "the port is right, and the baud rate matches (usually 57600).")
        print(f"telemetry: connected (system {self.master.target_system})")
        return True

    def read_once(self, timeout: float = 1.0, want_projection: bool = False):
        """Pump messages until there is a usable fix, or the timeout expires.

        Returns as soon as a fix arrives rather than always burning the full timeout,
        because during live detection this runs once per frame and a wasted second
        per frame would stall the whole pipeline.
        """
        if self.master is None:
            raise RuntimeError("Telemetry is not connected. Call connect() first.")

        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            msg = self.master.recv_match(
                type=["GLOBAL_POSITION_INT", "GPS_RAW_INT", "VFR_HUD"],
                blocking=True, timeout=remaining)
            if msg is None:
                continue
            self.messages += 1
            kind = msg.get_type()
            if kind == "GLOBAL_POSITION_INT":
                self._update_global_position(msg)
            elif kind == "GPS_RAW_INT":
                self._update_gps(msg)
            elif kind == "VFR_HUD":
                self._update_heading(msg)

            ready = self.state.usable_for_projection if want_projection else self.state.has_fix
            if ready:
                return self.state
        return self.state if self.state.has_fix else None

    # ---------------------------------------------------------------- updates
    def _update_global_position(self, msg) -> None:
        lat, lon = getattr(msg, "lat", 0), getattr(msg, "lon", 0)
        if lat != 0 and lon != 0:
            self.state.lat = float(lat) / 1e7
            self.state.lon = float(lon) / 1e7
        # relative_alt is millimetres above the home point, and may be 0 or negative.
        rel = getattr(msg, "relative_alt", None)
        if rel is not None:
            self.state.alt_m = float(rel) / 1000.0
        hdg = getattr(msg, "hdg", 65535)
        if hdg != 65535:
            self.state.hdg_deg = float(hdg) / 100.0
        self.state.timestamp = time.time()

    def _update_gps(self, msg) -> None:
        """Fallback position when GLOBAL_POSITION_INT is not being sent.

        GPS_RAW_INT altitude is above mean sea level, not above the ground, so it is
        deliberately not used for height: projecting a frame onto the ground needs
        height above the ground beneath the drone.
        """
        lat, lon = getattr(msg, "lat", 0), getattr(msg, "lon", 0)
        if lat != 0 and lon != 0:
            self.state.lat = float(lat) / 1e7
            self.state.lon = float(lon) / 1e7
        self.state.timestamp = time.time()

    def _update_heading(self, msg) -> None:
        h = getattr(msg, "heading", None)
        if h is not None and 0 <= h <= 360:
            self.state.hdg_deg = float(h)
        self.state.timestamp = time.time()

    def get_state(self) -> TelemetryState:
        return self.state

    def close(self) -> None:
        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TelemetryLog:
    """Telemetry recorded alongside a video, for analysing the flight afterwards.

    Live flights give a fix per frame. For a recorded flight the fixes are saved as
    a log and matched back to frames by time, so a video can be re-analysed later
    with real coordinates instead of guesses.
    """

    def __init__(self, fixes: list[dict] | None = None):
        self.fixes = sorted(fixes or [], key=lambda f: f.get("timestamp", 0))

    @classmethod
    def load(cls, path):
        import json
        from pathlib import Path
        p = Path(path)
        data = json.loads(p.read_text())
        return cls(data["fixes"] if isinstance(data, dict) else data)

    def save(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps({"fixes": self.fixes}, indent=2))

    def append(self, state: TelemetryState) -> None:
        if state.has_fix:
            self.fixes.append(state.as_dict())

    def at(self, t_epoch: float, max_gap_s: float = 2.0) -> Optional[dict]:
        """Closest fix to a moment in time, or None if none is near enough."""
        if not self.fixes:
            return None
        best = min(self.fixes, key=lambda f: abs(f.get("timestamp", 0) - t_epoch))
        return best if abs(best.get("timestamp", 0) - t_epoch) <= max_gap_s else None


def read_pixhawk(connection: str, baud: int = 57600, timeout: float = 5.0) -> dict:
    """Connect, read one fix, disconnect. Used by the command-line test."""
    t = PixhawkTelemetry(connection=connection, baud=baud)
    try:
        t.connect()
        state = t.read_once(timeout=timeout)
        if state is None:
            raise RuntimeError("Connected to the Pixhawk, but no GPS fix arrived. "
                               "Outdoors with a clear sky view, a cold start can take "
                               "60 seconds or more.")
        return state.as_dict()
    finally:
        t.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test the Pixhawk telemetry link.")
    ap.add_argument("--connection", required=True,
                    help="COM5, /dev/ttyUSB0, or udp:127.0.0.1:14550")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--watch", action="store_true", help="keep printing until Ctrl+C")
    a = ap.parse_args()

    try:
        if a.watch:
            t = PixhawkTelemetry(a.connection, a.baud)
            t.connect()
            print("\n  lat         lon          alt      hdg    age")
            while True:
                s = t.read_once(timeout=1.0)
                if s is None:
                    print("  waiting for a GPS fix...", end="\r")
                else:
                    print(f"  {s.lat:<11.6f} {s.lon:<12.6f} {s.alt_m:>6.1f} m "
                          f"{(s.hdg_deg if s.hdg_deg is not None else float('nan')):>6.1f}  "
                          f"{s.age_s:>4.1f}s", end="\r")
                time.sleep(0.2)
        else:
            r = read_pixhawk(a.connection, a.baud, a.timeout)
            print("\nPixhawk telemetry:")
            print(f"  latitude : {r['lat']}")
            print(f"  longitude: {r['lon']}")
            print(f"  altitude : {r['alt_m']} m above home")
            print(f"  heading  : {r['hdg_deg']} deg")
    except KeyboardInterrupt:
        print("\nstopped.")
    except Exception as exc:
        print(f"\ntelemetry error: {exc}")
        raise SystemExit(1)
