"""weather_sat.py — Radio Tuna: unattended Meteor LRPT pass recorder.

Leave it running. It predicts every Meteor-M2 pass over your location,
wakes at AOS, records baseband IQ at 137.9 MHz for the whole pass, logs
it, and goes back to sleep — harvesting real samples for the adaptive
LRPT decoder (OQPSK 72k -> Viterbi r=1/2 -> RS(255,223) -> derandomizer,
the same FEC back-half we forged on ATSC).

NOAA APT is dead (NOAA-15/19 decommissioned Aug 2025), so the only live
137 MHz targets are the digital Meteor-M2 birds — which is exactly where
our soft-decision FEC has an edge over stock decoders on weak passes.

Modes:
  passes  — print upcoming passes over the next N hours (NO SDR needed)
  record  — record one capture right now for --secs (SDR sanity check)
  watch   — run forever: sleep -> record each pass above --min-elev -> repeat

Examples:
  python weather_sat.py passes --hours 24
  python weather_sat.py record --secs 30            # is the SDR alive?
  python weather_sat.py watch  --min-elev 15        # harvest all day

Location defaults to the DC metro (coarse). Override per-run:
  set WXSAT_LAT=..  set WXSAT_LON=..  set WXSAT_ALT=..   (or --lat/--lon)
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


def _ensure_sdr_dll_path():
    """Make SoapySDR's driver modules loadable even when launched via a bare
    radioconda python.exe (not an activated shell): add radioconda's
    Library\\bin and the SDRplay API dir to the DLL search path. Without this,
    every SoapySDR module fails LoadLibrary and enumerate() sees 0 devices.
    No-op off Windows / where the dirs are absent (Pi, Linux)."""
    if os.name != "nt":
        return
    root = Path(sys.executable).resolve().parent   # radioconda root
    for p in (root / "Library" / "bin",
              Path(r"C:\Program Files\SDRplay\API\x64"),
              Path(r"C:\Program Files\SDRplay\API")):
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass


_ensure_sdr_dll_path()

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)
CAP_DIR = LAB / "wxsat"
CAP_DIR.mkdir(exist_ok=True)
TLE_CACHE = LAB / "weather_tle.txt"
CAP_LOG = LAB / "wxsat_captures.jsonl"
STATUS_FILE = LAB / "wxsat_status.json"   # heartbeat for the panel

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle"
TLE_MAX_AGE_H = 12.0
SATS = ["METEOR-M2 3", "METEOR-M2 4"]
FREQ_HZ = 137.9e6
DEFAULT_FS = 250_000          # LRPT occupies ~140 kHz; 250 k is comfy (~1 MB/s)
LEAD_S = 20                   # start recording this many s before AOS
TAIL_S = 20                   # ... and keep going this long past LOS

# Observer (coarse DC metro default; override via env or flags)
OBS_LAT = float(os.environ.get("WXSAT_LAT", "38.90"))
OBS_LON = float(os.environ.get("WXSAT_LON", "-77.03"))
OBS_ALT = float(os.environ.get("WXSAT_ALT", "50"))

# SDR defaults (match hd_radio.py: SDRplay RSPdx, Antenna A)
DEF_DRIVER = os.environ.get("WXSAT_DRIVER", "sdrplay")
DEF_ANTENNA = os.environ.get("WXSAT_ANTENNA", "Antenna A")


# --------------------------------------------------------------------------
# orbital prediction  (sgp4 only: TEME -> ECEF via GMST -> topocentric elev)
# --------------------------------------------------------------------------
def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


_STATUS = {"state": "init"}


def write_status(**kw):
    """Merge fields and atomically write the heartbeat the panel reads."""
    _STATUS.update(kw)
    _STATUS["updated"] = utcnow().isoformat(timespec="seconds") + "Z"
    try:
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_STATUS))
        tmp.replace(STATUS_FILE)
    except Exception:
        pass


def load_tles(refresh=True):
    """Return {name: Satrec} for our SATS, refreshing the cache if stale."""
    from sgp4.api import Satrec
    stale = True
    if TLE_CACHE.exists():
        age_h = (time.time() - TLE_CACHE.stat().st_mtime) / 3600.0
        stale = age_h > TLE_MAX_AGE_H
    if refresh and stale:
        try:
            req = urllib.request.Request(TLE_URL, headers={"User-Agent": "radiotuna-wxsat"})
            data = urllib.request.urlopen(req, timeout=30).read()
            TLE_CACHE.write_bytes(data)
            print(f"[tle] refreshed {TLE_CACHE.name} ({len(data)} bytes)")
        except Exception as e:
            print(f"[tle] refresh failed ({e}); using cached copy")
    if not TLE_CACHE.exists():
        sys.exit("[tle] no TLE cache and refresh failed — need internet once")
    lines = [l.rstrip() for l in TLE_CACHE.read_text().splitlines()]
    sats = {}
    for i in range(len(lines) - 2):
        name = lines[i].strip()
        if name in SATS and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            sats[name] = Satrec.twoline2rv(lines[i + 1], lines[i + 2])
    missing = [s for s in SATS if s not in sats]
    if missing:
        print(f"[tle] warning: not found in cache: {missing}")
    return sats


def _observer_ecef():
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    latr = math.radians(OBS_LAT)
    lonr = math.radians(OBS_LON)
    N = a / math.sqrt(1 - e2 * math.sin(latr) ** 2)
    ox = (N + OBS_ALT) * math.cos(latr) * math.cos(lonr)
    oy = (N + OBS_ALT) * math.cos(latr) * math.sin(lonr)
    oz = (N * (1 - e2) + OBS_ALT) * math.sin(latr)
    up = (math.cos(latr) * math.cos(lonr),
          math.cos(latr) * math.sin(lonr),
          math.sin(latr))
    return (ox, oy, oz), up


_OBS_ECEF, _OBS_UP = _observer_ecef()


def elevation_deg(sat, when):
    from sgp4.api import jday
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond * 1e-6)
    e, r, _ = sat.sgp4(jd, fr)
    if e != 0:
        return None
    d = (jd - 2451545.0) + fr
    g = math.radians((280.46061837 + 360.98564736629 * d) % 360.0)
    cg, sg = math.cos(g), math.sin(g)
    x = r[0] * cg + r[1] * sg
    y = -r[0] * sg + r[1] * cg
    z = r[2]
    rx = x * 1000 - _OBS_ECEF[0]
    ry = y * 1000 - _OBS_ECEF[1]
    rz = z * 1000 - _OBS_ECEF[2]
    rng = math.sqrt(rx * rx + ry * ry + rz * rz)
    sinel = (rx * _OBS_UP[0] + ry * _OBS_UP[1] + rz * _OBS_UP[2]) / rng
    return math.degrees(math.asin(max(-1, min(1, sinel))))


def predict_passes(sats, start, hours, min_elev, step_s=30):
    """List of pass dicts sorted by AOS, over [start, start+hours]."""
    end = start + timedelta(hours=hours)
    out = []
    for name, sat in sats.items():
        t = start
        cur = None
        while t <= end:
            el = elevation_deg(sat, t)
            if el is not None:
                if el > 0 and cur is None:
                    cur = {"sat": name, "aos": t, "max": el, "maxt": t}
                elif el > 0:
                    if el > cur["max"]:
                        cur["max"] = el
                        cur["maxt"] = t
                elif cur:
                    cur["los"] = t
                    if cur["max"] >= min_elev:
                        out.append(cur)
                    cur = None
            t += timedelta(seconds=step_s)
    out.sort(key=lambda p: p["aos"])
    return out


def _fmt_local(t, off_h):
    return (t + timedelta(hours=off_h)).strftime("%a %H:%M")


# --------------------------------------------------------------------------
# SDR capture
# --------------------------------------------------------------------------
def open_sdr(freq_hz, fs, driver, antenna, gain_db):
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device(f"driver={driver}")
    try:
        sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    except Exception:
        sdr.setSampleRate(SOAPY_SDR_RX, 0, 2_000_000)   # SDRplay fallback
    actual_fs = sdr.getSampleRate(SOAPY_SDR_RX, 0)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
    if antenna:
        try:
            sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
        except Exception:
            pass
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)   # AGC off
    except Exception:
        pass
    if driver == "sdrplay":
        # SDRplay: lower IFGR = more gain. Map a 0..50 dB request loosely.
        try:
            sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", max(20, 59 - gain_db))
            sdr.writeSetting("rfgain_sel", "4")
        except Exception:
            pass
    else:
        try:
            sdr.setGain(SOAPY_SDR_RX, 0, gain_db)
        except Exception:
            pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    return sdr, st, actual_fs


def _rms_dbfs(i16buf):
    """Signal level of an interleaved-int16 IQ buffer, in dBFS (full = 0)."""
    x = i16buf.astype(np.float32)
    p = float(np.mean(x * x)) + 1e-9
    return 10 * math.log10(p / (32768.0 ** 2))


def capture_to_file(sdr, st, out_path, n_seconds, actual_fs, sat,
                    stop_at=None, live=False):
    """Stream CS16 IQ to out_path. Returns (n_samples, bytes)."""
    import SoapySDR
    n_want = int(n_seconds * actual_fs)
    buf = np.empty(2 * 65536, np.int16)
    got = 0
    start = time.time()
    last_stat = 0.0
    level = -99.0
    with open(out_path, "wb") as f:
        while got < n_want:
            if stop_at and utcnow() >= stop_at:
                break
            r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
            if r.ret > 0:
                f.write(buf[:2 * r.ret].tobytes())
                got += r.ret
                level = _rms_dbfs(buf[:2 * r.ret])
            elif r.ret < 0 and r.ret != -1:   # -1 = timeout, tolerable
                print(f"[cap] stream error {r.ret}, stopping")
                break
            now = time.time()
            if live and now - last_stat >= 1.5:
                last_stat = now
                write_status(state="recording", sat=sat,
                             elapsed_s=round(now - start, 1),
                             target_s=round(n_seconds, 1),
                             mb=round(got * 4 / 1e6, 1),
                             level_dbfs=round(level, 1),
                             file=out_path.name)
    return got, got * 4  # CS16 = 4 bytes/sample


def write_sidecar(out_path, meta):
    Path(str(out_path) + ".json").write_text(json.dumps(meta, indent=2))


def log_capture(meta):
    with open(CAP_LOG, "a") as f:
        f.write(json.dumps(meta) + "\n")


def do_record(freq_hz, secs, args, sat="manual", aos=None, los=None):
    """Open SDR, record, write file + sidecar + log. Returns meta or None."""
    stamp = utcnow().strftime("%Y%m%d_%H%M%S")
    tag = sat.replace(" ", "").replace("-", "")
    out = CAP_DIR / f"lrpt_{tag}_{stamp}.cs16"
    try:
        sdr, st, actual_fs = open_sdr(freq_hz, args.fs, args.driver,
                                      args.antenna, args.gain)
    except Exception as e:
        print(f"[sdr] open failed: {e}")
        write_status(state="error", sat=sat, note=f"SDR open failed: {e}")
        return None
    print(f"[cap] {sat}: recording -> {out.name} @ {freq_hz/1e6:.3f} MHz, "
          f"fs={actual_fs/1e3:.0f}k, up to {secs:.0f}s")
    stop_at = utcnow() + timedelta(seconds=secs)
    try:
        n, nbytes = capture_to_file(sdr, st, out, secs, actual_fs, sat,
                                    stop_at=stop_at, live=True)
    finally:
        try:
            sdr.deactivateStream(st)
            sdr.closeStream(st)
        except Exception:
            pass
    meta = {
        "ts": utcnow().isoformat(timespec="seconds") + "Z",
        "sat": sat, "file": str(out), "freq_hz": freq_hz,
        "fs_hz": actual_fs, "format": "cs16", "n_samples": n,
        "bytes": nbytes, "dur_s": round(n / actual_fs, 1) if actual_fs else 0,
        "aos": aos.isoformat() + "Z" if aos else None,
        "los": los.isoformat() + "Z" if los else None,
        "obs": [OBS_LAT, OBS_LON],
    }
    write_sidecar(out, meta)
    log_capture(meta)
    write_status(state="idle", last_capture={
        "sat": sat, "file": out.name, "dur_s": meta["dur_s"],
        "mb": round(nbytes / 1e6, 1), "ts": meta["ts"]})
    print(f"[cap] done: {meta['dur_s']}s, {nbytes/1e6:.0f} MB")
    return meta


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_passes(args):
    sats = load_tles(refresh=not args.no_refresh)
    ps = predict_passes(sats, utcnow(), args.hours, args.min_elev)
    print(f"\nObserver {OBS_LAT:.2f},{OBS_LON:.2f}  next {args.hours}h  "
          f"(local = UTC{args.utc_off:+d})  min-elev {args.min_elev}deg\n")
    if not ps:
        print("  (no passes above threshold in window)")
        return
    for p in ps:
        dur = (p["los"] - p["aos"]).total_seconds() / 60
        flag = "  <-- GOOD" if p["max"] >= 25 else ""
        print(f"  {_fmt_local(p['maxt'], args.utc_off)} local  {p['sat']:<12}"
              f"  max {p['max']:4.1f}deg  {dur:4.1f}min{flag}")
    print(f"\n{len(ps)} passes. Good (>25deg): "
          f"{sum(1 for p in ps if p['max'] >= 25)}")


def cmd_record(args):
    do_record(args.mhz * 1e6, args.secs, args, sat="manual")


def cmd_watch(args):
    print(f"[watch] observer {OBS_LAT:.2f},{OBS_LON:.2f}  min-elev "
          f"{args.min_elev}deg  freq {FREQ_HZ/1e6:.3f} MHz  fs {args.fs/1e3:.0f}k")
    print("[watch] Ctrl+C to stop. Recording every Meteor pass above threshold.\n")
    while True:
        sats = load_tles(refresh=True)
        ps = predict_passes(sats, utcnow(), args.horizon, args.min_elev)
        if not ps:
            print(f"[watch] no pass in next {args.horizon}h; re-checking in 1h")
            write_status(state="waiting", next_sat=None,
                         note=f"no pass in next {args.horizon}h")
            _sleep_interruptible(3600)
            continue
        nxt = ps[0]
        aos, los = nxt["aos"], nxt["los"]
        rec_start = aos - timedelta(seconds=LEAD_S)
        wait = (rec_start - utcnow()).total_seconds()
        print(f"[watch] next: {nxt['sat']} at "
              f"{_fmt_local(nxt['maxt'], args.utc_off)} local, "
              f"max {nxt['max']:.0f}deg — recording in {wait/60:.1f} min")
        write_status(state="waiting", next_sat=nxt["sat"],
                     next_max_elev=round(nxt["max"], 1),
                     next_aos=aos.isoformat() + "Z",
                     rec_start=rec_start.isoformat() + "Z",
                     next_los=los.isoformat() + "Z",
                     min_elev=args.min_elev, freq_hz=FREQ_HZ)
        if wait > 0:
            _sleep_interruptible(wait)
        secs = (los + timedelta(seconds=TAIL_S) - utcnow()).total_seconds()
        if secs < 30:
            continue   # missed it / too short; loop picks the next
        cap_path = do_record(FREQ_HZ, secs, args, sat=nxt["sat"], aos=aos,
                             los=los)
        # H8 reactivation tripwire: MER-check the pass peak so the day
        # Meteor's LRPT wakes up we KNOW immediately, not days later
        try:
            if isinstance(cap_path, dict) and cap_path.get("file"):
                latest = Path(cap_path["file"])
            else:
                latest = max(CAP_DIR.glob("lrpt_*.cs16"),
                             key=lambda p: p.stat().st_mtime)
            import subprocess as _sp
            r = _sp.run([sys.executable, str(HERE / "lrpt.py"), "decode",
                         "--mid", str(latest)], capture_output=True,
                        text=True, timeout=600)
            blob = (r.stdout or "")
            mer_line = next((l for l in blob.splitlines() if "MER dial" in l),
                            "").strip()
            print(f"[watch] post-pass check: {mer_line}")
            if "locked=True" in blob:
                print("*" * 62)
                print("*** METEOR LRPT LOCK - THE BIRD IS BACK - TELL THE "
                      "HUMAN! ***")
                print("*" * 62)
                write_status(note="METEOR LRPT LOCKED - reactivation caught!")
        except Exception as e:
            print(f"[watch] post-pass check skipped: {e}")
        _sleep_interruptible(10)   # gap so we don't re-detect the same pass


def _sleep_interruptible(seconds):
    """Sleep in short chunks so Ctrl+C stays responsive and the panel keeps a
    fresh liveness heartbeat during the long waits between passes."""
    end = time.time() + seconds
    last_hb = 0.0
    while time.time() < end:
        time.sleep(min(5, max(0.1, end - time.time())))
        if time.time() - last_hb >= 30:
            write_status()          # bump 'updated' so liveness is provable
            last_hb = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, help="observer latitude (override)")
    ap.add_argument("--lon", type=float, help="observer longitude (override)")
    ap.add_argument("--utc-off", type=int, default=-4, help="local = UTC + this")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("passes", help="print upcoming passes (no SDR)")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--min-elev", type=float, default=15)
    p.add_argument("--no-refresh", action="store_true")

    r = sub.add_parser("record", help="record one capture now")
    r.add_argument("--mhz", type=float, default=137.9)
    r.add_argument("--secs", type=float, default=30)
    r.add_argument("--fs", type=float, default=DEFAULT_FS)
    r.add_argument("--driver", default=DEF_DRIVER)
    r.add_argument("--antenna", default=DEF_ANTENNA)
    r.add_argument("--gain", type=float, default=40)

    w = sub.add_parser("watch", help="run forever, record every pass")
    w.add_argument("--min-elev", type=float, default=15)
    w.add_argument("--horizon", type=float, default=14, help="lookahead hours")
    w.add_argument("--fs", type=float, default=DEFAULT_FS)
    w.add_argument("--driver", default=DEF_DRIVER)
    w.add_argument("--antenna", default=DEF_ANTENNA)
    w.add_argument("--gain", type=float, default=40)
    w.add_argument("--utc-off", type=int, default=-4)

    args = ap.parse_args()
    global OBS_LAT, OBS_LON, _OBS_ECEF, _OBS_UP
    if args.lat is not None:
        OBS_LAT = args.lat
    if args.lon is not None:
        OBS_LON = args.lon
    _OBS_ECEF, _OBS_UP = _observer_ecef()

    if args.cmd == "passes":
        cmd_passes(args)
    elif args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "watch":
        cmd_watch(args)


if __name__ == "__main__":
    main()
