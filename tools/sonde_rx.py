"""sonde_rx.py - wxTuna radiosonde receiver: capture/decode RS41
weather-balloon telemetry (the radiosonde_auto_rx borrow, our stack).

Pipeline: RSPdx (discone, Antenna C) or a saved cf32 -> FM
discriminator -> 48 kHz wav -> rs1729's rs41mod decoder (the same core
radiosonde_auto_rx trusts) -> parsed telemetry lines with position and
weather data.

  python sonde_rx.py decode --cf32 FILE --fs 100000     # offline
  python sonde_rx.py hunt [--mhz 404.000] [--secs 120]  # live capture

The hunt starts where this station decoded frames before (lab/sonde_hits.json,
kept by the hunt itself) and, when lab/sonde_site.json names a launch site to
watch, on the frequency SondeHub reports in the air. Both files are local.
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)
RS41MOD = Path(os.environ.get("RS41MOD",
                              r"Z:\src\rs-decoders\demod\mod\rs41mod.exe"))


def iq_wav(iq, fs, out_path):
    """IQ as stereo 16-bit wav - rs41mod's --iq2 mode decodes this
    directly with its own tuned FSK demod (validated on the sigidwiki
    reference recording: full telemetry)."""
    iq = iq / max(np.percentile(np.abs(iq), 99.5), 1e-9)
    pcm = np.empty((len(iq), 2), np.int16)
    pcm[:, 0] = np.clip(iq.real * 24000, -32000, 32000).astype(np.int16)
    pcm[:, 1] = np.clip(iq.imag * 24000, -32000, 32000).astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())


def fm_demod_wav(iq, fs, out_path, aud=48_000):
    """FSK baseband -> audio wav the rs decoders expect."""
    x = filtfilt(firwin(257, 12e3 / (fs / 2)), [1.0], iq).astype(np.complex64)
    d = np.angle(x[1:] * np.conj(x[:-1])).astype(np.float32)
    d = d / max(np.percentile(np.abs(d), 99), 1e-9)
    from math import gcd
    g = gcd(aud, int(fs))
    a = resample_poly(d, aud // g, int(fs) // g)
    pcm = (np.clip(a, -1, 1) * 24000).astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(aud)
        w.writeframes(pcm.tobytes())
    return len(pcm) / aud


HITS = LAB / "sonde_hits.json"     # what this station decoded before; lab/ is not tracked
SITE = LAB / "sonde_site.json"     # optional {"lat": .., "lon": .., "km": 300}: the launch site to watch
# Two demodulators, because they fail differently (37 saved July recordings, good frames:
# --iq2 573, the FM path with the carrier offset removed 821; the second one is the only
# one that still decodes a sonde more than about 3 kHz off the aim).
DECODERS = (["--iq2"], ["--IQ", "0.0", "--lpIQ", "--dc"])


def decode_wav(wav_path, verbose=True, iq=False, opts=None):
    """Run rs41mod, return (frames, raw output lines)."""
    cmd = [str(RS41MOD), "-vv", "--ecc", "--crc", "--ptu2", "--dewp"]
    if opts:
        cmd += list(opts)
    elif iq:
        cmd.append("--iq2")
    r = subprocess.run(cmd + [str(wav_path)],
                      capture_output=True, text=True, timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    frames = [ln for ln in out.splitlines() if "lat:" in ln or "h:" in ln]
    if verbose:
        for ln in frames[:12]:
            print("  " + ln.strip()[:190])
        if not frames:
            tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
            for ln in tail:
                print("  (no frames) " + ln.strip()[:90])
    return frames, out


def decode_best(wav_path):
    """The same pass through each demodulator; the one with the most frames wins."""
    best, best_out, best_name = [], "", " ".join(DECODERS[0])
    for opts in DECODERS:
        frames, out = decode_wav(wav_path, verbose=False, opts=opts)
        if len(frames) > len(best):
            best, best_out, best_name = frames, out, " ".join(opts)
    return best, best_out, best_name


def cmd_decode(args):
    iq = np.fromfile(args.cf32, dtype=np.complex64)
    print(f"[decode] {len(iq)/args.fs:.1f}s from {args.cf32}")
    wav = LAB / "sonde_iq.wav"
    iq_wav(iq, args.fs, wav)
    frames, _, how = decode_best(wav)
    for ln in frames[:12]:
        print("  " + ln.strip()[:190])
    if not frames:                      # fallback: our own FM demod path
        wav2 = LAB / "sonde_demod.wav"
        fm_demod_wav(iq, args.fs, wav2)
        frames, _ = decode_wav(wav2)
        how = "own FM demod"
    print(f"[decode] {len(frames)} telemetry lines ({how})")
    return 0 if frames else 1


# ── WHERE TO LISTEN (2026-11-10) ────────────────────────────────────────────
# Counted over 165 scheduled hunts: every flight that was ever decoded was on
# one frequency, and it was never the one the hunt was started on. The hits
# came only on days the band scan happened to pick it; the scan re-aimed 94
# times and 45 of those went to two local carriers that are not sondes. So the
# station now remembers where frames were decoded (lab/sonde_hits.json) and
# starts there; the band scan is a side trip of one short pass, and a
# frequency the scan keeps picking with nothing to decode is struck off.
def load_hits():
    try:
        with open(HITS, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    d.setdefault("hits", {})
    d.setdefault("duds", {})
    return d


def save_hits(d):
    tmp = str(HITS) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, str(HITS))


def fkey(mhz):
    return "%.3f" % mhz


def home_freqs(d):
    """Frequencies that carried decoded frames before, most flights first."""
    rows = sorted(d["hits"].items(), key=lambda kv: (-kv[1].get("flights", 0), kv[0]))
    return [float(k) for k, _ in rows]


def struck_off(d, mhz, limit=4):
    """True for a frequency tried `limit` times as a side trip with never a frame."""
    k = fkey(mhz)
    return k not in d["hits"] and d["duds"].get(k, 0) >= limit


def sondehub_freqs():
    """Frequencies of the RS41 sondes SondeHub's listeners report in the air around the
    point in lab/sonde_site.json, nearest first; the ones that have come down are left
    out. [] with no such file or no answer.
    The point is the launch site being watched, never this station's own position."""
    try:
        with open(SITE, encoding="utf-8") as f:
            site = json.load(f)
        lat0, lon0 = float(site["lat"]), float(site["lon"])
        url = ("https://api.v2.sondehub.org/sondes?lat=%.2f&lon=%.2f&distance=%d&last=600"
               % (lat0, lon0, int(float(site.get("km", 300)) * 1000)))
        req = urllib.request.Request(url, headers={"User-Agent": "wxTuna sonde_rx",
                                                   "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=10) as r:
            blob = r.read()
        if blob[:2] == bytes([0x1F, 0x8B]):
            blob = gzip.decompress(blob)
        data = json.loads(blob.decode("utf-8"))
    except Exception:
        return []
    rows = []
    for s in data.values():
        try:
            if not str(s.get("type", "")).startswith("RS41"):
                continue
            mhz = float(s.get("tx_frequency") or s["frequency"])
            mhz = round(mhz * 100) / 100.0
            la, lo = math.radians(float(s["lat"])), math.radians(float(s["lon"]))
            a = (math.sin((la - math.radians(lat0)) / 2) ** 2
                 + math.cos(la) * math.cos(math.radians(lat0))
                 * math.sin((lo - math.radians(lon0)) / 2) ** 2)
            km = 2 * 6371.0 * math.asin(math.sqrt(a))
            if float(s.get("alt", 0.0)) < 1000 and km > 40:      # landed, or about to
                continue
            rows.append((km, mhz, float(s.get("alt", 0.0))))
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort()
    out = []
    for km, mhz, alt in rows:
        if mhz not in [m for m, _, _ in out]:
            out.append((mhz, km, alt))
    return out


def radio_suspects():
    """Programs that may be holding the radio: Python and the SDR applications running now."""
    if sys.platform != "win32":
        return []
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match "
          "'python|dect2|SDRuno|sdrpp|SDRconnect|CubicSDR|sdrangel|rsp_tcp|SoapySDR' } | "
          "ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.Name, $_.CommandLine }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=40)
    except Exception:
        return []
    me = str(os.getpid())
    return [ln.strip()[:170] for ln in (r.stdout or "").splitlines()
            if ln.strip() and ln.split()[0] != me]


def band_scan(sdr, st, buf, fs_back, skip, SOAPY_SDR_RX):
    """3 s of the whole 399-407 MHz band: the strongest signal with a sonde's FSK width,
    leaving out the frequencies in `skip`. MHz on the 10 kHz raster, or None."""
    sdr.setSampleRate(SOAPY_SDR_RX, 0, 8e6)
    sdr.setFrequency(SOAPY_SDR_RX, 0, 403e6)
    ch2 = []
    t1 = time.time()
    while time.time() - t1 < 3:
        r2 = sdr.readStream(st, [buf], len(buf), timeoutUs=800000)
        if r2.ret > 0:
            ch2.append(buf[:r2.ret].copy())
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs_back)
    xs = np.concatenate(ch2) if ch2 else np.zeros(4096, np.complex64)
    nn2 = 1 << 20
    if len(xs) < nn2:
        return None
    S = np.abs(np.fft.fftshift(np.fft.fft(xs[:nn2] * np.hanning(nn2))))**2
    fgrid = (np.fft.fftshift(np.fft.fftfreq(nn2, 1/8e6)) + 403e6)
    P = 10*np.log10(S + 1e-20)
    floor = np.median(P)
    Pm = P.copy()
    per_hz = nn2 / 8e6
    for mhz in skip:
        i0 = int((mhz * 1e6 - 403e6 + 4e6) * per_hz)
        Pm[max(0, i0 - 4000):max(0, i0 + 4000)] = floor
    for _k in range(8):
        i2 = int(np.argmax(Pm))
        if Pm[i2] < floor + 12:
            break
        span = np.where(P[max(0, i2-3000):i2+3000] > P[i2]-15)[0]
        bw = (span.max()-span.min())*(8e6/nn2) if len(span) else 0
        if 3000 < bw < 40000:      # sonde-like FSK width
            return round(fgrid[i2] / 1e4) / 100.0
        Pm[max(0, i2-4000):i2+4000] = floor
    return None


def cmd_hunt(args):
    """Live: capture on the discone, decode, log, repeat."""
    sys.path.insert(0, os.environ.get("RADIO_LOCK_DIR", "Z:/src/gr-radiotuna/tools"))
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    t_hunt0 = time.time()
    print(f"[hunt] {time.strftime('%Y-%m-%d %H:%M:%S')} start", flush=True)

    # ── WARDEN REGISTRATION (2026-07-30) ────────────────────────────────────
    # Why this exists: the 06:40 hunt on 7/30 died instantly with
    # "no available RSP devices found". prop_atlas had begun a sweep the same
    # minute (10:40:29Z) and holds the radio for ~13 MINUTES, while the retry
    # ladder below only spans ~12 s.
    #
    # The fix is NOT a longer ladder. prop_atlas is already a good citizen — it
    # acquires via radio_lock at priority 20 (background) and polls
    # should_yield() mid-sweep — but this hunt opened the device DIRECTLY and
    # never registered intent, so the want-file was never written, the atlas
    # never learned anyone wanted the radio, and the cooperative preemption that
    # was sitting there ready to work could not fire.
    #
    # A scheduled sonde pass is a timed event that cannot be retried later (the
    # balloon is gone), so it takes priority 100 — the sat-pass tier — and waits
    # for the holder to yield rather than racing it. Degrades gracefully: if
    # radio_lock is unavailable we fall through to the bare open exactly as
    # before, so this can never be the reason a hunt fails. (8/20: an agent
    # proposed 80 + stand-down; reverted — the 100/best-effort policy is
    # deliberate, Felbs's call to change.)
    _lock = None
    radio_lock = None
    try:
        import radio_lock
        _lock = radio_lock.Holder("sonde_rx", "radiosonde hunt",
                                  100, wait_s=float(os.environ.get(
                                      "SONDE_LOCK_WAIT", "180")))
        _lock.__enter__()
        if not getattr(_lock, "ok", True):
            h = radio_lock.status() or {}
            print(f"[hunt] warden: radio busy ({h.get('owner','?')}, "
                  f"{h.get('purpose','?')}) and would not yield - "
                  f"continuing anyway (best effort)", flush=True)
        else:
            print("[hunt] warden: radio acquired at priority 100 (sat-pass tier)",
                  flush=True)
            try:
                radio_lock.clear_stop("sonde_rx")
            except Exception:
                pass
    except ImportError:
        print("[hunt] warden unavailable (radio_lock not importable) — "
              "opening directly", flush=True)
    except Exception as e:
        print(f"[hunt] warden acquire failed ({e}) — opening directly", flush=True)

    def let_go():
        if _lock is not None:
            try:
                _lock.__exit__(None, None, None)
                print("[hunt] warden: radio released", flush=True)
            except Exception:
                pass

    def asked_to_leave():
        """The warden's word: a higher-priority waiter, or a stop request for this hunt."""
        if radio_lock is None or _lock is None:
            return None
        try:
            radio_lock.heartbeat()
            why = radio_lock.should_yield()
            if not why and radio_lock.stop_requested("sonde_rx"):
                radio_lock.clear_stop("sonde_rx")
                why = "stop requested"
            return why
        except Exception:
            return None

    # ── OPENING THE RADIO (2026-11-10) ──────────────────────────────────────
    # 128 of 165 scheduled hunts ended right here: "no available RSP devices
    # found" after a 60 s ladder, with the warden lock free, so the holder was
    # a program that does not use the warden. A hunt that gives up after one
    # minute of a 100 minute window has thrown the flight away; this one keeps
    # asking every 15 s for as long as a pass can still fit, says who is
    # running that could have the radio, and still leaves when the warden asks.
    window = args.duration * 60 if args.duration else 1800
    sdr, tries = None, 0
    while sdr is None:
        try:
            sdr = SoapySDR.Device("driver=sdrplay")
        except Exception as e:
            tries += 1
            left = window - (time.time() - t_hunt0)
            if tries == 1 or tries % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] radio will not open ({e}); "
                      f"try {tries}, {left/60:.0f} min of the window left", flush=True)
            if tries == 2:
                for ln in radio_suspects():
                    print("[hunt]   running now: " + ln, flush=True)
            why = asked_to_leave()
            if why or left < args.secs + 60:
                print(f"[hunt] giving up on the radio after {tries} tries "
                      f"({why or 'window over'}): NO HUNT this window", flush=True)
                let_go()
                return 2
            time.sleep(15)
    if tries:
        print(f"[{time.strftime('%H:%M:%S')}] radio opened on try {tries + 1}", flush=True)

    # ── WHERE TO LISTEN ─────────────────────────────────────────────────────
    hits = load_hits()
    if args.pin:
        homes = [args.mhz]
        print(f"[hunt] pinned to {args.mhz:.3f} MHz", flush=True)
    else:
        homes = home_freqs(hits)
        if homes:
            n = hits["hits"][fkey(homes[0])].get("flights", 0)
            print(f"[hunt] aim {homes[0]:.3f} MHz: frames were decoded there on {n} "
                  f"flight(s); --mhz {args.mhz:.3f} is a side trip", flush=True)
        if args.mhz not in homes and not struck_off(hits, args.mhz):
            homes.append(args.mhz)
        if not homes:
            homes = [args.mhz]
    heard = {"rows": [], "t": 0.0}

    def ask_sondehub():
        heard["rows"] = sondehub_freqs()
        heard["t"] = time.time()

    def sondehub_says():
        """Put what SondeHub hears in the air at the front of the list."""
        rows = heard["rows"]
        if rows and not args.pin:
            mhz, km, alt = rows[0]
            if homes[0] != mhz:
                if mhz in homes:
                    homes.remove(mhz)
                homes.insert(0, mhz)
                print(f"[hunt] SondeHub: an RS41 on {mhz:.3f} MHz is in the air "
                      f"({alt/1000:.1f} km up) - aiming there", flush=True)

    if SITE.exists() and not args.pin:
        ask_sondehub()
        if not heard["rows"]:
            print("[hunt] SondeHub: no RS41 reported in the air yet", flush=True)
        sondehub_says()

    FS = 1e6
    cur = homes[0]
    sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e6 + 100e3)       # DC offside
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, args.antenna)          # discone
    except Exception:
        pass
    # GAIN (2026-11-10): measured on a live sonde, six settings interleaved,
    # same minutes. On the discone the LNA at step 0 put the signal 2.8 dB over
    # the floor where step 4 (the old setting) had it 1.0 dB over: about 5 dB
    # of signal to noise, the difference between no frames and frames. Both TV
    # antenna ports were worse than the discone at either step.
    sdr.setGainMode(SOAPY_SDR_RX, 0, False)
    sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", args.ifgr)
    sdr.writeSetting("rfgain_sel", str(args.rfgain))
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(st)
    print(f"[hunt] {cur:.3f} MHz on {args.antenna}, LNA step {args.rfgain}, IFGR {args.ifgr}, "
          f"{args.secs:.0f}s per pass", flush=True)
    buf = np.zeros(262144, np.complex64)
    logf = LAB / f"sonde_log_{time.strftime('%Y%m%d')}.txt"
    dry = 0                    # passes in a row with nothing
    side = 0                   # side trips taken
    locked = None              # the frequency that gave frames in this hunt
    counted = set()            # frequencies already counted as a flight today
    total = 0
    trip = False               # this pass is a side trip (alternate or band-scan pick)
    asker = None
    try:
        while True:
            if args.duration and (time.time() - t_hunt0) > args.duration * 60:
                print("[hunt] duration reached - standing down", flush=True)
                break
            ch = []
            t0 = time.time()
            _hb = t0
            _yield = None
            secs = min(args.secs, 60) if trip else args.secs
            while time.time() - t0 < secs:
                r = sdr.readStream(st, [buf], len(buf), timeoutUs=800000)
                if r.ret > 0:
                    ch.append(buf[:r.ret].copy())
                # HEARTBEAT (2026-07-31): registering with the warden is not
                # enough — the lock's TTL is 90 s, and a capture that streams
                # for minutes without heartbeating goes STALE mid-window. On
                # 7/31 the atlas swept this hunt's silent lock and tried to
                # open a device the hunt still physically held ("SDR would
                # not open" x2). A holder that cannot be heard from is
                # indistinguishable from a dead one, by design — so speak.
                # (8/20: the want-file is polled too; 11/10: and the stop-file.)
                if time.time() - _hb > 30:
                    _yield = asked_to_leave()
                    _hb = time.time()
                    if _yield:
                        break
            if _yield:
                print(f"[hunt] warden: {_yield} — winding down", flush=True)
                break
            if not ch:
                # a starved pass (stream interrupted - e.g. an API service
                # restart mid-hunt, 8/04) is ONE lost pass, not a lost hunt:
                # log the void honestly and keep hunting.
                print(f"[{time.strftime('%H:%M:%S')}] EMPTY PASS - 0 chunks "
                      f"delivered (stream interrupted?), continuing", flush=True)
                continue
            x = np.concatenate(ch)
            n = np.arange(len(x), dtype=np.float64)
            x = (x * np.exp(-2j * np.pi * (-100e3) / FS * n)).astype(np.complex64)
            y = resample_poly(x, 1, 10).astype(np.complex64)   # 100 kS/s
            wav = LAB / "sonde_live.wav"
            iq_wav(y, 100_000, wav)
            frames, out, how = decode_best(wav)
            stamp = time.strftime("%H:%M:%S")
            if frames:
                print(f"[{stamp}] {len(frames)} frames on {cur:.3f} MHz ({how}):", flush=True)
                for ln in frames[-3:]:
                    print("   " + ln.strip()[:190], flush=True)
                with open(logf, "a", encoding="utf-8") as f:
                    for ln in frames:
                        f.write(f"{stamp} {ln.strip()}\n")
                e = hits["hits"].setdefault(fkey(cur), {"flights": 0, "frames": 0})
                if cur not in counted:
                    counted.add(cur)
                    e["flights"] = e.get("flights", 0) + 1
                e["frames"] = e.get("frames", 0) + len(frames)
                e["last"] = time.strftime("%Y-%m-%d")
                hits["duds"].pop(fkey(cur), None)
                save_hits(hits)
                total += len(frames)
                locked, dry, trip = cur, 0, False
            else:
                dry += 1
                print(f"[{stamp}] no frames on {cur:.3f} MHz (sonde weak/absent)", flush=True)
                if trip and fkey(cur) not in hits["hits"]:      # a home with history is never a dud
                    hits["duds"][fkey(cur)] = hits["duds"].get(fkey(cur), 0) + 1
                    save_hits(hits)
                    if struck_off(hits, cur) and cur in homes[1:]:
                        homes.remove(cur)
                        print(f"[hunt] {cur:.3f} MHz struck off: tried "
                              f"{hits['duds'][fkey(cur)]} times, never a frame", flush=True)
                if total and dry >= 10:
                    print(f"[hunt] {total} frames today and nothing for {dry} passes: "
                          f"the sonde is down or over the horizon - standing down", flush=True)
                    break
                if locked is not None and dry >= 4:
                    locked = None
                nxt, trip = (locked if locked is not None else homes[0]), False
                if locked is None and not args.pin:
                    # SondeHub again every 10 minutes while nothing decodes (a late launch)
                    if SITE.exists() and time.time() - heard["t"] > 600 and (
                            asker is None or not asker.is_alive()):
                        asker = threading.Thread(target=ask_sondehub, daemon=True)
                        asker.start()
                    sondehub_says()
                    nxt = homes[0]
                    # two passes of three at home; the third is a side trip, in turn:
                    # the other frequencies with history, then whatever the band scan sees
                    if dry % 3 == 0:
                        side += 1
                        turn = (side - 1) % len(homes)        # the alternates, then the scan
                        if turn < len(homes) - 1:
                            nxt, trip = homes[1 + turn], True
                        else:
                            skip = [float(m) for m in hits["duds"] if struck_off(hits, float(m))]
                            found = band_scan(sdr, st, buf, FS, skip + [homes[0]], SOAPY_SDR_RX)
                            if found and abs(found - homes[0]) > 0.01:
                                nxt, trip = found, True
                                print(f"[hunt] band scan: sonde-like signal at "
                                      f"{found:.3f} MHz - one short pass there", flush=True)
                cur = nxt
                sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e6 + 100e3)
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sdr.deactivateStream(st)
            sdr.closeStream(st)
        except Exception:
            pass
        # Release the warden hold LAST, after the device is closed — otherwise a
        # waiter (prop_atlas resuming its sweep) can win the lock and try to open
        # a radio we have not let go of yet.
        let_go()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--cf32", required=True)
    d.add_argument("--fs", type=float, default=100_000)
    h = sub.add_parser("hunt")
    h.add_argument("--mhz", type=float, default=404.000,
                   help="where to listen when this station has no history yet; with "
                        "history (lab/sonde_hits.json) it is only a side trip")
    h.add_argument("--secs", type=float, default=120)
    h.add_argument("--once", action="store_true")
    h.add_argument("--pin", action="store_true",
                   help="hold the commanded frequency - no history, no SondeHub, no "
                        "band scan (the 8/02 lesson: a LOCAL sonde-like interferer "
                        "out-shouts a real distant sonde and steals the aim)")
    h.add_argument("--duration", type=float, default=0,
                   help="stop after N minutes (0 = forever)")
    h.add_argument("--antenna", default=os.environ.get("SONDE_ANTENNA", "Antenna C"))
    h.add_argument("--rfgain", type=int, default=int(os.environ.get("SONDE_RFGAIN", "0")),
                   help="RSP LNA step (rfgain_sel); 0 is the most gain")
    h.add_argument("--ifgr", type=int, default=int(os.environ.get("SONDE_IFGR", "40")),
                   help="RSP IF gain reduction, dB")
    args = ap.parse_args()
    sys.exit(cmd_decode(args) if args.cmd == "decode" else cmd_hunt(args))


if __name__ == "__main__":
    main()
