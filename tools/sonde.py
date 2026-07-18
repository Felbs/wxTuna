"""sonde.py - wxTuna campaign 2: radiosondes (weather balloons), RS41 @ 400-406 MHz.

Twice a day, every day, your National Weather Service office releases a
balloon carrying an RS41 radiosonde: GFSK 4800 bd telemetry protected by
Reed-Solomon RS(255,231) + CRC - our FEC ladder's home turf. The signal
starts strong overhead and fades as the balloon drifts 100+ km downrange:
a scheduled, guaranteed, marginal-by-the-end transmitter. Everything the
Meteor birds refused to be.

Predicting "passes" (three layers):
  1. SCHEDULE - launches are fixed: ~11:00Z and ~23:00Z daily (released
     ~1 h before the 00Z/12Z synoptic hours). `schedule` computes them.
  2. LIVE RADAR - SondeHub (the global amateur receiver network) has a
     public API listing every sonde in the air RIGHT NOW: position,
     altitude, and the exact transmit frequency. `radar` queries it.
  3. RF SCAN - `scan` sweeps 400-406 MHz with the SDR and finds the
     carrier empirically (the ground truth).

Modes:
  selftest  - GFSK demod + sync-correlator roundtrip on synthetic IQ
  schedule  - next launch windows (local time), no radio, no net
  radar     - live sondes near you via SondeHub (net, no radio)
  scan      - sweep 400-406 MHz for a sonde carrier (radio)
  capture   - record IQ at a frequency for N seconds (radio)
  hunt      - radar -> pick nearest sonde's frequency -> capture + detect

Observer defaults to a metro-coarse location; override WXSAT_LAT/LON.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from weather_sat import _ensure_sdr_dll_path, OBS_LAT, OBS_LON   # noqa: E402

_ensure_sdr_dll_path()

LAB = HERE.parent / "lab"
SONDE_DIR = LAB / "sonde"
SONDE_DIR.mkdir(parents=True, exist_ok=True)

BAUD = 4800.0
# RS41 whitened frame header (the 8-byte sync every receiver correlates on;
# confirmed against live frames on first catch - flagged until then)
RS41_SYNC = bytes([0x86, 0x35, 0xF4, 0x40, 0x93, 0xDF, 0x1A, 0x60])
# ^ those are the DE-WHITENED header bytes (what lands in the decoded
# frame). What actually FLIES is the whitened bit pattern below - the
# sync correlator must hunt the on-air bits, not the frame bytes.
# (Verified 7/18 against rs41mod.c:170 after our correlator scored
# chance on a live capture with a +16 dB clock line - ledger law #1:
# selftests prove machinery, only live signals prove constants.)
RS41_SYNC_ONAIR = ("00001000011011010101001110001000"
                   "01000100011010010100100000011111")
SONDEHUB = ("https://api.v2.sondehub.org/sondes"
            "?lat={lat}&lon={lon}&distance={m}&last=10800")   # distance in METERS


# ==========================================================================
# launch schedule
# ==========================================================================
def next_launches(n=4, release_min_before=60):
    """NWS upper-air releases: ~1 h before 00Z and 12Z, daily, forever."""
    now = datetime.now(timezone.utc)
    out = []
    day = now.date()
    while len(out) < n:
        for syn_h in (0, 12):
            syn = datetime(day.year, day.month, day.day, syn_h,
                           tzinfo=timezone.utc)
            rel = syn - timedelta(minutes=release_min_before)
            if syn_h == 0:
                rel += timedelta(days=1)     # 00Z sonde launches ~23:00Z prior day... same date math
                rel -= timedelta(days=1)
                syn_show = syn if rel > now else None
            if rel > now and len(out) < n:
                out.append({"release_utc": rel, "synoptic": f"{syn_h:02d}Z",
                            "window_end": rel + timedelta(hours=2, minutes=30)})
        day += timedelta(days=1)
    return out


def cmd_schedule(args):
    print(f"observer {OBS_LAT:.2f},{OBS_LON:.2f} | local = UTC{args.utc_off:+d}")
    print("NWS radiosonde releases (~60 min before 00Z/12Z synoptic):\n")
    for L in next_launches(args.count):
        loc = L["release_utc"] + timedelta(hours=args.utc_off)
        end = L["window_end"] + timedelta(hours=args.utc_off)
        dt_min = (L["release_utc"] - datetime.now(timezone.utc)).total_seconds() / 60
        print(f"  {loc:%a %H:%M} local  ({L['synoptic']} sounding)  "
              f"flight window -> ~{end:%H:%M}  [in {dt_min/60:.1f} h]")
    print("\nballoon rises ~2 h to ~30 km (radio horizon 600+ km!), then")
    print("bursts; strongest overhead early, marginal at the far end - the")
    print("adaptive-decode playground. Use `radar` for live positions.")


# ==========================================================================
# SondeHub live radar
# ==========================================================================
def fetch_radar(km=350):
    url = SONDEHUB.format(lat=OBS_LAT, lon=OBS_LON, m=int(km * 1000))
    req = urllib.request.Request(url, headers={"User-Agent": "wxTuna-sonde"})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    sondes = []
    now = datetime.now(timezone.utc)
    for serial, t in (data or {}).items():
        try:
            # skip stale entries (landed/lost birds linger in the API for
            # hours - we once hunted a corpse on the ground 87 km away)
            dt = t.get("datetime", "")
            if dt:
                age = (now - datetime.fromisoformat(
                    dt.replace("Z", "+00:00"))).total_seconds()
                if age > 1200:
                    continue
            lat, lon = float(t["lat"]), float(t["lon"])
            d = haversine_km(OBS_LAT, OBS_LON, lat, lon)
            sondes.append({
                "serial": serial, "lat": lat, "lon": lon, "km": round(d, 1),
                "alt_m": int(float(t.get("alt", 0))),
                "mhz": float(t.get("frequency", 0)) or None,
                "type": t.get("type", "?"),
                "time": t.get("datetime", "?"),
            })
        except Exception:
            continue
    sondes.sort(key=lambda s: s["km"])
    return sondes


def haversine_km(la1, lo1, la2, lo2):
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def cmd_radar(args):
    print(f"[radar] querying SondeHub for sondes within {args.km} km ...")
    try:
        sondes = fetch_radar(args.km)
    except Exception as e:
        print(f"[radar] API error: {e}")
        return
    if not sondes:
        print("[radar] no sondes aloft in range right now - check `schedule`")
        return
    print(f"[radar] {len(sondes)} sonde(s) in the air:\n")
    for s in sondes:
        f = f"{s['mhz']:.3f} MHz" if s["mhz"] else "freq ?"
        print(f"  {s['serial']:<12} {s['type']:<6} {s['km']:>6.1f} km away  "
              f"alt {s['alt_m']:>6} m  {f}  ({s['time'][11:19]}Z)")
    return sondes


# ==========================================================================
# GFSK demod + sync correlate (numba)
# ==========================================================================
def _bits_from_bytes(bb):
    out = np.zeros(len(bb) * 8, np.int8)
    for i, byte in enumerate(bb):
        for k in range(8):
            out[8 * i + k] = (byte >> (7 - k)) & 1
    return out


_SYNC_PM = (np.array([int(c) for c in RS41_SYNC_ONAIR], np.int8)
            * 2 - 1).astype(np.float32)


def _bitsync_impl(disc, sps):
    """FM-discriminated signal -> soft bits at BAUD via boxcar integrate-
    and-dump with a simple early/late clock nudge."""
    N = disc.shape[0]
    nb = int(N / sps) - 2
    soft = np.empty(nb, np.float32)
    pos = 0.0
    isps = int(sps)
    for k in range(nb):
        p = int(pos)
        if p + isps >= N:
            nb = k
            break
        acc = 0.0
        for j in range(isps):
            acc += disc[p + j]
        soft[k] = acc
        # early/late: compare halves, nudge sampling phase
        h1 = 0.0
        h2 = 0.0
        half = isps // 2
        for j in range(half):
            h1 += disc[p + j]
            h2 += disc[p + half + j]
        if soft[k] > 0:
            pos += sps + (0.05 if h2 > h1 else -0.05)
        else:
            pos += sps + (0.05 if h1 > h2 else -0.05)
    return soft[:nb]


def _syncscan_impl(soft, sync_pm):
    """Correlate the 64-chip sync against soft bits; return best hits."""
    n = soft.shape[0]
    m = sync_pm.shape[0]
    best = np.empty(64, np.int64)
    bestv = np.empty(64, np.float32)
    nb = 0
    thresh = 0.75 * m
    k = 0
    while k < n - m:
        c = 0.0
        for j in range(m):
            c += (1.0 if soft[k + j] > 0 else -1.0) * sync_pm[j]
        if c >= thresh:
            if nb < 64:
                best[nb] = k
                bestv[nb] = c
                nb += 1
            k += m
        else:
            k += 1
    return best[:nb], bestv[:nb]


def _gardner_impl(d, sps, kp, ki):
    """H3: proper Gardner timing recovery with a PI loop filter, replacing
    the fixed-stride integrate-and-dump. Tracks clock offset AND drift -
    the crude sync smeared 2560-bit frames beyond recognition at low SNR."""
    N = d.shape[0]
    cap = int(N / (sps * 0.97)) + 16
    out = np.empty(cap, np.float32)
    nb = 0
    pos = sps
    freq = sps
    prev = 0.0
    fmin = sps * 0.98
    fmax = sps * 1.02
    while pos < N - 2 and nb < cap:
        i = int(pos)
        fr = pos - i
        s = d[i] * (1 - fr) + d[i + 1] * fr
        m = pos - freq * 0.5
        j = int(m)
        mf = m - j
        mid = d[j] * (1 - mf) + d[j + 1] * mf if j >= 0 else 0.0
        e = mid * ((1.0 if s > 0 else -1.0) - (1.0 if prev > 0 else -1.0))
        freq += ki * e
        if freq < fmin:
            freq = fmin
        elif freq > fmax:
            freq = fmax
        pos += freq + kp * e
        out[nb] = s
        nb += 1
        prev = s
    return out[:nb]


if _HAVE_NUMBA:
    _bitsync = njit(cache=True)(_bitsync_impl)
    _syncscan = njit(cache=True)(_syncscan_impl)
    _gardner = njit(cache=True)(_gardner_impl)
else:
    _bitsync = _bitsync_impl
    _syncscan = _syncscan_impl
    _gardner = _gardner_impl


def find_carrier(iq, fs, min_snr_db=6.0):
    """Locate the strongest narrowband carrier in the window: (offset_hz,
    snr_db). The sonde is rarely exactly at the tuned center."""
    N = 1 << 16
    if len(iq) < N:
        N = 1 << int(np.log2(max(1024, len(iq))))
    nseg = min(20, len(iq) // N)
    seg = iq[:nseg * N].reshape(nseg, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(P + 1e-12)
    med = float(np.median(db))
    c = N // 2
    db[c - 3:c + 4] = med                # ignore DC spike
    pk = int(np.argmax(db))
    snr = float(db[pk] - med)
    off = (pk - c) * fs / N
    return (off, snr) if snr >= min_snr_db else (0.0, snr)


def find_fsk_pair(iq, fs, search_hz=30_000, spacing=4800.0, tol=700.0):
    """Find the RS41's signature: TWO spectral lobes ~4.8 kHz apart within
    +-search_hz of center. Returns (center_offset_hz, pair_snr_db) or
    (None, best_single_snr). Beats picking the strongest peak, which is
    often a neighboring interferer."""
    N = 8192
    nseg = max(2, min(400, len(iq) // N))
    seg = iq[:nseg * N].reshape(nseg, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(P + 1e-12)
    med = float(np.median(db))
    c = N // 2
    db[c - 2:c + 3] = med
    binw = fs / N
    k = int(search_hz / binw)
    win = db[c - k:c + k] - med
    # peaks above 5 dB
    idx = [i for i in range(1, len(win) - 1)
           if win[i] > 5.0 and win[i] >= win[i - 1] and win[i] >= win[i + 1]]
    best = None
    for a in idx:
        for b in idx:
            if b <= a:
                continue
            df = (b - a) * binw
            if abs(df - spacing) < tol:
                score = win[a] + win[b]
                if best is None or score > best[2]:
                    best = (a, b, score)
    if best is None:
        return None, (max(win) if len(win) else 0.0)
    center = ((best[0] + best[1]) / 2 - k) * binw
    return float(center), float(best[2] / 2)


def detect_rs41(iq, fs):
    """IQ -> carrier find + mix to DC -> FM discriminator -> bit sync ->
    RS41 sync correlation (both FSK polarities). The live truth dial."""
    from scipy.signal import resample_poly
    from math import gcd
    off, snr = find_fsk_pair(iq, fs)
    if off is None:                       # no lobe pair: old strongest-peak path
        off, snr = find_carrier(iq, fs)
    if off != 0.0:
        n = np.arange(len(iq), dtype=np.float64)
        iq = (iq * np.exp(-2j * np.pi * off / fs * n)).astype(np.complex64)
    target = 19_200                      # 4 samples/bit; +-9.6 kHz AA filter
    g = gcd(int(target), int(fs))        # keeps ~6 dB of noise out vs 48k
    x = resample_poly(iq, int(target) // g, int(fs) // g).astype(np.complex64)
    d = x[1:] * np.conj(x[:-1])
    disc = np.angle(d).astype(np.float32)
    disc -= np.float32(np.mean(disc))    # center the two FSK tones
    sync_used = "gardner"
    soft = _gardner(disc, target / BAUD, 0.03, 0.0008)
    hits, scores, pol = _scan_both(soft)
    if len(hits) == 0:                   # A/B fallback: the old crude sync
        soft2 = _bitsync(disc, target / BAUD)
        h2, s2, p2 = _scan_both(soft2)
        if len(h2) > 0:
            hits, scores, pol, sync_used = h2, s2, p2, "integrate-dump"
            soft = soft2
    return len(hits), {"hits": hits.tolist(), "scores": scores.tolist(),
                       "n_bits": len(soft), "carrier_off_hz": round(off),
                       "carrier_snr_db": round(snr, 1), "polarity": pol,
                       "sync": sync_used}


def _scan_both(soft):
    hits, scores = _syncscan(soft, _SYNC_PM)
    if len(hits) > 0:
        return hits, scores, "normal"
    hits2, scores2 = _syncscan(-soft, _SYNC_PM)
    if len(hits2) > 0:
        return hits2, scores2, "inverted"
    return hits, scores, "normal"


# ==========================================================================
# SDR
# ==========================================================================
def open_sdr(freq_hz, antenna, gain_db=40, fs=250_000):
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
    except Exception:
        pass
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
    except Exception:
        pass
    try:
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", max(20, 59 - gain_db))
        sdr.writeSetting("rfgain_sel", "0")   # max LNA - 20 mW from 70 km away
    except Exception:
        pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    return sdr, st


def grab(sdr, st, secs, fs=250_000):
    n_want = int(secs * fs)
    buf = np.empty(2 * 65536, np.int16)
    out = np.empty(2 * n_want, np.int16)
    got = 0
    while got < n_want:
        r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
        if r.ret > 0:
            n = min(r.ret, n_want - got)
            out[2 * got: 2 * (got + n)] = buf[:2 * n]
            got += n
        elif r.ret < 0 and r.ret != -1:
            break
    iq = (out[0::2].astype(np.float32) + 1j * out[1::2].astype(np.float32)) / 32768.0
    return iq[:got].astype(np.complex64)


def close_sdr(sdr, st):
    try:
        sdr.deactivateStream(st)
        sdr.closeStream(st)
    except Exception:
        pass


def cmd_scan(args):
    print(f"[scan] sweeping 400.0-406.0 MHz on {args.antenna} ...")
    hits = []
    sdr, st = open_sdr(400.1e6, args.antenna, args.gain)
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX
    for cf in np.arange(400.1e6, 406.0e6, 0.2e6):
        sdr.setFrequency(SOAPY_SDR_RX, 0, float(cf))
        time.sleep(0.08)
        iq = grab(sdr, st, 0.25)
        spec = np.abs(np.fft.fftshift(np.fft.fft(iq[:65536] * np.hanning(min(65536, len(iq))))))
        db = 20 * np.log10(spec + 1e-9)
        med = np.median(db)
        pk = float(db.max() - med)
        pk_off = (int(np.argmax(db)) - len(db) // 2) * 250e3 / len(db)
        if pk > args.thresh:
            f_mhz = (cf + pk_off) / 1e6
            hits.append((f_mhz, pk))
            print(f"  {f_mhz:.3f} MHz  peak +{pk:.0f} dB  <-- candidate")
    close_sdr(sdr, st)
    if not hits:
        print(f"[scan] nothing above +{args.thresh} dB - no sonde airborne nearby?")
    return hits


def cmd_capture(args):
    print(f"[capture] {args.secs:.0f}s @ {args.mhz:.3f} MHz on {args.antenna}")
    sdr, st = open_sdr(args.mhz * 1e6, args.antenna, args.gain)
    iq = grab(sdr, st, args.secs)
    close_sdr(sdr, st)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = SONDE_DIR / f"rs41_{args.mhz:.3f}_{stamp}.cs16".replace(".", "p", 1)
    (np.round(np.column_stack([iq.real, iq.imag]).ravel() * 32767)
     .astype(np.int16)).tofile(out)
    json.dump({"freq_hz": args.mhz * 1e6, "fs_hz": 250_000.0,
               "format": "cs16", "n_samples": len(iq)},
              open(str(out) + ".json", "w"))
    n, det = detect_rs41(iq, 250_000.0)
    print(f"[capture] saved {out.name} ({len(iq)/250e3:.1f}s)")
    print(f"[detect] RS41 sync hits: {n}"
          + (f"  scores {['%.0f' % s for s in det['scores'][:5]]}" if n else ""))
    if n:
        try:                # full decode: the 7/18 first-light pipeline
            from rs41_decode import decode_capture
            rows = [r for r in decode_capture(iq) if r["gps_ok"]]
            if rows:
                r = rows[-1]
                print(f"[DECODE] {len(rows)} CRC-verified frames - "
                      f"{r['serial']} @ {r['lat']:.4f},{r['lon']:.4f} "
                      f"{r['alt_m']:.0f} m  {r['v_kmh']:.0f} km/h  "
                      f"batt {r['batt_v']:.1f} V")
                with open(SONDE_DIR.parent / "sonde_track.jsonl", "a") as tf:
                    for r in rows:
                        r["capture"] = out.name
                        tf.write(json.dumps(r) + "\n")
            else:
                print("[DECODE] sync found but no CRC-verified frames "
                      "(weak signal - IQ saved for replay)")
        except Exception as e:
            print(f"[DECODE] decode error: {e}")
    return n


def cmd_hunt(args):
    sondes = cmd_radar(args)
    if not sondes:
        return
    tgt = next((s for s in sondes if s["mhz"]), None)
    if not tgt:
        print("[hunt] radar shows sondes but no frequency - run `scan`")
        return
    print(f"\n[hunt] target {tgt['serial']} at {tgt['km']} km, "
          f"{tgt['mhz']:.3f} MHz - capturing {args.secs:.0f}s ...")
    args.mhz = tgt["mhz"]
    cmd_capture(args)


# ==========================================================================
# selftest
# ==========================================================================
def cmd_selftest(args):
    print("=" * 62)
    print("wxTuna sonde self-test (GFSK demod + RS41 sync correlator)")
    print("=" * 62)
    rng = np.random.default_rng(5)
    fs = 250_000.0
    sps_tx = fs / BAUD
    # frame: preamble + sync + random payload, GFSK at +/-2.4 kHz deviation
    payload = rng.integers(0, 256, 300, dtype=np.uint8).tobytes()
    # transmit what actually flies: preamble bits + ON-AIR (whitened)
    # sync pattern + payload bits
    bits = np.concatenate([
        _bits_from_bytes(b"\xAA" * 8),
        np.array([int(c) for c in RS41_SYNC_ONAIR], np.uint8),
        _bits_from_bytes(payload)])
    # build the FSK phase ramp
    dev = 2400.0
    t_total = int(len(bits) * sps_tx) + 1000
    freq = np.zeros(t_total, np.float32)
    for i, b in enumerate(bits):
        a = int(i * sps_tx)
        z = int((i + 1) * sps_tx)
        freq[a:z] = dev if b else -dev
    phase = np.cumsum(2 * np.pi * freq / fs)
    iq = np.exp(1j * phase).astype(np.complex64) * 0.5
    # carrier offset + noise
    n = np.arange(len(iq))
    iq = iq * np.exp(1j * 2 * np.pi * 900 / fs * n)
    iq += (rng.normal(0, 0.12, len(iq)) + 1j * rng.normal(0, 0.12, len(iq))
           ).astype(np.complex64)
    nhits, det = detect_rs41(iq, fs)
    ok = nhits >= 1
    print(f"  synthetic RS41 burst (noise + 900 Hz offset): sync hits={nhits}"
          f"  {'OK' if ok else 'FAIL'}")
    if ok:
        print(f"  correlation scores: {['%.0f' % s for s in det['scores'][:3]]} / 64 max")
    print("=" * 62)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 62)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utc-off", type=int, default=-4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sc = sub.add_parser("schedule")
    sc.add_argument("--count", type=int, default=4)
    r = sub.add_parser("radar")
    r.add_argument("--km", type=float, default=350)
    s = sub.add_parser("scan")
    s.add_argument("--antenna", default="Antenna C")
    s.add_argument("--gain", type=float, default=40)
    s.add_argument("--thresh", type=float, default=15)
    c = sub.add_parser("capture")
    c.add_argument("--mhz", type=float, required=True)
    c.add_argument("--secs", type=float, default=20)
    c.add_argument("--antenna", default="Antenna C")
    c.add_argument("--gain", type=float, default=40)
    h = sub.add_parser("hunt")
    h.add_argument("--km", type=float, default=350)
    h.add_argument("--secs", type=float, default=20)
    h.add_argument("--antenna", default="Antenna C")
    h.add_argument("--gain", type=float, default=40)
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "schedule":
        cmd_schedule(args)
    elif args.cmd == "radar":
        cmd_radar(args)
    elif args.cmd == "scan":
        cmd_scan(args)
    elif args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "hunt":
        cmd_hunt(args)


if __name__ == "__main__":
    main()
