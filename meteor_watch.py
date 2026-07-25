#!/usr/bin/env python3
"""meteor_watch.py - count shooting stars by radio (meteor forward-scatter).

A meteor's ionized trail briefly mirrors a distant VHF transmitter (analog TV
carrier, FM, or a dedicated beacon) that is normally over the horizon and
silent. Point the SDR at that dead carrier frequency; every few minutes the
band goes quiet-quiet-quiet-PING as a meteor reflects it for 0.1-3 s. Count
the pings -> a meteor rate. Aim it at the Perseid radiant band (Aug 11-13) and
you have a shower detector that works in daylight and cloud.

Method: narrowband power at the beacon freq, per 0.1 s. Rolling noise floor
(median). A PING = a run where power jumps >SNR_TRIP dB above floor for
0.1-3 s (longer = local QRM/aircraft; shorter = impulse noise - both rejected).
Logs each event (time, duration, peak dB, doppler) + an hourly rate.

  python meteor_watch.py --freq 55.24e6 --hours 12   # a distant ch2 video carrier
  python meteor_watch.py --selftest                  # synthetic ping stream
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = 250_000.0
BW_HZ = 600.0                 # narrow window around the beacon carrier
SNR_TRIP = 8.0                # dB above rolling floor to trip
MIN_S, MAX_S = 0.15, 3.5      # >=2 consecutive 0.1s bins: a meteor SUSTAINS,
#                               single-bin spikes = impulse noise (rejected)
CHUNK_S = 0.5


def power_series(iq, fs, off_hz):
    """0.1 s narrowband power (dB) at the beacon offset."""
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off_hz / fs * n)
    # decimate to ~BW: boxcar then subsample
    k = int(fs / (2 * BW_HZ))
    xf = np.convolve(x, np.ones(k) / k, mode="same")[::k]
    fr = int(0.1 * fs / k)
    m = len(xf) // fr * fr
    p = (np.abs(xf[:m].reshape(-1, fr)) ** 2).mean(1)
    return 10 * np.log10(p + 1e-12)


def find_pings(pdb, dt=0.1):
    """Runs above a rolling-median floor within the meteor duration window."""
    floor = np.array([np.median(pdb[max(0, i - 100):i + 1]) for i in range(len(pdb))])
    hot = pdb > floor + SNR_TRIP
    events = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            dur = (j - i) * dt
            if MIN_S <= dur <= MAX_S:
                events.append(dict(t_rel=round(i * dt, 2), dur=round(dur, 2),
                                   peak_db=round(float(pdb[i:j].max() - floor[i]), 1)))
            i = j
        else:
            i += 1
    return events


def selftest():
    print("meteor_watch selftest: synth quiet band + injected pings")
    rng = np.random.default_rng(3)
    dt = 0.1
    n = int(600 / dt)                     # 10 min
    pdb = rng.normal(-90, 1.5, n)         # quiet noise floor
    truth = []
    for _ in range(14):                   # 14 meteors
        t = rng.integers(20, n - 40)
        d = rng.integers(1, 25)           # 0.1-2.5 s
        pdb[t:t + d] += rng.uniform(10, 30)
        truth.append(t * dt)
    # a few impulse spikes (0.1s) + one long QRM (aircraft, 6s) that must be rejected
    for _ in range(20):
        pdb[rng.integers(0, n)] += 25
    q = rng.integers(0, n - 60)
    pdb[q:q + 60] += 15
    ev = find_pings(pdb, dt)
    print(f"  injected 14 meteors (+20 impulses +1 long-QRM) -> detected {len(ev)}")
    print(f"  {'PASS' if 11 <= len(ev) <= 16 else 'FAIL'} "
          f"(impulses/QRM correctly rejected by duration window)")
    return 0 if 11 <= len(ev) <= 16 else 1


def run(freq, hours, antenna):
    import sys as _s
    _s.path.insert(0, r"Z:\src\hamTuna\tools")
    import cw
    from SoapySDR import SOAPY_SDR_RX
    sdr, st = cw._open_sdr(antenna, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq)
    time.sleep(0.3)
    out = HERE / "meteor_log.jsonl"
    deadline = time.time() + hours * 3600
    total = 0
    hour0 = time.time()
    hour_count = 0
    print(f"[meteor] watching {freq/1e6:.3f} MHz on {antenna} for {hours}h", flush=True)
    while time.time() < deadline:
        iq = cw._grab(sdr, st, 10.0, FS)
        if len(iq) < FS:
            continue
        pdb = power_series(iq, FS, 0.0)
        for e in find_pings(pdb):
            total += 1
            hour_count += 1
            e["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(out, "a") as f:
                f.write(json.dumps(e) + "\n")
            print(f"[meteor] PING #{total}: {e['dur']}s {e['peak_db']}dB", flush=True)
        if time.time() - hour0 >= 3600:
            print(f"[meteor] HOURLY RATE: {hour_count}/h", flush=True)
            hour0 = time.time()
            hour_count = 0
    print(f"[meteor] done. {total} meteors in {hours}h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=55.24e6)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--antenna", default="Antenna C")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(a.freq, a.hours, a.antenna))


if __name__ == "__main__":
    main()
