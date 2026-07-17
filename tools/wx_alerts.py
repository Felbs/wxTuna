"""wx_alerts.py - wxTuna: decode SAME/EAS alert bursts from NOAA Weather Radio.

The screech before every weather alert is data: AFSK 520.83 baud
(mark 2083.3 Hz / space 1562.5 Hz), a header like

  ZCZC-WXR-TOR-011001+0030-1930015-KHB36/NWS-

carrying the event code (TOR = tornado warning), the exact FIPS county
codes, duration, and issue time. Decoding it gives machine-readable,
area-selectable weather alerts - the seed of the wxTuna meteorology
station (better than transcribing the voice: NOAA already structured it).

Modes:
  selftest - synthetic SAME burst -> demod -> parsed header roundtrip
  monitor  - listen on an NWR channel; print + log any burst heard
             (bursts only air during real alerts + weekly tests, so this
             is a leave-running tool, and per Law 2 the tone/baud
             constants are only field-proven when the first real alert
             lands)

Example:  python wx_alerts.py monitor --khz 162550
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from weather_sat import _ensure_sdr_dll_path   # noqa: E402

_ensure_sdr_dll_path()

LAB = HERE.parent / "lab"
ALOG = LAB / "wx_alerts.jsonl"

FS = 250_000.0
AUD = 41_667.0          # 80 samples per 520.833 bd bit exactly
BAUD = 520.8333
MARK, SPACE = 2083.3333, 1562.5

EVENTS = {"TOR": "TORNADO WARNING", "SVR": "SEVERE THUNDERSTORM WARNING",
          "FFW": "FLASH FLOOD WARNING", "SVA": "SEVERE T-STORM WATCH",
          "TOA": "TORNADO WATCH", "WIN": "WINTER STORM WARNING",
          "RWT": "REQUIRED WEEKLY TEST", "RMT": "REQUIRED MONTHLY TEST",
          "SPS": "SPECIAL WEATHER STATEMENT", "FLW": "FLOOD WARNING"}


def afsk_softbits(audio, fs, baud=BAUD, mark=MARK, space=SPACE):
    n = np.arange(len(audio))
    spb = fs / baud
    w = int(spb)
    box = np.ones(w, np.float32) / w
    em = np.abs(np.convolve(audio * np.exp(-2j * np.pi * mark / fs * n),
                            box, mode="same")).astype(np.float32)
    es = np.abs(np.convolve(audio * np.exp(-2j * np.pi * space / fs * n),
                            box, mode="same")).astype(np.float32)
    d = em - es
    nb = int(len(d) / spb) - 2
    soft = np.empty(nb, np.float32)
    pos = 0.0
    for k in range(nb):
        p = int(pos)
        if p + w >= len(d):
            soft = soft[:k]
            break
        soft[k] = float(np.mean(d[p:p + w]))
        pos += spb
    return soft


def bits_to_text(bits):
    """SAME bytes go LSB-first on the air; ASCII out."""
    out = []
    for k in range(0, len(bits) - 7, 8):
        b = sum(int(bits[k + i]) << i for i in range(8))
        out.append(b)
    return bytes(out)


def find_same(audio, fs):
    """Hunt the 0xAB preamble run, then decode the ZCZC header."""
    soft = afsk_softbits(audio, fs)
    hits = []
    for sgn in (1.0, -1.0):
        bits = (soft * sgn > 0).astype(np.int8)
        raw = bits_to_text(bits)
        # search all 8 bit-phases for the preamble+ZCZC
        for ph in range(8):
            by = bits_to_text(bits[ph:])
            i = by.find(b"ZCZC-")
            if i >= 0:
                end = by.find(b"-", i + 40)
                blob = by[i:i + 268]
                txt = blob.split(b"\xab")[0]
                try:
                    hits.append(txt.decode("ascii", errors="replace"))
                except Exception:
                    pass
                break
        if hits:
            break
    return hits


def parse_same(header):
    """ZCZC-ORG-EEE-PSSCCC(+more)+TTTT-JJJHHMM-SENDER-"""
    try:
        parts = header.strip("-").split("-")
        org, eee = parts[1], parts[2]
        area_blob = "-".join(parts[3:])
        plus = area_blob.index("+")
        fips = area_blob[:plus].split("-")
        rest = area_blob[plus + 1:].split("-")
        dur = rest[0]
        issued = rest[1] if len(rest) > 1 else "?"
        sender = rest[2] if len(rest) > 2 else "?"
        return {"org": org, "event": eee,
                "event_name": EVENTS.get(eee, eee),
                "fips": fips, "duration": f"{dur[:2]}h{dur[2:]}m",
                "issued_jjjhhmm": issued, "sender": sender}
    except Exception:
        return {"raw": header}


# ==========================================================================
def synth_same(header, fs=AUD, noise=0.05):
    msg = b"\xab" * 16 + header.encode("ascii")
    bits = []
    for byte in msg:
        for i in range(8):
            bits.append((byte >> i) & 1)
    spb = fs / BAUD
    total = int(len(bits) * spb) + 200
    freq = np.zeros(total, np.float32)
    for i, b in enumerate(bits):
        a, z = int(i * spb), int((i + 1) * spb)
        freq[a:z] = MARK if b else SPACE
    ph = np.cumsum(2 * np.pi * freq / fs)
    audio = np.sin(ph).astype(np.float32)
    rng = np.random.default_rng(4)
    return audio + rng.normal(0, noise, len(audio)).astype(np.float32)


def cmd_selftest(args):
    print("=" * 62)
    print("wxTuna SAME/EAS self-test")
    print("=" * 62)
    hdr = "ZCZC-WXR-TOR-011001-024031+0030-1930015-KHB36/NWS-"
    audio = synth_same(hdr)
    hits = find_same(audio, AUD)
    ok = bool(hits) and "TOR" in hits[0]
    print(f"  bursts found: {len(hits)}")
    if hits:
        p = parse_same(hits[0])
        print(f"  header: {hits[0][:60]}")
        print(f"  parsed: {p.get('event_name')} | counties {p.get('fips')} "
              f"| {p.get('duration')}")
        ok &= p.get("event") == "TOR" and "011001" in p.get("fips", [])
    print("=" * 62)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 62)
    return 0 if ok else 1


def cmd_monitor(args):
    """NBFM-demod an NWR channel and watch for SAME bursts, forever."""
    from scipy.signal import resample_poly
    from math import gcd
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, args.khz * 1e3)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, args.antenna)
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", 25)
        sdr.writeSetting("rfgain_sel", "0")
    except Exception:
        pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    print(f"[monitor] {args.khz} kHz - watching for SAME bursts (Ctrl+C stops)")
    buf = np.empty(2 * 65536, np.int16)
    g = gcd(int(AUD * 6), int(FS))
    try:
        while True:
            n_want = int(8 * FS)
            out = np.empty(2 * n_want, np.int16)
            got = 0
            while got < n_want:
                r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
                if r.ret > 0:
                    n = min(r.ret, n_want - got)
                    out[2 * got:2 * (got + n)] = buf[:2 * n]
                    got += n
            iq = ((out[0::2].astype(np.float32)
                   + 1j * out[1::2].astype(np.float32)) / 32768.0)
            disc = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
            audio = resample_poly(disc, int(AUD * 6) // g, int(FS) // g)
            audio = resample_poly(audio, 1, 6).astype(np.float32)
            hits = find_same(audio, AUD)
            for h in hits:
                p = parse_same(h)
                line = {"ts": datetime.now(timezone.utc).isoformat(), **p}
                print(f"[ALERT] {p.get('event_name', h)}  {p}")
                with open(ALOG, "a") as f:
                    f.write(json.dumps(line) + "\n")
    except KeyboardInterrupt:
        pass
    sdr.deactivateStream(st)
    sdr.closeStream(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    m = sub.add_parser("monitor")
    m.add_argument("--khz", type=float, default=162550)
    m.add_argument("--antenna", default="Antenna C")
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "monitor":
        cmd_monitor(args)


if __name__ == "__main__":
    main()
