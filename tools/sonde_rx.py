"""sonde_rx.py - wxTuna radiosonde receiver: capture/decode RS41
weather-balloon telemetry (the radiosonde_auto_rx borrow, our stack).

Pipeline: RSPdx (discone, Antenna C) or a saved cf32 -> FM
discriminator -> 48 kHz wav -> rs1729's rs41mod decoder (the same core
radiosonde_auto_rx trusts) -> parsed telemetry lines with position and
weather data.

  python sonde_rx.py decode --cf32 FILE --fs 100000     # offline
  python sonde_rx.py hunt [--mhz 404.000] [--secs 120]  # live capture
"""
import argparse
import json
import os
import subprocess
import sys
import time
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


def decode_wav(wav_path, verbose=True, iq=False):
    """Run rs41mod, return (frames, raw output lines)."""
    cmd = [str(RS41MOD), "-vv", "--ecc", "--crc"]
    if iq:
        cmd.append("--iq2")
    r = subprocess.run(cmd + [str(wav_path)],
                      capture_output=True, text=True, timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    frames = [ln for ln in out.splitlines() if "lat:" in ln or "h:" in ln]
    if verbose:
        for ln in frames[:12]:
            print("  " + ln.strip()[:110])
        if not frames:
            tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
            for ln in tail:
                print("  (no frames) " + ln.strip()[:90])
    return frames, out


def cmd_decode(args):
    iq = np.fromfile(args.cf32, dtype=np.complex64)
    print(f"[decode] {len(iq)/args.fs:.1f}s from {args.cf32}")
    wav = LAB / "sonde_iq.wav"
    iq_wav(iq, args.fs, wav)
    frames, _ = decode_wav(wav, iq=True)
    if not frames:                      # fallback: our own FM demod path
        wav2 = LAB / "sonde_demod.wav"
        fm_demod_wav(iq, args.fs, wav2)
        frames, _ = decode_wav(wav2)
    print(f"[decode] {len(frames)} telemetry lines")
    return 0 if frames else 1


def cmd_hunt(args):
    """Live: capture on the discone, decode, log, repeat."""
    sys.path.insert(0, r"Z:\src\gr-radiotuna\tools")
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    for attempt in range(5):
        try:
            sdr = SoapySDR.Device("driver=sdrplay")
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3)
    FS = 1e6
    sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, args.mhz * 1e6 + 100e3)  # DC offside
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, "Antenna C")           # discone
    except Exception:
        pass
    sdr.setGainMode(SOAPY_SDR_RX, 0, False)
    sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", 40)
    sdr.writeSetting("rfgain_sel", "4")
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(st)
    print(f"[hunt] {args.mhz:.3f} MHz on the discone, "
          f"{args.secs:.0f}s per pass", flush=True)
    buf = np.zeros(262144, np.complex64)
    logf = LAB / f"sonde_log_{time.strftime('%Y%m%d')}.txt"
    try:
        while True:
            ch = []
            t0 = time.time()
            while time.time() - t0 < args.secs:
                r = sdr.readStream(st, [buf], len(buf), timeoutUs=800000)
                if r.ret > 0:
                    ch.append(buf[:r.ret].copy())
            x = np.concatenate(ch)
            n = np.arange(len(x), dtype=np.float64)
            x = (x * np.exp(-2j * np.pi * (-100e3) / FS * n)).astype(np.complex64)
            y = resample_poly(x, 1, 10).astype(np.complex64)   # 100 kS/s
            wav = LAB / "sonde_live.wav"
            iq_wav(y, 100_000, wav)
            frames, out = decode_wav(wav, verbose=False, iq=True)
            stamp = time.strftime("%H:%M:%S")
            if frames:
                print(f"[{stamp}] {len(frames)} frames:", flush=True)
                for ln in frames[-3:]:
                    print("   " + ln.strip()[:110], flush=True)
                with open(logf, "a", encoding="utf-8") as f:
                    for ln in frames:
                        f.write(f"{stamp} {ln.strip()}\n")
            else:
                print(f"[{stamp}] no frames (sonde weak/absent)", flush=True)
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
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--cf32", required=True)
    d.add_argument("--fs", type=float, default=100_000)
    h = sub.add_parser("hunt")
    h.add_argument("--mhz", type=float, default=404.000)
    h.add_argument("--secs", type=float, default=120)
    h.add_argument("--once", action="store_true")
    args = ap.parse_args()
    sys.exit(cmd_decode(args) if args.cmd == "decode" else cmd_hunt(args))


if __name__ == "__main__":
    main()
