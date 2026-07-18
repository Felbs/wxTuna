"""lrpt.py - Meteor-M2 LRPT decoder for Radio Tuna.  The TV Tuna method on 137.9.

Meteor-M2-3/M2-4 transmit LRPT: QPSK @ 72k sym/s, then the CCSDS FEC stack -
r=1/2 K=7 convolutional (Viterbi) -> CCSDS derandomizer -> RS(255,223) -> CADU
frames (ASM 0x1ACFFC1D) -> VCDU -> image packets. That FEC back-half is the
SAME chain we forged on ATSC, which is the whole reason weather sats fit our
method: our SOFT-decision Viterbi + erasure/GMD Reed-Solomon should out-decode
stock demods on weak / low-elevation passes.

Pipeline (this file), each stage independently testable:
  IQ .cs16 -> [demod] soft QPSK symbols (+ live MER dial)
           -> [softbits] QPSK -> LLRs
           -> [viterbi]  CCSDS r=1/2 K=7 soft Viterbi         <- our edge
           -> [derandomize] CCSDS PN
           -> [frame sync] ASM 0x1ACFFC1D
           -> [RS + image]  (stage 2 - calibrates against a real pass)

Commands:
  python lrpt.py selftest              # prove the DSP+FEC engines (synthetic)
  python lrpt.py decode <capture.cs16> # run a real capture: lock? MER? frames?

Honesty: the FEC + demod ENGINES are validated here by synthetic round-trips.
End-to-end image reconstruction (VCDU->DCT->PNG) and the exact CCSDS bit
conventions (code phase, differential/IQ swap, RS dual-basis) get calibrated
against the first real locked pass - you cannot debug those blind. Run this on
tonight's 22:11 M2-3 capture and we close that loop.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                       # numba absent (e.g. Pi) -> pure-python
    _HAVE_NUMBA = False

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"

SYM_RATE = 72_000.0
ASM = 0x1ACFFC1D            # CCSDS attached sync marker


# ==========================================================================
# CCSDS r=1/2, K=7 convolutional code  (G1=0o171, G2=0o133)
# ==========================================================================
G1 = 0o171
G2 = 0o133
K = 7
NST = 1 << (K - 1)          # 64 states


def _popcount(x):
    return bin(x).count("1")


# precompute transition tables: for (state, bit) -> next_state, (o1,o2)
_NEXT = np.zeros((NST, 2), np.int32)
_OUT = np.zeros((NST, 2, 2), np.int8)     # expected output bits as +1/-1
for s in range(NST):
    for b in (0, 1):
        reg = (b << (K - 1)) | s          # 7-bit register, newest at top
        o1 = _popcount(reg & G1) & 1
        o2 = _popcount(reg & G2) & 1
        _NEXT[s, b] = reg >> 1            # drop oldest bit
        _OUT[s, b, 0] = 1 if o1 == 0 else -1
        _OUT[s, b, 1] = 1 if o2 == 0 else -1


def conv_encode(bits):
    """CCSDS r=1/2 encoder. bits (0/1) -> code bits (0/1), length 2N."""
    out = np.empty(2 * len(bits), np.int8)
    s = 0
    for i, b in enumerate(bits):
        reg = (int(b) << (K - 1)) | s
        out[2 * i] = _popcount(reg & G1) & 1
        out[2 * i + 1] = _popcount(reg & G2) & 1
        s = reg >> 1
    return out


def viterbi_decode(soft):
    """Soft-input Viterbi. `soft` = code symbols mapped 0->+1, 1->-1 (with
    noise), length 2N. Returns hard bits (0/1), length N.

    This is the engine our ATSC work sharpens (SOVA reliability, then the
    erasure/GMD RS ladder downstream). Correctness proven by selftest."""
    N = len(soft) // 2
    r = soft.reshape(N, 2).astype(np.float32)
    NEG = -1e9
    pm = np.full(NST, NEG, np.float32)
    pm[0] = 0.0
    tb = np.zeros((N, NST), np.int8)      # traceback: which bit led here
    prev = np.zeros((N, NST), np.int32)
    # transition sources: for each next_state, the two (state,bit) that reach it
    for n in range(N):
        r1, r2 = r[n, 0], r[n, 1]
        new_pm = np.full(NST, NEG, np.float32)
        new_tb = np.zeros(NST, np.int8)
        new_prev = np.zeros(NST, np.int32)
        for s in range(NST):
            if pm[s] <= NEG / 2:
                continue
            for b in (0, 1):
                ns = _NEXT[s, b]
                bm = r1 * _OUT[s, b, 0] + r2 * _OUT[s, b, 1]
                cand = pm[s] + bm
                if cand > new_pm[ns]:
                    new_pm[ns] = cand
                    new_tb[ns] = b
                    new_prev[ns] = s
        pm = new_pm
        tb[n] = new_tb
        prev[n] = new_prev
    # traceback from best state
    s = int(np.argmax(pm))
    bits = np.empty(N, np.int8)
    for n in range(N - 1, -1, -1):
        bits[n] = tb[n, s]
        s = prev[n, s]
    return bits


# --- fast streaming Viterbi (numba-jitted, bounded memory) ----------------
# Same code + metric as viterbi_decode(), but with a depth-D sliding traceback
# so memory is O(D*64) not O(N*64) -> handles a whole 65M-symbol pass. The
# pure-python version above stays as the reference the selftest checks against.
_NEXT32 = _NEXT.astype(np.int32)
_OUTA = np.ascontiguousarray(_OUT[:, :, 0].astype(np.float32))
_OUTB = np.ascontiguousarray(_OUT[:, :, 1].astype(np.float32))


def _viterbi_stream_impl(r, NEXT, OUTA, OUTB, D):
    N = r.shape[0] // 2
    NST = NEXT.shape[0]
    NEG = np.float32(-1e30)
    pm = np.full(NST, NEG, np.float32)
    pm[0] = np.float32(0.0)
    newpm = np.empty(NST, np.float32)
    tb = np.zeros((D, NST), np.uint8)        # winning input bit
    tbp = np.zeros((D, NST), np.int32)       # winning predecessor state
    out = np.empty(N, np.uint8)
    nout = 0
    for n in range(N):
        r1 = r[2 * n]
        r2 = r[2 * n + 1]
        for ns in range(NST):
            newpm[ns] = NEG
        slot = n % D
        for s in range(NST):
            pms = pm[s]
            if pms <= NEG / 2:
                continue
            for b in range(2):
                ns = NEXT[s, b]
                cand = pms + r1 * OUTA[s, b] + r2 * OUTB[s, b]
                if cand > newpm[ns]:
                    newpm[ns] = cand
                    tb[slot, ns] = b
                    tbp[slot, ns] = s
        for ns in range(NST):
            pm[ns] = newpm[ns]
        if n >= D - 1:
            best = 0
            bv = pm[0]
            for s in range(1, NST):
                if pm[s] > bv:
                    bv = pm[s]
                    best = s
            st = best
            for d in range(D - 1):
                st = tbp[(n - d) % D, st]
            out[nout] = tb[(n - (D - 1)) % D, st]
            nout += 1
    # flush the final D-1 bits from the best terminal state
    best = 0
    bv = pm[0]
    for s in range(1, NST):
        if pm[s] > bv:
            bv = pm[s]
            best = s
    st = best
    m = D - 1 if D - 1 < N else N
    tmp = np.empty(m, np.uint8)
    for d in range(m):
        nn = N - 1 - d
        tmp[d] = tb[nn % D, st]
        st = tbp[nn % D, st]
    for d in range(m):
        out[nout] = tmp[m - 1 - d]
        nout += 1
    return out[:nout]


if _HAVE_NUMBA:
    _viterbi_stream = njit(cache=True)(_viterbi_stream_impl)
else:
    _viterbi_stream = _viterbi_stream_impl


def viterbi_decode_fast(soft, D=48):
    """Full-pass Viterbi. Identical code/metric to viterbi_decode() but
    jitted with a depth-D sliding traceback (D >= ~6K is effectively optimal)."""
    soft = np.ascontiguousarray(soft, np.float32)
    return _viterbi_stream(soft, _NEXT32, _OUTA, _OUTB, np.int64(D))


# ==========================================================================
# CCSDS derandomizer (PN: x^8 + x^7 + x^5 + x^3 + 1, seed 0xFF)
# ==========================================================================
def _ccsds_pn(nbytes):
    reg = 0xFF
    out = np.empty(nbytes, np.uint8)
    for i in range(nbytes):
        byte = 0
        for _ in range(8):
            bit = (reg >> 7) & 1
            byte = (byte << 1) | bit
            fb = ((reg >> 7) ^ (reg >> 6) ^ (reg >> 4) ^ (reg >> 2)) & 1
            reg = ((reg << 1) | fb) & 0xFF
        out[i] = byte
    return out


def derandomize(data):
    pn = _ccsds_pn(len(data))
    return np.bitwise_xor(data.astype(np.uint8), pn)


# ==========================================================================
# QPSK demod:  CS16 IQ -> soft symbols  (RRC + timing + Costas) + MER dial
# ==========================================================================
def read_iq(path):
    path = Path(path)
    raw = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    iq = raw[0::2] + 1j * raw[1::2]
    fs = 250_000.0
    side = Path(str(path) + ".json")
    if side.exists():
        import json
        try:
            fs = float(json.loads(side.read_text()).get("fs_hz", fs))
        except Exception:
            pass
    return iq.astype(np.complex64), fs


def capture_fs(path):
    """Sample rate from the sidecar json (250k default) - no IQ read needed."""
    import json
    side = Path(str(path) + ".json")
    if side.exists():
        try:
            return float(json.loads(side.read_text()).get("fs_hz", 250_000.0))
        except Exception:
            pass
    return 250_000.0


def read_iq_slice(path, offset_s, secs, fs):
    """Read ONLY the requested window from disk (memory = window, not file)."""
    path = Path(path)
    total = path.stat().st_size // 4          # complex samples on disk
    start = max(0, min(int(offset_s * fs), total))
    n = max(0, min(int(secs * fs), total - start))
    raw = np.fromfile(path, dtype=np.int16, count=2 * n, offset=start * 4)
    iq = (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0
    return iq.astype(np.complex64)


def rrc_taps(beta, sps, span=8):
    N = span * sps
    t = (np.arange(-N, N + 1)) / sps
    h = np.zeros_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-8:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * x) - 1) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - beta)) + 4 * beta * x * np.cos(np.pi * x * (1 + beta))) \
                / (np.pi * x * (1 - (4 * beta * x) ** 2))
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def demod(iq, fs, beta=0.6):
    """IQ -> recovered QPSK symbols + a lock/MER report. Gardner timing at
    2 sps, then a QPSK Costas loop. Returns (symbols, info)."""
    from scipy.signal import resample_poly
    # DC block
    iq = iq - np.mean(iq)
    # resample to exactly 2 samples/symbol
    target_fs = 2 * SYM_RATE
    from math import gcd
    up = int(target_fs)
    down = int(fs)
    g = gcd(up, down)
    x = resample_poly(iq, up // g, down // g).astype(np.complex64)
    sps = 2
    # RRC matched filter
    h = rrc_taps(beta, sps)
    x = np.convolve(x, h, mode="same").astype(np.complex64)
    x /= (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-9)

    # --- Gardner timing recovery (2 sps -> 1 sps), linear interpolation ---
    mu = 0.0
    i = sps
    out = []
    prev_sample = 0j
    mid_prev = 0j
    gain = 0.02
    N = len(x)
    while i < N - 2:
        base = int(i)
        frac = i - base
        s = x[base] * (1 - frac) + x[base + 1] * frac                # symbol
        m_idx = i - sps / 2
        mb = int(m_idx); mf = m_idx - mb
        mid = x[mb] * (1 - mf) + x[mb + 1] * mf                      # halfway
        e = (np.real(mid) * (np.real(prev_sample) - np.real(s))
             + np.imag(mid) * (np.imag(prev_sample) - np.imag(s)))
        mu = gain * e
        out.append(s)
        prev_sample = s
        i += sps + mu
    syms = np.array(out, np.complex64)

    # --- Costas loop (QPSK, decision-directed) ---
    phase = 0.0
    freq = 0.0
    a = 0.01
    b = a * a / 4
    rec = np.empty(len(syms), np.complex64)
    for n, s in enumerate(syms):
        v = s * np.exp(-1j * phase)
        rec[n] = v
        # QPSK phase error: distance to nearest (±1±1j)/√2
        err = np.sign(np.real(v)) * np.imag(v) - np.sign(np.imag(v)) * np.real(v)
        freq += b * err
        phase += freq + a * err
    # MER dial: how tightly symbols cluster on the QPSK constellation
    ideal = (np.sign(np.real(rec)) + 1j * np.sign(np.imag(rec))) / np.sqrt(2)
    rec_n = rec / (np.sqrt(np.mean(np.abs(rec) ** 2)) + 1e-9)
    err_pow = np.mean(np.abs(rec_n - ideal) ** 2) + 1e-9
    mer_db = 10 * np.log10(1.0 / err_pow)
    info = {"n_symbols": len(rec), "mer_db": round(float(mer_db), 2),
            "locked": bool(mer_db > 6.0)}
    return rec, info


# --- fast demod (numba-jitted Gardner + Costas, preallocated) --------------
# Same math as demod() above, which stays as the validation oracle. The pure-
# python loops box millions of numpy scalars (slow + fragments the allocator -
# observed 31 GB RSS on one 25 s window); these kernels allocate two arrays.
def _gardner_impl(x, sps, gain):
    N = x.shape[0]
    cap = int(N / (sps - 0.2)) + 16
    out = np.empty(cap, np.complex128)
    n = 0
    i = float(sps)
    prev_re = 0.0
    prev_im = 0.0
    half = sps / 2.0
    while i < N - 2 and n < cap:
        base = int(i)
        frac = i - base
        sre = x[base].real * (1 - frac) + x[base + 1].real * frac
        sim = x[base].imag * (1 - frac) + x[base + 1].imag * frac
        m = i - half
        mb = int(m)
        mf = m - mb
        mre = x[mb].real * (1 - mf) + x[mb + 1].real * mf
        mim = x[mb].imag * (1 - mf) + x[mb + 1].imag * mf
        e = mre * (prev_re - sre) + mim * (prev_im - sim)
        out[n] = complex(sre, sim)
        n += 1
        prev_re = sre
        prev_im = sim
        i += sps + gain * e
    return out[:n]


def _costas_impl(syms, a, b):
    n = syms.shape[0]
    rec = np.empty(n, np.complex128)
    phase = 0.0
    freq = 0.0
    for k in range(n):
        c = math.cos(phase)
        s = math.sin(phase)
        xr = syms[k].real
        xi = syms[k].imag
        vre = xr * c + xi * s
        vim = xi * c - xr * s
        rec[k] = complex(vre, vim)
        sgn_r = 1.0 if vre > 0.0 else (-1.0 if vre < 0.0 else 0.0)
        sgn_i = 1.0 if vim > 0.0 else (-1.0 if vim < 0.0 else 0.0)
        err = sgn_r * vim - sgn_i * vre
        freq += b * err
        phase += freq + a * err
    return rec


if _HAVE_NUMBA:
    _gardner = njit(cache=True)(_gardner_impl)
    _costas = njit(cache=True)(_costas_impl)
else:
    _gardner = _gardner_impl
    _costas = _costas_impl


def demod_fast(iq, fs, beta=0.6):
    """Same pipeline as demod() with the two feedback loops jitted.
    Identical math; validated against the reference in selftest."""
    from scipy.signal import resample_poly
    from math import gcd
    iq = iq - np.mean(iq)
    up = int(2 * SYM_RATE)
    down = int(fs)
    g = gcd(up, down)
    x = resample_poly(iq, up // g, down // g).astype(np.complex64)
    sps = 2
    h = rrc_taps(beta, sps)
    x = np.convolve(x, h, mode="same").astype(np.complex64)
    x /= (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-9)
    syms = _gardner(x.astype(np.complex128), float(sps), 0.02)
    rec = _costas(syms, 0.01, 0.01 ** 2 / 4).astype(np.complex64)
    ideal = (np.sign(np.real(rec)) + 1j * np.sign(np.imag(rec))) / np.sqrt(2)
    rec_n = rec / (np.sqrt(np.mean(np.abs(rec) ** 2)) + 1e-9)
    err_pow = np.mean(np.abs(rec_n - ideal) ** 2) + 1e-9
    mer_db = 10 * np.log10(1.0 / err_pow)
    info = {"n_symbols": len(rec), "mer_db": round(float(mer_db), 2),
            "locked": bool(mer_db > 6.0)}
    return rec, info


def qpsk_softbits(syms):
    """QPSK Gray -> soft bits (I then Q per symbol), scaled to ~±1."""
    s = syms / (np.mean(np.abs(syms)) + 1e-9)
    bits = np.empty(2 * len(s), np.float32)
    bits[0::2] = np.real(s)
    bits[1::2] = np.imag(s)
    return bits


# ==========================================================================
# frame sync
# ==========================================================================
def find_asm(bits):
    """Scan a hard bitstream for the 32-bit ASM; return byte-offsets of hits."""
    packed = np.packbits(bits[: (len(bits) // 8) * 8])
    asm = np.array([0x1A, 0xCF, 0xFC, 0x1D], np.uint8)
    hits = []
    for i in range(len(packed) - 4):
        if np.array_equal(packed[i:i + 4], asm):
            hits.append(i)
    return hits


# ==========================================================================
# self-tests  (prove the engines on synthetic data)
# ==========================================================================
def selftest_viterbi():
    print("[selftest] CCSDS Viterbi r=1/2 K=7 - encode -> AWGN -> soft decode")
    rng = np.random.default_rng(1)
    nbits = 4000
    ok_all = True
    for ebn0 in (6.0, 4.0, 2.0, 0.0):
        bits = rng.integers(0, 2, nbits).astype(np.int8)
        code = conv_encode(bits)                      # 0/1
        tx = np.where(code == 0, 1.0, -1.0)           # BPSK map
        # noise scaled for rate-1/2 (2 code bits per info bit)
        sigma = 1 / np.sqrt(2 * 10 ** (ebn0 / 10) * 0.5)
        rx = tx + rng.normal(0, sigma, len(tx)).astype(np.float32)
        dec = viterbi_decode(rx.astype(np.float32))
        # account for K-1 tail: compare the reliable middle
        errs = int(np.sum(dec[:nbits - K] != bits[:nbits - K]))
        ber = errs / (nbits - K)
        flag = "OK" if (ebn0 >= 4 and ber == 0) or ebn0 < 4 else "FAIL"
        if ebn0 >= 4 and ber > 0:
            ok_all = False
        print(f"   Eb/N0={ebn0:4.1f} dB   BER={ber:.4f}   ({errs} errs)  {flag}")
    print("   => Viterbi engine", "VALIDATED\n" if ok_all else "PROBLEM\n")
    return ok_all


def selftest_viterbi_fast():
    import time
    print("[selftest] fast Viterbi - matches reference + full-pass throughput"
          + ("" if _HAVE_NUMBA else "  (numba MISSING: pure-python fallback)"))
    rng = np.random.default_rng(7)
    nbits = 4000
    bits = rng.integers(0, 2, nbits).astype(np.int8)
    tx = np.where(conv_encode(bits) == 0, 1.0, -1.0)
    sigma = 1 / np.sqrt(2 * 10 ** (4.0 / 10) * 0.5)
    rx = (tx + rng.normal(0, sigma, len(tx))).astype(np.float32)
    ref = viterbi_decode(rx)
    fast = viterbi_decode_fast(rx)                     # first call = JIT compile
    n = min(len(ref), len(fast))
    agree = int(np.sum(ref[:n - K] == fast[:n - K]))
    match = agree / (n - K)
    ber = int(np.sum(fast[:nbits - K] != bits[:nbits - K])) / (nbits - K)
    print(f"   fast vs reference agreement: {match*100:.2f}%   fast BER: {ber:.4f}")
    # throughput on a large block (warm cache already compiled above)
    big = rng.integers(0, 2, 400_000).astype(np.int8)
    btx = np.where(conv_encode(big) == 0, 1.0, -1.0).astype(np.float32)
    t = time.time()
    viterbi_decode_fast(btx)
    dt = time.time() - t
    rate = len(big) / dt
    print(f"   throughput: {len(big)} bits in {dt:.2f}s = {rate/1e6:.2f}M bits/s")
    print(f"   => a 15-min pass (~65M bits) decodes in ~{65e6/rate/60:.1f} min "
          f"(was ~191 min)\n")
    return match > 0.999 and ber == 0.0


def selftest_pn():
    print("[selftest] CCSDS derandomizer - XOR is its own inverse")
    rng = np.random.default_rng(2)
    data = rng.integers(0, 256, 1024).astype(np.uint8)
    back = derandomize(derandomize(data))
    ok = np.array_equal(data, back)
    print("   round-trip", "OK" if ok else "FAIL",
          "| first PN bytes:", list(_ccsds_pn(4)), "\n")
    return ok


def selftest_demod():
    print("[selftest] QPSK demod - synthetic 72k in 250k IQ, freq+timing offset")
    rng = np.random.default_rng(3)
    nsym = 6000
    syms = (rng.integers(0, 2, nsym) * 2 - 1) + 1j * (rng.integers(0, 2, nsym) * 2 - 1)
    syms = syms.astype(np.complex64) / np.sqrt(2)
    sps0 = 250_000 / SYM_RATE
    # upsample to 250k with an RRC pulse
    from scipy.signal import resample_poly
    base = np.zeros(int(nsym * sps0) + 10, np.complex64)
    idx = (np.arange(nsym) * sps0).astype(int)
    base[idx] = syms
    h = rrc_taps(0.6, 8)
    tx = np.convolve(base, np.interp(np.linspace(0, len(h) - 1, int(len(h) * sps0 / 8)),
                                     np.arange(len(h)), h), mode="same").astype(np.complex64)
    # add small carrier offset + noise
    t = np.arange(len(tx))
    tx = tx * np.exp(1j * 2 * np.pi * 800 / 250_000 * t)
    tx += (rng.normal(0, 0.05, len(tx)) + 1j * rng.normal(0, 0.05, len(tx))).astype(np.complex64)
    rec, info = demod(tx, 250_000.0)
    print(f"   recovered {info['n_symbols']} symbols | MER {info['mer_db']} dB | "
          f"locked={info['locked']}")
    ok = info["locked"]
    print("   => demod", "LOCKED\n" if ok else "did not lock (tune loop gains)\n")
    return ok


def _synthetic_qpsk(nsym=6000, seed=3):
    """The selftest_demod signal generator, shared with the fast-demod test."""
    rng = np.random.default_rng(seed)
    syms = (rng.integers(0, 2, nsym) * 2 - 1) + 1j * (rng.integers(0, 2, nsym) * 2 - 1)
    syms = syms.astype(np.complex64) / np.sqrt(2)
    sps0 = 250_000 / SYM_RATE
    base = np.zeros(int(nsym * sps0) + 10, np.complex64)
    idx = (np.arange(nsym) * sps0).astype(int)
    base[idx] = syms
    h = rrc_taps(0.6, 8)
    tx = np.convolve(base, np.interp(np.linspace(0, len(h) - 1, int(len(h) * sps0 / 8)),
                                     np.arange(len(h)), h), mode="same").astype(np.complex64)
    t = np.arange(len(tx))
    tx = tx * np.exp(1j * 2 * np.pi * 800 / 250_000 * t)
    tx += (rng.normal(0, 0.05, len(tx)) + 1j * rng.normal(0, 0.05, len(tx))).astype(np.complex64)
    return tx


def selftest_demod_fast():
    import time
    print("[selftest] fast demod - matches reference + speed"
          + ("" if _HAVE_NUMBA else "  (numba MISSING: python fallback)"))
    tx = _synthetic_qpsk()
    t0 = time.time(); rec_r, info_r = demod(tx, 250_000.0); t_ref = time.time() - t0
    t0 = time.time(); rec_f, info_f = demod_fast(tx, 250_000.0); t_warm = time.time() - t0
    t0 = time.time(); rec_f, info_f = demod_fast(tx, 250_000.0); t_fast = time.time() - t0
    dmer = abs(info_r["mer_db"] - info_f["mer_db"])
    same_lock = info_r["locked"] == info_f["locked"]
    print(f"   reference: MER {info_r['mer_db']} dB locked={info_r['locked']} in {t_ref:.2f}s")
    print(f"   fast     : MER {info_f['mer_db']} dB locked={info_f['locked']} in {t_fast:.3f}s"
          f"  (JIT compile pass took {t_warm:.1f}s)")
    print(f"   agreement: dMER={dmer:.2f} dB, lock match={same_lock}, "
          f"speedup {t_ref/max(t_fast,1e-9):.0f}x")
    ok = same_lock and dmer < 0.5
    print("   => fast demod", "VALIDATED\n" if ok else "MISMATCH\n")
    return ok


def cmd_selftest(args):
    print("=" * 62)
    print("LRPT engine self-test - validating the pieces we can prove now")
    print("=" * 62)
    a = selftest_viterbi()
    af = selftest_viterbi_fast()
    b = selftest_pn()
    c = selftest_demod()
    cf = selftest_demod_fast()
    c = c and cf
    print("=" * 62)
    print(f"Viterbi {'PASS' if a and af else 'FAIL'} | derandomizer {'PASS' if b else 'FAIL'} "
          f"| demod {'PASS' if c else 'FAIL'}")
    print("Next: run  lrpt.py decode <tonight's 22:11 capture>  to lock a real")
    print("pass, then calibrate RS + image reconstruction against it.")
    print("=" * 62)


def cmd_decode(args):
    path = Path(args.capture)
    if not path.exists():
        sys.exit(f"no such capture: {path}")
    fs = capture_fs(path)
    dur = (path.stat().st_size // 4) / fs
    print(f"[decode] {path.name}: {dur:.1f}s @ {fs/1e3:.0f} kHz")
    off = args.offset
    if args.mid:
        off = max(0.0, dur / 2 - args.secs / 2)   # seek to the pass peak
    print(f"[decode] window: {off:.0f}s -> {off+args.secs:.0f}s (slice-read only)")
    chunk = read_iq_slice(path, off, args.secs, fs)
    if args.shift_khz:
        # wideband capture: mix the sub-band at +shift kHz down to DC first
        # (resample_poly's anti-alias low-pass then rejects everything else)
        t = np.arange(len(chunk), dtype=np.float64)
        chunk = (chunk * np.exp(-2j * np.pi * args.shift_khz * 1e3 / fs * t)
                 ).astype(np.complex64)
        print(f"[decode] extracted sub-band at {args.shift_khz:+.0f} kHz from center")
    rec, info = demod_fast(chunk, fs)
    pol = "normal"
    if not info["locked"]:
        # IQ inversion is the #1 first-lock gotcha (spectrum mirrored by the
        # SDR); try the conjugate and keep whichever clusters tighter.
        rec2, info2 = demod_fast(np.conj(chunk), fs)
        if info2["mer_db"] > info["mer_db"]:
            rec, info, pol = rec2, info2, "IQ-conjugated"
    print(f"[decode] MER dial: {info['mer_db']} dB  (lock threshold ~6 dB)  "
          f"locked={info['locked']}  polarity={pol}")
    if not info["locked"]:
        print("[decode] no QPSK lock on either polarity - likely no Meteor signal")
        print("         (if a pass WAS overhead, suspect the antenna port). Noise.")
        return
    print("[decode] LOCKED. Running Viterbi over a segment to search for ASMs ...")
    sb = qpsk_softbits(rec[: args.vitsyms])
    bits = viterbi_decode_fast(sb)
    hits = find_asm(bits)
    print(f"[decode] Viterbi produced {len(bits)} bits; ASM 0x1ACFFC1D hits: {len(hits)}")
    if hits:
        print(f"         first ASM byte-offsets: {hits[:6]}")
        print("         -> real CADU frames! next stage: RS(255,223) + image.")
    else:
        print("         no ASMs yet - try IQ-swap/conjugate or code phase (calibration).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("decode")
    d.add_argument("capture")
    d.add_argument("--secs", type=float, default=20, help="seconds of IQ to demod")
    d.add_argument("--offset", type=float, default=0.0, help="seconds into the capture to start")
    d.add_argument("--mid", action="store_true", help="seek to the pass peak (middle)")
    d.add_argument("--shift-khz", type=float, default=0.0,
                   help="sub-band offset from capture center in kHz (wideband captures)")
    d.add_argument("--vitsyms", type=int, default=40000, help="symbols through Viterbi")
    args = ap.parse_args()
    if args.cmd == "selftest":
        cmd_selftest(args)
    elif args.cmd == "decode":
        cmd_decode(args)


if __name__ == "__main__":
    main()
