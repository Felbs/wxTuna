"""rs41_decode.py - wxTuna: full RS41 frame decoder (first light 2026-07-18,
serial X5130033 over Clifton VA, 17/20 CRC-verified frames from a frozen
capture). The chain, each stage earned by a failed simpler version:

  IQ -> FSK-pair centering -> FM discriminator -> moving-median drift
  removal -> 10 sps matched filter -> ON-AIR sync correlation (whitened
  bits, not frame bytes) -> chunk-wise eye tracking (clock slip killed
  the GPS fields while early fields decoded - re-find the timing phase
  every 518 bits) -> LSB-first packing -> XOR de-whitening -> zero-pad
  320->518 (parity covers the padded frame) -> RS(255,231) with
  confidence-guided erasures -> field parse with per-block CRC16.

Usage:
  python rs41_decode.py <capture.cs16> [...]     # decode, print track rows
  python rs41_decode.py selftest                 # RS codec proves itself
"""
import sys
from math import gcd, sqrt, atan2, degrees
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MASK = bytes([0x96,0x83,0x3E,0x51,0xB1,0x49,0x08,0x98,0x32,0x05,0x59,0x0E,0xF9,0x44,0xC6,0x26,
0x21,0x60,0xC2,0xEA,0x79,0x5D,0x6D,0xA1,0x54,0x69,0x47,0x0C,0xDC,0xE8,0x5C,0xF1,
0xF7,0x76,0x82,0x7F,0x07,0x99,0xA2,0x2C,0x93,0x7C,0x30,0x63,0xF5,0x10,0x2E,0x61,
0xD0,0xBC,0xB4,0xB6,0x06,0xAA,0xF4,0x23,0x78,0x6E,0x3B,0xAE,0xBF,0x7B,0x4C,0xC1])
FRAME_LEN = 518
NDATA = 320
SYNC_ONAIR = ("00001000011011010101001110001000"
              "01000100011010010100100000011111")

# ---------------- GF(256)/RS(255,231), poly 0x11D, b=0 ----------------
EXP = [0]*512
LOG = [0]*256
_v = 1
for _i in range(255):
    EXP[_i] = _v; LOG[_v] = _i
    _v <<= 1
    if _v & 0x100:
        _v ^= 0x11D
for _i in range(255, 512):
    EXP[_i] = EXP[_i-255]


def _gmul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a]+LOG[b]]


def _ginv(a):
    return EXP[255-LOG[a]]


def _pmul(a, b):
    r = [0]*(len(a)+len(b)-1)
    for i, ca in enumerate(a):
        if ca:
            for j, cb in enumerate(b):
                r[i+j] ^= _gmul(ca, cb)
    return r


def _peval(p, x):
    y = 0
    for c in reversed(p):
        y = _gmul(y, x) ^ c
    return y


def rs_encode(data231):
    g = [1]
    for i in range(24):
        g = _pmul(g, [EXP[i], 1])
    rem = [0]*24 + list(data231)
    for i in range(254, 23, -1):
        c = rem[i]
        if c:
            for j, gc in enumerate(g):
                rem[i-24+j] ^= _gmul(c, gc)
    return rem[:24] + list(data231)


def rs_decode(cw, era=()):
    """Errors-and-erasures RS(255,231). Returns (n_corrected or -1, cw).
    CAUTION (law): a 'successful' decode near the erasure limit can be a
    miscorrection - always confirm with an external CRC."""
    S = [_peval(cw, EXP[j]) for j in range(24)]
    if max(S) == 0:
        return 0, cw
    Gam = [1]
    for p in era:
        Gam = _pmul(Gam, [1, EXP[p]])
    T = _pmul(S, Gam)[:24]
    e = len(era)
    L = 0; Lam = [1]; B = [1]
    for nn in range(e, 24):
        d = T[nn]
        for j in range(1, L+1):
            if j < len(Lam) and nn-j >= 0:
                d ^= _gmul(Lam[j], T[nn-j])
        B = [0]+B
        if d:
            Tp = list(Lam)
            while len(Lam) < len(B):
                Lam.append(0)
            for j in range(len(B)):
                Lam[j] ^= _gmul(d, B[j])
            if 2*L <= nn-e:
                L = nn-e+1-L
                B = [_gmul(_ginv(d), c) for c in Tp]
    Psi = _pmul(Lam, Gam)
    pos = [i for i in range(255) if _peval(Psi, EXP[(255-i) % 255]) == 0]
    if not pos:
        return -1, cw
    Om = _pmul(S, Psi)[:24]
    dPsi = Psi[1::2]
    fixed = list(cw)
    for i in pos:
        Xinv = EXP[(255-i) % 255]
        den = _peval(dPsi, _gmul(Xinv, Xinv))
        if den == 0:
            return -1, cw
        fixed[i] ^= _gmul(EXP[i % 255], _gmul(_peval(Om, Xinv), _ginv(den)))
    if max(_peval(fixed, EXP[j]) for j in range(24)) == 0:
        return len(pos), fixed
    return -1, cw


def _crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
    return crc


def _blk_ok(fr, pos):
    ln = fr[pos+1]
    if pos+2+ln+2 > len(fr):
        return False
    return _crc16(fr[pos+2:pos+2+ln]) == (fr[pos+2+ln] | (fr[pos+3+ln] << 8))


def _ecef_to_lla(x, y, z):
    a = 6378137.0; e2 = 6.69437999014e-3
    lon = atan2(y, x); p = sqrt(x*x+y*y)
    lat = atan2(z, p*(1-e2)); alt = 0.0
    for _ in range(6):
        N = a/sqrt(1-e2*np.sin(lat)**2)
        alt = p/np.cos(lat) - N
        lat = atan2(z, p*(1-e2*N/(N+alt)))
    return degrees(lat), degrees(lon), alt


def decode_capture(path_or_iq, fs=250_000.0, verbose=False):
    """Returns list of frame dicts: frnr, serial, batt_v, lat, lon, alt_m,
    v_kmh, gps_ok, ecc. Only trust rows with gps_ok=True."""
    from sonde import find_fsk_pair, find_carrier
    from scipy.signal import resample_poly
    from scipy.ndimage import median_filter
    if isinstance(path_or_iq, (str, Path)):
        raw = np.fromfile(str(path_or_iq), np.int16).astype(np.float32)/32768.0
        iq = (raw[0::2] + 1j*raw[1::2]).astype(np.complex64)
    else:
        iq = path_or_iq
    off, _ = find_fsk_pair(iq, fs)
    if off is None:
        off, _ = find_carrier(iq, fs)
    n = np.arange(len(iq), dtype=np.float64)
    iq = (iq*np.exp(-2j*np.pi*off/fs*n)).astype(np.complex64)
    AUD = 48000
    g = gcd(AUD, int(fs))
    x = resample_poly(iq, AUD//g, int(fs)//g).astype(np.complex64)
    disc = np.angle(x[1:]*np.conj(x[:-1])).astype(np.float32)
    disc -= median_filter(disc, size=2401, mode="nearest")
    mf = np.convolve(disc, np.ones(10, np.float32)/10, mode="same")
    SPS = AUD/4800.1
    pat = (np.array([int(c) for c in SYNC_ONAIR], np.float32)*2-1)
    nb = int((len(mf)-12)/SPS)
    cidx = (np.arange(nb)*SPS).astype(int)
    cc = np.correlate(np.sign(mf[cidx]).astype(np.float32), pat, mode="valid")
    locs = []
    for L in np.where(cc >= 58)[0]:
        if not locs or L-locs[-1] > 3000:
            locs.append(int(L))
    out = []
    for L in locs:
        s0 = L*SPS
        best = (-1.0, 0.0)
        for ph in np.arange(-6, 6.5, 0.5):
            idx = (s0+ph+np.arange(64)*SPS).astype(int)
            idx = idx[idx < len(mf)]
            if len(idx) < 64:
                continue
            sc = float(np.dot(np.sign(mf[idx]), pat))
            if sc > best[0]:
                best = (sc, ph)
        CH = 518
        soft = np.empty(FRAME_LEN*8, np.float32)
        ph_acc = best[1]
        for k in range(8):
            base = s0+ph_acc+(k*CH)*SPS
            bc = (-1.0, 0.0)
            for dph in np.arange(-2.0, 2.25, 0.25):
                idx = (base+dph+np.arange(CH)*SPS).astype(int)
                idx = np.minimum(idx, len(mf)-1)
                m = float(np.mean(np.abs(mf[idx])))
                if m > bc[0]:
                    bc = (m, dph)
            ph_acc += bc[1]
            idx = (base+bc[1]+np.arange(CH)*SPS).astype(int)
            idx = np.minimum(idx, len(mf)-1)
            soft[k*CH:(k+1)*CH] = mf[idx]
        bits = (soft > 0).astype(np.uint8)
        bb = np.packbits(bits.reshape(-1, 8), axis=1, bitorder="little").ravel()
        frame = bytearray(b ^ MASK[i % 64] for i, b in enumerate(bb))
        bconf = np.abs(soft).reshape(-1, 8).min(axis=1)
        for i in range(NDATA, FRAME_LEN):
            frame[i] = 0
            bconf[i] = 1e9
        eccs = []
        for k in (0, 1):
            cw = [0]*255; cpos = [0]*255
            for i in range(24):
                cw[i] = frame[8+i+24*k]; cpos[i] = 8+i+24*k
            for i in range(231):
                fp = 56+2*i+k
                cw[24+i] = frame[fp]; cpos[24+i] = fp
            ne, fx = rs_decode(cw)
            if ne < 0:
                order = np.argsort([bconf[cpos[i]] for i in range(255)])
                for nera in (8, 16, 22):
                    ne, fx = rs_decode(cw, era=[int(o) for o in order[:nera]])
                    if ne >= 0:
                        break
            if ne >= 0:
                for i in range(255):
                    frame[cpos[i]] = fx[i]
            eccs.append(ne)
        gps = _blk_ok(frame, 0x112)
        ex = int.from_bytes(frame[0x114:0x118], "little", signed=True)/100.0
        ey = int.from_bytes(frame[0x118:0x11C], "little", signed=True)/100.0
        ez = int.from_bytes(frame[0x11C:0x120], "little", signed=True)/100.0
        la, lo, al = _ecef_to_lla(ex, ey, ez)
        vx = int.from_bytes(frame[0x120:0x122], "little", signed=True)/100.0
        vy = int.from_bytes(frame[0x122:0x124], "little", signed=True)/100.0
        vz = int.from_bytes(frame[0x124:0x126], "little", signed=True)/100.0
        out.append({
            "frnr": int.from_bytes(frame[0x3B:0x3D], "little"),
            "serial": "".join(chr(b) if 32 <= b < 127 else "?"
                              for b in frame[0x3D:0x45]),
            "batt_v": frame[0x45]/10.0,
            "lat": round(la, 5), "lon": round(lo, 5), "alt_m": round(al, 1),
            "v_kmh": round(sqrt(vx*vx+vy*vy+vz*vz)*3.6, 1),
            "vv_ms": round(vz and (vx*ex+vy*ey+vz*ez) /
                           max(sqrt(ex*ex+ey*ey+ez*ez), 1e-9), 2),
            "gps_ok": bool(gps), "ecc": tuple(eccs)})
    return out


def cmd_selftest():
    rng = np.random.default_rng(1)
    d = [int(v) for v in rng.integers(0, 256, 231)]
    cw = rs_encode(d)
    assert max(_peval(cw, EXP[j]) for j in range(24)) == 0
    c2 = list(cw)
    for p in (3, 50, 99, 140, 200, 254, 10, 77, 160, 220, 30, 111):
        c2[p] ^= int(rng.integers(1, 256))
    ne, fx = rs_decode(c2)
    assert ne == 12 and fx == cw, "12-error fix failed"
    c3 = list(cw)
    for p in range(20):
        c3[p+30] ^= int(rng.integers(1, 256))
    _, fx3 = rs_decode(c3, era=list(range(30, 50)))
    assert fx3 == cw, "erasure fix failed"
    print("SELFTEST PASS (RS 12-error + 20-erasure)")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if sys.argv[1] == "selftest":
        return cmd_selftest()
    for f in sys.argv[1:]:
        rows = decode_capture(f)
        ok = [r for r in rows if r["gps_ok"]]
        print(f"{Path(f).name}: {len(rows)} frames, {len(ok)} CRC-verified")
        for r in ok:
            print(f"  #{r['frnr']} {r['serial']} {r['lat']:.4f} {r['lon']:.4f} "
                  f"{r['alt_m']:.0f}m {r['v_kmh']:.0f}km/h batt {r['batt_v']}V")
    return 0


if __name__ == "__main__":
    sys.exit(main())
