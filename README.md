<img src="assets/logo.svg" align="right" width="120" alt="wxTuna logo — birds in a satellite dish">

# wxTuna 🐟📡

**Adaptive weather-satellite decoding — the TV Tuna method, aimed at the sky.**

Born 2026-07-17 from [Radio Tuna](https://github.com/Felbs/gr-radiotuna) and
[Software-TV-Tuner](https://github.com/Felbs/Software-TV-Tuner) (TV Tuna),
where the method was forged against ATSC television on marginal antennas.
wxTuna points the same adaptive, self-calibrating decoder at the weather
birds passing overhead.

## Why weather satellites fit the method
NOAA's analog APT is gone (NOAA-15/19 were decommissioned in August 2025).
What's left on 137 MHz is **digital**: the **Meteor-M2** birds transmit
**LRPT** — QPSK at 72k symbols/sec wrapped in the CCSDS forward-error-
correction stack:

```
QPSK 72k  ->  Viterbi r=1/2 K=7  ->  CCSDS derandomizer  ->  RS(255,223)  ->  CADU frames  ->  image
```

That FEC back-half is the **same chain we sharpened on ATSC** — soft-decision
Viterbi feeding an erasure/GMD Reed-Solomon ladder. Consumer decoders treat a
weak pass as take-it-or-leave-it; wxTuna surfaces the decoder's own confidence
(a live MER dial straight off the demodulator) and hill-climbs against it, the
same closed loop that runs TV Tuna. The bet: **we recover images from low,
noisy passes that stock decoders drop.**

Geostationary **GOES HRIT** (1.7 GHz) is on the roadmap as the always-on
development bench.

## Tools
| Tool | What it does |
|---|---|
| `tools/weather_sat.py` | Unattended pass **recorder** — predicts every Meteor pass (sgp4), wakes at AOS, records baseband IQ at 137.9 MHz for the whole pass, logs it. Modes: `passes` / `record` / `watch`. |
| `tools/weather_sat_panel.py` | **Glass-cockpit web panel** (localhost:8644) — live state, pass schedule, signal meter, and click-to-render spectrum/waterfall of every capture. Read-only; safe to run alongside the recorder. |
| `tools/lrpt.py` | The **decoder** — QPSK demod (RRC → Gardner timing → Costas, + MER dial) → CCSDS soft Viterbi → derandomizer → CADU frame sync. `selftest` proves the engines; `decode <capture>` runs a real pass. |

## Quickstart
```bash
# see when the birds fly over (no radio needed)
python tools/weather_sat.py passes --hours 24

# leave it running all day; it records every pass automatically
python tools/weather_sat.py watch --min-elev 15

# open the panel to watch it work
python tools/weather_sat_panel.py          # -> http://localhost:8644

# prove the decoder engines, then decode a captured pass
python tools/lrpt.py selftest
python tools/lrpt.py decode lab/wxsat/lrpt_*.cs16
```
(`lab/` is empty on a fresh clone — the watch loop records passes into it;
`lrpt.py decode` needs at least one recorded capture first.)
Set your location with `WXSAT_LAT` / `WXSAT_LON` env vars (or `--lat/--lon`);
the default is metro-coarse. Runs under **radioconda** (needs `SoapySDR`,
`numpy`, `scipy`, `sgp4`, `numba`, `matplotlib`, `Pillow`).

## Status (early)
- ✅ Recorder + web panel: working, capturing real IQ off an SDRplay RSPdx.
- ✅ Decoder **engines validated on synthetic**: CCSDS Viterbi BER=0 at Eb/N0 ≥ 2 dB;
  numba-jitted streaming Viterbi is bit-identical to the reference at **2.28M bits/s**
  (a full 15-min pass decodes in ~30 s).
- ⏳ **Stage 2** (calibrated against the first real locked pass): Reed-Solomon,
  exact CCSDS bit conventions, and VCDU → image reconstruction to PNG. You can't
  debug those blind — they get tuned against real frames.

## Hardware
- **137 MHz (Meteor LRPT):** any RTL-SDR / SDRplay + a V-dipole or QFH antenna.
  Optional Nooelec SAWbird+ NOAA (137 MHz) LNA for weak locations.
- **1.7 GHz (GOES, roadmap):** a 1.7 GHz dish/grid + a mandatory Nooelec
  SAWbird+ GOES LNA at the feed.

## The name
"wx" is the ham-radio shorthand for weather. The working title may change —
a leading candidate is **Birdbath**, after the hobbyist habit of pointing a
dish skyward to catch "birds" (satellites). Which is also the logo: little
birds splashing in a satellite dish. 🐦

---
*Part of the Tuna family: [TV Tuna](https://github.com/Felbs/Software-TV-Tuner)
· [Radio Tuna](https://github.com/Felbs/gr-radiotuna) · wxTuna. Every digital
decoder secretly knows how well it's doing — adaptive decoding closes the loop.*
