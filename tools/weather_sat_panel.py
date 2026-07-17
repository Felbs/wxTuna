"""weather_sat_panel.py — the weather-sat harvester's window.  localhost:8644

A glass cockpit for weather_sat.py: see exactly what the recorder is doing.
  - big STATE lamp: WAITING / RECORDING / IDLE / ERROR
  - live countdown to the next pass (sat, peak elevation, local time)
  - during a capture: elapsed/target bar, MB written, live signal meter
  - the upcoming pass schedule (next 24h), computed live from TLEs
  - capture history from the ledger, each with a click-to-render spectrum

Run alongside `weather_sat.py watch`; the panel only reads state files, so
it never touches the SDR (safe to leave open during a recording).

  python weather_sat_panel.py        # run under radioconda (SoapySDR + matplotlib)
"""
import io
import json
import math
import sys
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weather_sat as ws   # predictor + paths, shared

PORT = 8644

_sched = {"t": 0.0, "data": []}
_satcache = {"t": 0.0, "sats": None}


def _sats():
    if time.time() - _satcache["t"] > 1800 or _satcache["sats"] is None:
        _satcache["sats"] = ws.load_tles(refresh=False) if ws.TLE_CACHE.exists() \
            else ws.load_tles(refresh=True)
        _satcache["t"] = time.time()
    return _satcache["sats"]


def schedule(min_elev=15, hours=24):
    if time.time() - _sched["t"] > 60:
        try:
            ps = ws.predict_passes(_sats(), ws.utcnow(), hours, min_elev)
            _sched["data"] = [{
                "sat": p["sat"],
                "aos": p["aos"].isoformat() + "Z",
                "peak": p["maxt"].isoformat() + "Z",
                "los": p["los"].isoformat() + "Z",
                "max_elev": round(p["max"], 1),
                "dur_min": round((p["los"] - p["aos"]).total_seconds() / 60, 1),
            } for p in ps]
        except Exception as e:
            _sched["data"] = [{"error": str(e)}]
        _sched["t"] = time.time()
    return _sched["data"]


def live_elev(sat_name):
    try:
        sat = _sats().get(sat_name)
        if not sat:
            return None
        return round(ws.elevation_deg(sat, ws.utcnow()), 1)
    except Exception:
        return None


def read_status():
    try:
        return json.loads(ws.STATUS_FILE.read_text())
    except Exception:
        return {"state": "offline", "note": "recorder not running (no heartbeat)"}


def read_captures(n=25):
    if not ws.CAP_LOG.exists():
        return []
    rows = []
    for line in ws.CAP_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows[-n:][::-1]


def spectrum_png(fname):
    """Render an average spectrum + waterfall PNG from a .cs16 capture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = ws.CAP_DIR / Path(fname).name
    if not path.exists():
        raise FileNotFoundError(fname)
    # read a bounded slice from the middle of the file
    itemsize = 2  # int16
    total_i16 = path.stat().st_size // itemsize
    n_cplx = min(2_000_000, total_i16 // 2)
    start_cplx = max(0, (total_i16 // 2 - n_cplx) // 2)
    raw = np.fromfile(path, dtype=np.int16, count=2 * n_cplx,
                      offset=start_cplx * 2 * itemsize).astype(np.float32) / 32768.0
    iq = raw[0::2] + 1j * raw[1::2]
    fs = 250_000.0
    try:
        fs = float(json.loads((str(path) + ".json")).get("fs_hz", fs)) \
            if False else fs
        meta = json.loads(Path(str(path) + ".json").read_text())
        fs = float(meta.get("fs_hz", fs))
    except Exception:
        pass
    NFFT = 4096
    nseg = len(iq) // NFFT
    if nseg < 2:
        raise ValueError("capture too short to analyze")
    win = np.hanning(NFFT).astype(np.float32)
    seg = iq[:nseg * NFFT].reshape(nseg, NFFT) * win
    F = np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)
    P = 20 * np.log10(np.abs(F) + 1e-9)
    avg = P.mean(axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs)) / 1e3  # kHz
    # thin the waterfall rows for display
    step = max(1, nseg // 300)
    water = P[::step]

    INK = "#0d1014"; AM = "#ffb43a"; TE = "#33d0c4"; MU = "#7c8794"; TX = "#eaf0f6"
    plt.rcParams.update({"figure.facecolor": INK, "savefig.facecolor": INK})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 6.4),
                                 gridspec_kw={"height_ratios": [1, 1.5]})
    a1.plot(freqs, avg, color=AM, lw=1.0)
    a1.set_title(f"{path.name}  ·  average spectrum", color=TX,
                 fontsize=10, loc="left")
    a1.set_xlabel("offset from 137.9 MHz (kHz)", color=MU, fontsize=8)
    a1.set_ylabel("dB", color=MU, fontsize=8)
    a1.axvspan(-72, 72, color=TE, alpha=0.10)   # ~LRPT occupied band
    for ax in (a1,):
        ax.set_facecolor("#0a0d11")
        ax.tick_params(colors=MU, labelsize=7)
        for s in ax.spines.values():
            s.set_color("#20272f")
    ext = [freqs[0], freqs[-1], 0, water.shape[0]]
    a2.imshow(water, aspect="auto", origin="lower", extent=ext,
              cmap="inferno", vmin=np.percentile(water, 5),
              vmax=np.percentile(water, 99.5))
    a2.set_title("waterfall (time ↓)", color=TX, fontsize=10, loc="left")
    a2.set_xlabel("offset (kHz)", color=MU, fontsize=8)
    a2.tick_params(colors=MU, labelsize=7)
    for s in a2.spines.values():
        s.set_color("#20272f")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=INK)
    plt.close(fig)
    return buf.getvalue()


def check_sdr():
    try:
        import SoapySDR
        SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
        devs = SoapySDR.Device.enumerate()
        names = []
        for d in devs:
            keys = {k: d[k] for k in d.keys()}
            names.append(keys.get("label") or keys.get("driver") or str(keys))
        return {"count": len(devs), "names": names}
    except Exception as e:
        return {"count": 0, "names": [], "error": str(e)}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weather Sat Harvester</title>
<style>
:root{--ink:#0d1014;--panel:#12161c;--panel2:#0a0d11;--am:#ffb43a;--te:#33d0c4;
--red:#ff5c4d;--tx:#eaf0f6;--mu:#7c8794;--line:#20272f;}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--tx);
font-family:"DejaVu Sans Mono",ui-monospace,Menlo,Consolas,monospace;font-size:14px}
.wrap{max-width:1080px;margin:0 auto;padding:18px}
header{display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--line);
padding-bottom:12px;margin-bottom:16px;flex-wrap:wrap}
h1{font-size:16px;margin:0;letter-spacing:.14em;color:var(--tx)}
.freq{color:var(--am);font-size:20px;letter-spacing:.06em}
.pill{margin-left:auto;padding:5px 14px;border-radius:20px;font-weight:bold;
letter-spacing:.12em;border:1px solid var(--line)}
.s-waiting{color:var(--te);border-color:var(--te)}
.s-recording{color:var(--red);border-color:var(--red);animation:pulse 1.2s infinite}
.s-idle{color:var(--mu)}.s-error{color:var(--red);border-color:var(--red)}
.s-offline{color:var(--mu);border-color:var(--line)}
@keyframes pulse{50%{opacity:.45}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px;margin-bottom:16px}
.big{font-size:34px;color:var(--am);letter-spacing:.04em}
.mut{color:var(--mu);font-size:12px}
.row{display:flex;gap:24px;flex-wrap:wrap;align-items:baseline}
.bar{height:12px;background:var(--panel2);border:1px solid var(--line);
border-radius:6px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--am)}
.meter>i{background:linear-gradient(90deg,var(--te),var(--am),var(--red))}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mu);font-weight:normal;font-size:11px;letter-spacing:.08em}
.good{color:var(--am)}
.caps{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.cap{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px;
cursor:pointer}.cap:hover{border-color:var(--te)}
.cap b{color:var(--te)}
button{background:var(--panel2);color:var(--tx);border:1px solid var(--line);
border-radius:6px;padding:6px 12px;font-family:inherit;cursor:pointer}
button:hover{border-color:var(--am)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;
align-items:center;justify-content:center;padding:20px;z-index:9}
#modal img{max-width:100%;max-height:90vh;border:1px solid var(--line);border-radius:8px}
h2{font-size:12px;letter-spacing:.12em;color:var(--mu);margin:0 0 10px;font-weight:normal}
.eyebrow{color:var(--te)}
</style></head><body><div class="wrap">
<header>
  <h1>WEATHER&nbsp;SAT · LRPT&nbsp;HARVESTER</h1>
  <span class="freq">137.900&nbsp;MHz</span>
  <span id="pill" class="pill s-offline">—</span>
</header>

<div class="card" id="statecard">
  <h2>STATUS</h2>
  <div id="statebody" class="mut">connecting…</div>
</div>

<div class="card">
  <h2>UPCOMING PASSES <span class="mut">· next 24h · <span class="eyebrow">Meteor-M2 only (APT retired)</span></span></h2>
  <table><thead><tr><th>LOCAL</th><th>SAT</th><th>MAX ELEV</th><th>LENGTH</th></tr></thead>
  <tbody id="sched"></tbody></table>
</div>

<div class="card">
  <h2>CAPTURES <span class="mut">· click a card to render its spectrum</span></h2>
  <div id="caps" class="caps"></div>
</div>

<div class="card">
  <h2>SDR</h2>
  <button onclick="checkSDR()">CHECK RADIO</button>
  <span id="sdr" class="mut" style="margin-left:12px">unknown</span>
</div>
</div>
<div id="modal" onclick="this.style.display='none'"><img id="modimg" src=""></div>
<script>
let skew=0; // server-client clock skew (ms)
function fmtLocal(iso,off){const d=new Date(iso);
  return d.toLocaleString([], {weekday:'short',hour:'2-digit',minute:'2-digit'});}
function hms(s){s=Math.max(0,Math.floor(s));let h=(s/3600|0),m=((s%3600)/60|0),x=s%60;
  return (h?h+'h ':'')+(m<10&&h?'0':'')+m+'m '+(x<10?'0':'')+x+'s';}
async function poll(){
  let r=await fetch('/api/state'); let j=await r.json();
  skew = Date.now() - new Date(j.now).getTime();
  let st=j.status||{}; let s=(st.state||'offline');
  let pill=document.getElementById('pill');
  pill.className='pill s-'+s; pill.textContent=s.toUpperCase();
  let b=document.getElementById('statebody');
  if(s==='waiting'){
    let target=st.rec_start? new Date(st.rec_start).getTime():null;
    b.innerHTML=`<div class="row"><div><div class="mut">NEXT PASS</div>
      <div class="big">${st.next_sat||'—'}</div></div>
      <div><div class="mut">PEAK ELEVATION</div><div class="big">${st.next_max_elev??'—'}&deg;</div></div>
      <div><div class="mut">RECORDING IN</div><div class="big" id="cd">…</div></div></div>
      <div class="mut" style="margin-top:8px">peak at ${st.peak_local||fmtLocal(st.next_aos)} · min-elev ${st.min_elev??15}&deg;</div>`;
    window._cd=target;
  } else if(s==='recording'){
    let pct=st.target_s? Math.min(100,100*st.elapsed_s/st.target_s):0;
    let lvl=st.level_dbfs??-99; let lp=Math.max(0,Math.min(100,(lvl+80)/80*100));
    b.innerHTML=`<div class="row"><div><div class="mut">RECORDING</div>
      <div class="big" style="color:var(--red)">${st.sat||''}</div></div>
      <div><div class="mut">LIVE ELEVATION</div><div class="big">${j.recording_elev??'—'}&deg;</div></div>
      <div><div class="mut">WRITTEN</div><div class="big">${st.mb??0} MB</div></div></div>
      <div class="mut" style="margin-top:10px">progress ${Math.round(pct)}% · ${Math.round(st.elapsed_s||0)}/${Math.round(st.target_s||0)}s → ${st.file||''}</div>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <div class="mut" style="margin-top:10px">signal ${lvl} dBFS</div>
      <div class="bar meter"><i style="width:${lp}%"></i></div>`;
    window._cd=null;
  } else if(s==='error'){
    b.innerHTML=`<div class="big" style="color:var(--red)">ERROR</div>
      <div class="mut">${st.note||''}</div>`;
    window._cd=null;
  } else if(s==='offline'){
    b.innerHTML=`<div class="mut">${st.note||'recorder not running'} — start it with<br>
      <code>weather_sat.py watch</code>. This panel is read-only and safe to leave open.</div>`;
    window._cd=null;
  } else {
    let lc=st.last_capture;
    b.innerHTML=`<div class="mut">IDLE — waiting for the scheduler.</div>`+
      (lc?`<div style="margin-top:8px">last: <b>${lc.sat}</b> · ${lc.dur_s}s · ${lc.mb} MB · ${lc.file}</div>`:'');
    window._cd=null;
  }
  // schedule
  let tb=document.getElementById('sched'); tb.innerHTML='';
  (j.passes||[]).forEach(p=>{if(p.error){tb.innerHTML=`<tr><td colspan=4 class="mut">${p.error}</td></tr>`;return;}
    let g=p.max_elev>=25?'good':'';
    tb.innerHTML+=`<tr><td>${fmtLocal(p.peak)}</td><td>${p.sat}</td>
      <td class="${g}">${p.max_elev}&deg;${p.max_elev>=25?' ★':''}</td><td>${p.dur_min} min</td></tr>`;});
  // captures
  let cd=document.getElementById('caps'); cd.innerHTML='';
  (j.captures||[]).forEach(c=>{let el=document.createElement('div');el.className='cap';
    el.innerHTML=`<b>${c.sat}</b><div class="mut">${(c.ts||'').replace('T',' ').replace('Z','')}</div>
      <div>${c.dur_s||0}s · ${((c.bytes||0)/1e6).toFixed(0)} MB</div>`;
    el.onclick=()=>showSpec(c.file);cd.appendChild(el);});
  if(!(j.captures||[]).length) cd.innerHTML='<div class="mut">no captures yet — they appear here as passes are recorded</div>';
}
function tick(){if(window._cd){let s=(window._cd-(Date.now()-skew))/1000;
  let e=document.getElementById('cd'); if(e)e.textContent=hms(s);}}
function showSpec(file){let f=file.split(/[\\/]/).pop();
  let m=document.getElementById('modal');document.getElementById('modimg').src='/api/spectrum?file='+encodeURIComponent(f);
  m.style.display='flex';}
async function checkSDR(){let e=document.getElementById('sdr');e.textContent='checking…';
  let r=await fetch('/api/sdr');let j=await r.json();
  e.textContent=j.count? (j.count+' device(s): '+j.names.join(', ')) : ('0 devices found'+(j.error?' — '+j.error:''));
  e.style.color=j.count?'var(--te)':'var(--red)';}
poll();setInterval(poll,2000);setInterval(tick,1000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/api/state":
                st = read_status()
                rec_elev = live_elev(st.get("sat")) if st.get("state") == "recording" else None
                return self._send(200, json.dumps({
                    "now": ws.utcnow().isoformat() + "Z",
                    "status": st,
                    "recording_elev": rec_elev,
                    "passes": schedule(st.get("min_elev", 15)),
                    "captures": read_captures(),
                }))
            if u.path == "/api/sdr":
                return self._send(200, json.dumps(check_sdr()))
            if u.path == "/api/spectrum":
                f = parse_qs(u.query).get("file", [""])[0]
                png = spectrum_png(f)
                return self._send(200, png, "image/png")
            return self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}))


def main():
    print(f"weather-sat panel -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
