"""
╔══════════════════════════════════════════════════════════╗
║       Dual Channel Audio Monitor — Tiempo Real           ║
║  Overlay · ΔFFT · FFT · Lissajous                        ║
╚══════════════════════════════════════════════════════════╝
Requisitos:
    pip install sounddevice numpy matplotlib scipy
    (entorno con TkAgg: conda activate audio)

Layout:
    ┌─────────────────────┬─────────────────────┐
    │  Ch1 + Ch2 overlay  │  FFT Ch1 + Ch2      │
    │  (trigger-estable)  │                     │
    ├─────────────────────┼─────────────────────┤
    │  FFT Ch1 − FFT Ch2  │  Lissajous          │
    │  (diferencia en dB) │                     │
    └─────────────────────┴─────────────────────┘
    │  Slider trigger level        [Rise/Fall] [Onda/Env] │
"""

import matplotlib
matplotlib.use('TkAgg')   # forzar backend interactivo antes de cualquier import de pyplot

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, sosfilt
import threading
from collections import deque

# ═══════════════════════════════════════════════════════
#  PARÁMETROS FIJOS
# ═══════════════════════════════════════════════════════
DISP_SAMPLES     = 2048
RING_SIZE        = DISP_SAMPLES * 4
FFT_SIZE         = 4096
SMOOTHING        = 0.55
ENV_CUTOFF       = 20
REFRESH_MS       = 33

BLOCK_CANDIDATES = (256, 512, 1024, 2048)

# ═══════════════════════════════════════════════════════
#  ESTADO COMPARTIDO
# ═══════════════════════════════════════════════════════
state = {}
lock  = threading.Lock()
ui    = dict(env_mode=False, trig_level=0.0, trig_edge='rise')

def init_state(sample_rate: int, block_size: int):
    state['sample_rate'] = sample_rate
    state['block_size']  = block_size
    state['ring'] = [
        deque(np.zeros(RING_SIZE, dtype=np.float32), maxlen=RING_SIZE),
        deque(np.zeros(RING_SIZE, dtype=np.float32), maxlen=RING_SIZE),
    ]
    state['spec_buf'] = [
        np.full(FFT_SIZE // 2 + 1, -90.0),
        np.full(FFT_SIZE // 2 + 1, -90.0),
    ]
    state['window']  = np.hanning(block_size).astype(np.float32)
    state['sos_env'] = butter(2, ENV_CUTOFF / (sample_rate / 2),
                              btype='low', output='sos')

# ═══════════════════════════════════════════════════════
#  SELECCIÓN Y NEGOCIACIÓN DE DISPOSITIVO
# ═══════════════════════════════════════════════════════
def select_device():
    devs = sd.query_devices()
    print("\n╔══ Dispositivos de entrada disponibles ══════════════════╗")
    for i, d in enumerate(devs):
        if d['max_input_channels'] < 1:
            continue
        stereo = " ★" if d['max_input_channels'] >= 2 else "  "
        print(f"  [{i:2d}]{stereo} {d['name'][:52]:<52} ({d['max_input_channels']}ch)")
    print("╚═════════════════════════════════════════════════════════╝")
    print("  ★ = tiene 2 o más canales de entrada\n")
    raw = input("Número de dispositivo [Enter = default del sistema]: ").strip()
    return int(raw) if raw else None


def negotiate_stream(device, n_ch):
    if device is not None:
        dev_info = sd.query_devices(device, 'input')
    else:
        idx = sd.default.device
        dev_info = sd.query_devices(
            idx[0] if isinstance(idx, (list, tuple)) else idx, 'input')
    native_sr     = int(dev_info['default_samplerate'])
    sr_candidates = [native_sr] + [r for r in (48000, 44100, 96000)
                                   if r != native_sr]
    for sr in sr_candidates:
        for bs in BLOCK_CANDIDATES:
            try:
                with sd.InputStream(device=device, samplerate=sr,
                                    channels=n_ch, blocksize=bs,
                                    dtype='float32'):
                    return sr, bs, dev_info
            except sd.PortAudioError:
                continue
    raise RuntimeError(f"No se pudo abrir el stream en '{dev_info['name']}'.")

# ═══════════════════════════════════════════════════════
#  CALLBACK DE AUDIO
# ═══════════════════════════════════════════════════════
def audio_callback(indata, frames, time, status):
    if status:
        print(f"  [!] {status}")
    bs, window = state['block_size'], state['window']
    ch0 = indata[:, 0].astype(np.float32)
    ch1 = indata[:, 1 if indata.shape[1] > 1 else 0].astype(np.float32)

    new_specs = []
    for sig in (ch0, ch1):
        padded = np.zeros(FFT_SIZE, dtype=np.float32)
        n = min(len(sig), bs)
        padded[:n] = sig[:n] * window[:n]
        fft = np.abs(np.fft.rfft(padded)) / bs
        new_specs.append(20 * np.log10(fft + 1e-12))

    with lock:
        state['ring'][0].extend(ch0)
        state['ring'][1].extend(ch1)
        sb = state['spec_buf']
        sb[0] = SMOOTHING * sb[0] + (1 - SMOOTHING) * new_specs[0]
        sb[1] = SMOOTHING * sb[1] + (1 - SMOOTHING) * new_specs[1]

# ═══════════════════════════════════════════════════════
#  TRIGGER  +  ENVOLVENTE
# ═══════════════════════════════════════════════════════
def find_trigger(sig, level, edge):
    search = sig[:RING_SIZE - DISP_SAMPLES]
    if edge == 'rise':
        idx = np.where((search[:-1] < level) & (search[1:] >= level))[0]
    else:
        idx = np.where((search[:-1] > level) & (search[1:] <= level))[0]
    return int(idx[0]) + 1 if len(idx) else None

def compute_envelope(sig):
    return sosfilt(state['sos_env'], np.abs(sig))

# ═══════════════════════════════════════════════════════
#  ESTILO VISUAL
# ═══════════════════════════════════════════════════════
BG = '#0d0d0d'
GR = '#1c1c1c'
ZL = '#2e2e2e'
C1 = '#2dcca0'
C2 = '#f07b50'
CD = '#c084fc'
CL = '#facc15'
CT = '#ff4466'

def style_ax(ax, title, color):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GR)
    ax.tick_params(colors='#555555', labelsize=8)
    ax.set_title(title, color=color, fontsize=9, pad=5)
    ax.xaxis.label.set_color('#555555')
    ax.yaxis.label.set_color('#555555')

def add_hgrid(ax, ys):
    for y in ys:
        ax.axhline(y, color=ZL if y == 0 else GR,
                   linewidth=0.8 if y == 0 else 0.5,
                   linestyle='--' if y == 0 else '-')

def _fill_verts(t, y):
    xs = np.concatenate([t, t[::-1]])
    ys = np.concatenate([y, np.zeros(len(y))])
    return np.column_stack([xs, ys])

# ═══════════════════════════════════════════════════════
#  FIGURA
# ═══════════════════════════════════════════════════════
def build_figure():
    sr    = state['sample_rate']
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / sr)
    fmask = freqs >= 20
    t_ms  = np.linspace(0, DISP_SAMPLES / sr * 1000, DISP_SAMPLES)

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle("Dual Channel Audio Monitor", color='#cccccc', fontsize=12, y=0.99)

    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1, 1, 0.18, 0.12],
        hspace=0.55, wspace=0.30,
        left=0.06, right=0.98, top=0.94, bottom=0.04,
    )

    # ── [0,0]  Overlay Ch1 + Ch2 ──────────────────────────────
    ax_ov = fig.add_subplot(gs[0, 0])
    style_ax(ax_ov, 'Ch1 + Ch2 — Overlay  (trigger auto-DC)', C1)
    ax_ov.set_xlim(0, t_ms[-1])
    ax_ov.set_ylim(-2.0, 2.0)
    ax_ov.set_xlabel("Tiempo (ms)")
    ax_ov.set_ylabel("Amplitud")
    add_hgrid(ax_ov, [-1.0, -0.5, 0, 0.5, 1.0])

    line_ov1, = ax_ov.plot(t_ms, np.zeros(DISP_SAMPLES),
                            color=C1, linewidth=1.3, zorder=3, label='Ch1')
    line_ov2, = ax_ov.plot(t_ms, np.zeros(DISP_SAMPLES),
                            color=C2, linewidth=1.3, zorder=3,
                            label='Ch2', alpha=0.85)
    trig_hline = ax_ov.axhline(0, color=CT, linewidth=0.8,
                                linestyle=':', zorder=4, alpha=0.8)
    trig_txt   = ax_ov.text(t_ms[-1] * 0.01, 0, ' ▶',
                             color=CT, fontsize=8, va='center', zorder=5)
    ax_ov.legend(loc='upper right', fontsize=7,
                 facecolor='#111111', edgecolor=GR, labelcolor='#aaaaaa')

    # ── Indicadores de DC offset ───────────────────────────
    # Línea horizontal punteada que sigue la media de cada canal
    dc_hline1 = ax_ov.axhline(0, color=C1, linewidth=0.9,
                               linestyle='--', zorder=4, alpha=0.6)
    dc_hline2 = ax_ov.axhline(0, color=C2, linewidth=0.9,
                               linestyle='--', zorder=4, alpha=0.6)
    # Etiqueta con el valor numérico en el borde derecho
    dc_txt1 = ax_ov.text(t_ms[-1] * 0.98, 0, '', color=C1,
                          fontsize=7, va='center', ha='right',
                          fontfamily='monospace', zorder=6)
    dc_txt2 = ax_ov.text(t_ms[-1] * 0.98, 0, '', color=C2,
                          fontsize=7, va='center', ha='right',
                          fontfamily='monospace', zorder=6)
    poly_ov1 = PolyCollection([_fill_verts(t_ms, np.zeros(DISP_SAMPLES))],
                               facecolors=[C1], alpha=0.07, zorder=2)
    poly_ov2 = PolyCollection([_fill_verts(t_ms, np.zeros(DISP_SAMPLES))],
                               facecolors=[C2], alpha=0.06, zorder=2)
    ax_ov.add_collection(poly_ov1)
    ax_ov.add_collection(poly_ov2)

    # ── [1,0]  Diferencia FFT (Ch1 dB − Ch2 dB) ───────────────
    ax_df = fig.add_subplot(gs[1, 0])
    style_ax(ax_df, "FFT Ch1 − FFT Ch2  (dB)", CD)
    ax_df.set_xscale('log')
    ax_df.set_xlim(20, sr / 2)
    ax_df.set_ylim(-40, 40)      # ±40 dB de margen
    ax_df.set_xlabel("Frecuencia (Hz)")
    ax_df.set_ylabel("Diferencia (dB)")
    for xg in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if xg < sr / 2:
            ax_df.axvline(xg, color=GR, linewidth=0.5, zorder=1)
    ax_df.axhline( 10, color=GR, linewidth=0.5, linestyle='--')
    ax_df.axhline(-10, color=GR, linewidth=0.5, linestyle='--')
    ax_df.axhline(  0, color=ZL, linewidth=0.9, linestyle='--')
    # Relleno positivo (Ch1 > Ch2) y negativo (Ch2 > Ch1) en colores distintos
    line_df, = ax_df.plot(freqs[fmask], np.zeros(fmask.sum()),
                           color=CD, linewidth=1.2, zorder=3)
    # Área bajo la curva — dos PolyCollections para pos/neg
    poly_df_pos = PolyCollection([], facecolors=[C1], alpha=0.15, zorder=2)
    poly_df_neg = PolyCollection([], facecolors=[C2], alpha=0.15, zorder=2)
    ax_df.add_collection(poly_df_pos)
    ax_df.add_collection(poly_df_neg)

    # ── [0,1]  FFT Ch1 + Ch2 superpuestas ─────────────────────
    ax_fft = fig.add_subplot(gs[0, 1])
    style_ax(ax_fft, "Espectro FFT — Ch1 + Ch2", C1)
    ax_fft.set_xscale('log')
    ax_fft.set_xlim(20, sr / 2)
    ax_fft.set_ylim(-90, 6)
    ax_fft.set_xlabel("Frecuencia (Hz)")
    ax_fft.set_ylabel("Nivel (dB)")
    for xg in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if xg < sr / 2:
            ax_fft.axvline(xg, color=GR, linewidth=0.5, zorder=1)
    ax_fft.axhline(-60, color=GR, linewidth=0.5, linestyle='--')
    ax_fft.axhline(-30, color=GR, linewidth=0.5, linestyle='--')
    ax_fft.axhline(  0, color=ZL, linewidth=0.8, linestyle='--')
    line_fft1, = ax_fft.plot(freqs[fmask], np.full(fmask.sum(), -90.0),
                              color=C1, linewidth=1.2, zorder=3, label='Ch1')
    line_fft2, = ax_fft.plot(freqs[fmask], np.full(fmask.sum(), -90.0),
                              color=C2, linewidth=1.2, zorder=3,
                              label='Ch2', alpha=0.85)
    ax_fft.legend(loc='upper right', fontsize=7,
                  facecolor='#111111', edgecolor=GR, labelcolor='#aaaaaa')

    # ── [1,1]  Lissajous ──────────────────────────────────────
    ax_li = fig.add_subplot(gs[1, 1])
    style_ax(ax_li, "Lissajous  (X=Ch1  Y=Ch2)", CL)
    ax_li.set_xlim(-1.15, 1.15)
    ax_li.set_ylim(-1.15, 1.15)
    ax_li.set_xlabel("Ch1")
    ax_li.set_ylabel("Ch2")
    ax_li.set_aspect('equal', adjustable='box')
    ax_li.axhline(0, color=ZL, linewidth=0.6)
    ax_li.axvline(0, color=ZL, linewidth=0.6)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_li.plot(np.cos(theta), np.sin(theta), color=GR, linewidth=0.5)
    liss_lc = LineCollection(np.zeros((DISP_SAMPLES - 1, 2, 2)),
                              cmap='YlOrRd', linewidth=1.0, zorder=3)
    liss_lc.set_array(np.linspace(0, 1, DISP_SAMPLES - 1))
    ax_li.add_collection(liss_lc)
    liss_dot, = ax_li.plot([0], [0], 'o', color=CL, markersize=4, zorder=5)

    # ── [2,*]  Slider trigger ──────────────────────────────────
    ax_sl = fig.add_axes([0.08, 0.115, 0.52, 0.025])
    ax_sl.set_facecolor('#111111')
    slider = Slider(ax_sl, 'Trigger ', -1.0, 1.0, valinit=0.0, color=CT)
    slider.label.set_color('#888888')
    slider.label.set_fontsize(8)
    slider.valtext.set_color(CT)
    slider.valtext.set_fontsize(8)

    def on_slider(val):
        ui['trig_level'] = val
        trig_hline.set_ydata([val, val])
        trig_txt.set_position((t_ms[-1] * 0.01, val))
    slider.on_changed(on_slider)

    # ── [2, der]  Botón Rise/Fall ──────────────────────────────
    ax_be = fig.add_axes([0.63, 0.095, 0.12, 0.050])
    btn_edge = Button(ax_be, '▲ Rise', color='#181818', hovercolor='#2a2a2a')
    btn_edge.label.set_color(CT)
    btn_edge.label.set_fontsize(9)
    btn_edge.label.set_fontfamily('monospace')

    def toggle_edge(_):
        if ui['trig_edge'] == 'rise':
            ui['trig_edge'] = 'fall'
            btn_edge.label.set_text('▼ Fall')
        else:
            ui['trig_edge'] = 'rise'
            btn_edge.label.set_text('▲ Rise')
    btn_edge.on_clicked(toggle_edge)

    # ── [2, der]  Botón Onda/Envolvente ───────────────────────
    ax_bm = fig.add_axes([0.77, 0.095, 0.20, 0.050])
    btn_mode = Button(ax_bm, '[ ONDA ] / Env',
                      color='#181818', hovercolor='#2a2a2a')
    btn_mode.label.set_color('#aaaaaa')
    btn_mode.label.set_fontsize(8)
    btn_mode.label.set_fontfamily('monospace')

    def toggle_mode(_):
        ui['env_mode'] = not ui['env_mode']
        if ui['env_mode']:
            btn_mode.label.set_text('Onda / [ ENV ]')
            ax_ov.set_ylim(-0.05, 1.15)
        else:
            btn_mode.label.set_text('[ ONDA ] / Env')
            ax_ov.set_ylim(-2.0, 2.0)
    btn_mode.on_clicked(toggle_mode)

    art = dict(
        line_ov1=line_ov1, line_ov2=line_ov2,
        poly_ov1=poly_ov1, poly_ov2=poly_ov2,
        trig_hline=trig_hline, trig_txt=trig_txt,
        dc_hline1=dc_hline1, dc_hline2=dc_hline2,
        dc_txt1=dc_txt1, dc_txt2=dc_txt2,
        line_df=line_df,
        poly_df_pos=poly_df_pos, poly_df_neg=poly_df_neg,
        line_fft1=line_fft1, line_fft2=line_fft2,
        liss_lc=liss_lc, liss_dot=liss_dot,
        ax_df=ax_df,           # necesario para recalcular fill pos/neg
        freqs_masked=freqs[fmask],
    )
    return fig, art, freqs, fmask, t_ms

# ═══════════════════════════════════════════════════════
#  TICK DE REFRESCO
# ═══════════════════════════════════════════════════════
def make_tick(fig, art, freqs, fmask, t_ms):

    def tick():
        with lock:
            raw0 = np.array(state['ring'][0], dtype=np.float32)
            raw1 = np.array(state['ring'][1], dtype=np.float32)
            db0  = state['spec_buf'][0].copy()
            db1  = state['spec_buf'][1].copy()

        env   = ui['env_mode']
        level = ui['trig_level']
        edge  = ui['trig_edge']

        # ── Onda con trigger ──────────────────────────────────
        proc0 = compute_envelope(raw0) if env else raw0
        proc1 = compute_envelope(raw1) if env else raw1

        # El trigger busca en la señal SIN DC para que funcione
        # aunque haya offset — pero mostramos la señal original
        trig_src = proc0 - np.mean(proc0)
        idx = find_trigger(trig_src, level, edge)
        if idx is None:
            idx = RING_SIZE - DISP_SAMPLES

        y0 = proc0[idx: idx + DISP_SAMPLES]
        y1 = proc1[idx: idx + DISP_SAMPLES]
        n  = min(len(y0), len(y1), DISP_SAMPLES)
        y0, y1, tm = y0[:n], y1[:n], t_ms[:n]

        art['line_ov1'].set_data(tm, y0)
        art['line_ov2'].set_data(tm, y1)
        art['poly_ov1'].set_verts([_fill_verts(tm, y0)])
        art['poly_ov2'].set_verts([_fill_verts(tm, y1)])

        # ── DC offset: media de la ventana visible ────────────
        dc0 = float(np.mean(y0))
        dc1 = float(np.mean(y1))
        art['dc_hline1'].set_ydata([dc0, dc0])
        art['dc_hline2'].set_ydata([dc1, dc1])
        def dc_label(v):
            arrow = '▲' if v > 0.01 else ('▼' if v < -0.01 else '●')
            return f'{arrow} DC {v:+.3f}'
        art['dc_txt1'].set_position((tm[-1] * 0.98, dc0))
        art['dc_txt2'].set_position((tm[-1] * 0.98, dc1))
        art['dc_txt1'].set_text(dc_label(dc0))
        art['dc_txt2'].set_text(dc_label(dc1))

        # ── Diferencia FFT ────────────────────────────────────
        diff_db = db0[fmask] - db1[fmask]
        fx      = art['freqs_masked']

        art['line_df'].set_data(fx, diff_db)

        # Relleno: verde donde Ch1 > Ch2, naranja donde Ch2 > Ch1
        pos = np.maximum(diff_db, 0)
        neg = np.minimum(diff_db, 0)

        def fft_fill_verts(x, y):
            # polígono cerrado sobre y=0 en escala log → usar x directamente
            xs = np.concatenate([x, x[::-1]])
            ys = np.concatenate([y, np.zeros(len(y))])
            return np.column_stack([xs, ys])

        art['poly_df_pos'].set_verts([fft_fill_verts(fx, pos)])
        art['poly_df_neg'].set_verts([fft_fill_verts(fx, neg)])

        # ── FFT ───────────────────────────────────────────────
        art['line_fft1'].set_ydata(db0[fmask])
        art['line_fft2'].set_ydata(db1[fmask])

        # ── Lissajous ─────────────────────────────────────────
        pts  = np.column_stack([y0, y1])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        art['liss_lc'].set_segments(segs)
        art['liss_lc'].set_array(np.linspace(0, 1, len(segs)))
        art['liss_dot'].set_data([y0[-1]], [y1[-1]])

        fig.canvas.draw_idle()

    return tick

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    device = select_device()

    if device is not None:
        _di = sd.query_devices(device, 'input')
    else:
        idx = sd.default.device
        _di = sd.query_devices(
            idx[0] if isinstance(idx, (list, tuple)) else idx, 'input')
    n_ch = min(2, _di['max_input_channels'])

    print("\n  Negociando parámetros con el dispositivo…")
    sample_rate, block_size, dev_info = negotiate_stream(device, n_ch)
    init_state(sample_rate, block_size)

    print(f"  → Dispositivo : {dev_info['name']}")
    print(f"  → Canales     : {n_ch}")
    print(f"  → Sample rate : {sample_rate} Hz")
    print(f"  → Block size  : {block_size} muestras  "
          f"({1000*block_size/sample_rate:.1f} ms/bloque)")
    print(f"  → Ventana     : {DISP_SAMPLES} muestras  "
          f"({1000*DISP_SAMPLES/sample_rate:.1f} ms visibles)")
    if n_ch < 2:
        print("  ⚠️  Dispositivo MONO — Ch2 replicará Ch1")
    print("\n  [Cierra la ventana para salir]\n")

    fig, art, freqs, fmask, t_ms = build_figure()
    tick = make_tick(fig, art, freqs, fmask, t_ms)

    timer = fig.canvas.new_timer(interval=REFRESH_MS)
    timer.add_callback(tick)

    stream = sd.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=n_ch,
        blocksize=block_size,
        callback=audio_callback,
        dtype='float32',
    )

    with stream:
        timer.start()
        plt.show()
        timer.stop()


if __name__ == '__main__':
    main()
