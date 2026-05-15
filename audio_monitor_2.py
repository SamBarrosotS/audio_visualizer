"""
╔══════════════════════════════════════════════════════════╗
║       Dual Channel Audio Monitor — Tiempo Real           ║
║  Overlay · ΔFFT · FFT · Envelopes                        ║
╚══════════════════════════════════════════════════════════╝
Requisitos:
    pip install sounddevice numpy matplotlib scipy

Layout:
    ┌─────────────────────┬─────────────────────┐
    │  Ch1 + Ch2 overlay  │  FFT Ch1 + Ch2      │
    │  (trigger absoluto) │                     │
    ├─────────────────────┼─────────────────────┤
    │  FFT Ch1 − FFT Ch2  │  Envelope Follower  │
    │  (diferencia en dB) │  Ch1 + Ch2          │
    └─────────────────────┴─────────────────────┘
    │  Slider trigger level        [Rise/Fall] [Onda/Env] │

FIXES respecto a la versión anterior:
  • Trigger sobre señal RAW (sin eliminar DC) → el nivel del slider
    corresponde exactamente a la posición en pantalla.
  • Eje Y del overlay auto-escalado (suavizado) → muestra el offset real.
  • FuncAnimation en vez de new_timer → sliders y botones interactivos.
  • Panel Lissajous eliminado → sustituido por Envelope Follower.
"""

import matplotlib
matplotlib.use('TkAgg')          # backend interactivo; cambiar a Qt5Agg si falla

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, sosfilt
import threading
from collections import deque

# ═══════════════════════════════════════════════════════
#  PARÁMETROS FIJOS
# ═══════════════════════════════════════════════════════
DISP_SAMPLES     = 2048          # muestras en el panel overlay
ENV_DISP_SAMPLES = 8192          # ventana más larga para ver envolventes
RING_SIZE        = max(DISP_SAMPLES, ENV_DISP_SAMPLES) * 4
FFT_SIZE         = 4096
SMOOTHING        = 0.55          # suavizado exponencial de la FFT
ENV_CUTOFF_HZ    = 20            # frecuencia de corte del follower (Hz)
REFRESH_MS       = 33            # ~30 fps
YLIM_ALPHA       = 0.06          # velocidad de auto-escala del eje Y overlay
PEAK_HOLD_FRAMES = 60            # frames que aguanta el peak hold (~2 s)
PEAK_DECAY       = 0.97          # factor de decaimiento tras hold

BLOCK_CANDIDATES = (256, 512, 1024, 2048)

# ═══════════════════════════════════════════════════════
#  ESTADO COMPARTIDO (hilo audio ↔ hilo GUI)
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
    state['sos_env'] = butter(
        2, ENV_CUTOFF_HZ / (sample_rate / 2), btype='low', output='sos')

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
#  CALLBACK DE AUDIO  (hilo de sounddevice)
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
#  TRIGGER  (señal RAW, nivel absoluto)
# ═══════════════════════════════════════════════════════
def find_trigger(sig, level, edge):
    """
    Busca el primer cruce del nivel en `sig` (sin eliminar DC).
    El nivel del slider es absoluto: corresponde directamente a la
    posición vertical de la forma de onda en pantalla.
    """
    search = sig[: len(sig) - DISP_SAMPLES]
    if edge == 'rise':
        idx = np.where((search[:-1] < level) & (search[1:] >= level))[0]
    else:
        idx = np.where((search[:-1] > level) & (search[1:] <= level))[0]
    return int(idx[0]) + 1 if len(idx) else None


def compute_envelope(sig):
    """Rectifica y pasa por LPF de 1er orden → envelope follower."""
    return sosfilt(state['sos_env'], np.abs(sig))

# ═══════════════════════════════════════════════════════
#  ESTILO VISUAL
# ═══════════════════════════════════════════════════════
BG = '#0d0d0d'
GR = '#1c1c1c'
ZL = '#2e2e2e'
C1 = '#2dcca0'   # Ch1 — verde agua
C2 = '#f07b50'   # Ch2 — naranja
CD = '#c084fc'   # diferencia FFT — lila
CL = '#facc15'   # envelope — amarillo
CT = '#ff4466'   # trigger — rojo


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
        ax.axhline(y,
                   color=ZL if y == 0 else GR,
                   linewidth=0.8 if y == 0 else 0.5,
                   linestyle='--' if y == 0 else '-')


def _fill_verts(t, y, base=0.0):
    """Vértices de un polígono cerrado entre y y base."""
    xs = np.concatenate([t, t[::-1]])
    ys = np.concatenate([y, np.full(len(y), base)])
    return np.column_stack([xs, ys])

# ═══════════════════════════════════════════════════════
#  FIGURA
# ═══════════════════════════════════════════════════════
def build_figure():
    sr       = state['sample_rate']
    freqs    = np.fft.rfftfreq(FFT_SIZE, 1 / sr)
    fmask    = freqs >= 20
    t_ms     = np.linspace(0, DISP_SAMPLES / sr * 1000, DISP_SAMPLES)
    t_env_ms = np.linspace(0, ENV_DISP_SAMPLES / sr * 1000, ENV_DISP_SAMPLES)

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        f"Dual Channel Audio Monitor — Eurorack  "
        f"[{sr//1000}kHz · env {ENV_CUTOFF_HZ}Hz]",
        color='#cccccc', fontsize=11, y=0.99)

    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1, 1, 0.18, 0.12],
        hspace=0.55, wspace=0.30,
        left=0.06, right=0.98, top=0.94, bottom=0.04,
    )

    # ── [0,0]  Overlay Ch1 + Ch2 ──────────────────────────────
    ax_ov = fig.add_subplot(gs[0, 0])
    style_ax(ax_ov, 'Ch1 + Ch2 — Overlay  (trigger absoluto + DC)', C1)
    ax_ov.set_xlim(0, t_ms[-1])
    ax_ov.set_ylim(-1.5, 1.5)
    ax_ov.set_xlabel("Tiempo (ms)")
    ax_ov.set_ylabel("Amplitud")
    add_hgrid(ax_ov, [-1.0, -0.5, 0, 0.5, 1.0])

    line_ov1, = ax_ov.plot(t_ms, np.zeros(DISP_SAMPLES),
                            color=C1, linewidth=1.3, zorder=3, label='Ch1')
    line_ov2, = ax_ov.plot(t_ms, np.zeros(DISP_SAMPLES),
                            color=C2, linewidth=1.3, zorder=3,
                            label='Ch2', alpha=0.85)
    trig_hline = ax_ov.axhline(0, color=CT, linewidth=0.9,
                                linestyle=':', zorder=4, alpha=0.9)
    trig_txt   = ax_ov.text(t_ms[-1] * 0.01, 0, ' ▶',
                             color=CT, fontsize=8, va='center', zorder=5)
    ax_ov.legend(loc='upper right', fontsize=7,
                 facecolor='#111111', edgecolor=GR, labelcolor='#aaaaaa')

    # Indicadores de DC offset — línea horizontal que sigue la media
    dc_hline1 = ax_ov.axhline(0, color=C1, linewidth=0.9,
                               linestyle='--', zorder=4, alpha=0.55)
    dc_hline2 = ax_ov.axhline(0, color=C2, linewidth=0.9,
                               linestyle='--', zorder=4, alpha=0.55)
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

    # ── [1,0]  Diferencia FFT ─────────────────────────────────
    ax_df = fig.add_subplot(gs[1, 0])
    style_ax(ax_df, "FFT Ch1 − FFT Ch2  (dB)", CD)
    ax_df.set_xscale('log')
    ax_df.set_xlim(20, sr / 2)
    ax_df.set_ylim(-40, 40)
    ax_df.set_xlabel("Frecuencia (Hz)")
    ax_df.set_ylabel("Diferencia (dB)")
    for xg in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if xg < sr / 2:
            ax_df.axvline(xg, color=GR, linewidth=0.5, zorder=1)
    ax_df.axhline( 10, color=GR, linewidth=0.5, linestyle='--')
    ax_df.axhline(-10, color=GR, linewidth=0.5, linestyle='--')
    ax_df.axhline(  0, color=ZL, linewidth=0.9, linestyle='--')
    line_df, = ax_df.plot(freqs[fmask], np.zeros(fmask.sum()),
                           color=CD, linewidth=1.2, zorder=3)
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

    # ── [1,1]  Envelope Follower Ch1 + Ch2 ────────────────────
    env_ms_label = f"{ENV_DISP_SAMPLES / sr * 1000:.0f} ms"
    ax_env = fig.add_subplot(gs[1, 1])
    style_ax(ax_env,
             f"Envelope Follower — Ch1 + Ch2  [{env_ms_label}]", CL)
    ax_env.set_xlim(0, t_env_ms[-1])
    ax_env.set_ylim(-0.05, 1.15)
    ax_env.set_xlabel("Tiempo (ms)")
    ax_env.set_ylabel("Amplitud")
    for yg in [0.25, 0.5, 0.75, 1.0]:
        ax_env.axhline(yg, color=GR, linewidth=0.5)
    ax_env.axhline(0, color=ZL, linewidth=0.8, linestyle='--')

    # Señal rectificada de fondo (muy tenue) → referencia visual
    line_rect1, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C1, linewidth=0.7, zorder=2, alpha=0.18)
    line_rect2, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C2, linewidth=0.7, zorder=2, alpha=0.16)
    # Envelope suavizado encima
    line_env1, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                              color=C1, linewidth=2.0, zorder=4, label='Env Ch1')
    line_env2, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                              color=C2, linewidth=2.0, zorder=4,
                              label='Env Ch2', alpha=0.85)
    poly_env1 = PolyCollection(
        [_fill_verts(t_env_ms, np.zeros(ENV_DISP_SAMPLES))],
        facecolors=[C1], alpha=0.13, zorder=3)
    poly_env2 = PolyCollection(
        [_fill_verts(t_env_ms, np.zeros(ENV_DISP_SAMPLES))],
        facecolors=[C2], alpha=0.11, zorder=3)
    ax_env.add_collection(poly_env1)
    ax_env.add_collection(poly_env2)
    # Peak hold
    env_peak1 = ax_env.axhline(0, color=C1, linewidth=1.4,
                                linestyle='--', zorder=5, alpha=0.75)
    env_peak2 = ax_env.axhline(0, color=C2, linewidth=1.4,
                                linestyle='--', zorder=5, alpha=0.75)
    env_txt1 = ax_env.text(t_env_ms[-1] * 0.98, 0.0, '',
                            color=C1, fontsize=7, va='bottom', ha='right',
                            fontfamily='monospace', zorder=6)
    env_txt2 = ax_env.text(t_env_ms[-1] * 0.98, 0.0, '',
                            color=C2, fontsize=7, va='top',   ha='right',
                            fontfamily='monospace', zorder=6)
    ax_env.legend(handles=[line_env1, line_env2],
                  loc='upper left', fontsize=7,
                  facecolor='#111111', edgecolor=GR, labelcolor='#aaaaaa')

    # ── Slider de trigger ──────────────────────────────────────
    # Rango extendido a ±1.5 para cubrir señales con DC offset
    ax_sl = fig.add_axes([0.08, 0.115, 0.52, 0.025])
    ax_sl.set_facecolor('#111111')
    slider = Slider(ax_sl, 'Trigger ', -1.5, 1.5, valinit=0.0, color=CT)
    slider.label.set_color('#888888')
    slider.label.set_fontsize(8)
    slider.valtext.set_color(CT)
    slider.valtext.set_fontsize(8)

    def on_slider(val):
        ui['trig_level'] = val
        trig_hline.set_ydata([val, val])
        trig_txt.set_position((t_ms[-1] * 0.01, val))
    slider.on_changed(on_slider)

    # ── Botón Rise / Fall ──────────────────────────────────────
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

    # ── Botón Onda / Envolvente ────────────────────────────────
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
        else:
            btn_mode.label.set_text('[ ONDA ] / Env')
    btn_mode.on_clicked(toggle_mode)

    art = dict(
        # overlay
        ax_ov=ax_ov,
        line_ov1=line_ov1,   line_ov2=line_ov2,
        poly_ov1=poly_ov1,   poly_ov2=poly_ov2,
        trig_hline=trig_hline, trig_txt=trig_txt,
        dc_hline1=dc_hline1, dc_hline2=dc_hline2,
        dc_txt1=dc_txt1,     dc_txt2=dc_txt2,
        # diferencia FFT
        line_df=line_df,
        poly_df_pos=poly_df_pos, poly_df_neg=poly_df_neg,
        # FFT
        line_fft1=line_fft1, line_fft2=line_fft2,
        # envelope
        ax_env=ax_env,
        line_rect1=line_rect1,   line_rect2=line_rect2,
        line_env1=line_env1,     line_env2=line_env2,
        poly_env1=poly_env1,     poly_env2=poly_env2,
        env_peak1=env_peak1,     env_peak2=env_peak2,
        env_txt1=env_txt1,       env_txt2=env_txt2,
        # datos auxiliares
        freqs_masked=freqs[fmask],
        t_env_ms=t_env_ms,
    )
    return fig, art, fmask, t_ms

# ═══════════════════════════════════════════════════════
#  TICK DE REFRESCO  (llamado por FuncAnimation)
# ═══════════════════════════════════════════════════════
def make_tick(art, fmask, t_ms):
    # Estado local de la closure ─────────────────────────
    ylim = [-1.5, 1.5]          # límites suavizados del eje Y overlay
    peak      = [0.0, 0.0]      # peak hold por canal
    peak_tmr  = [0,   0]        # contador de frames de hold

    def _dc_label(v):
        arrow = '▲' if v > 0.01 else ('▼' if v < -0.01 else '●')
        return f'{arrow} DC {v:+.3f}'

    def _fft_fill(x, y):
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([y, np.zeros(len(y))])
        return np.column_stack([xs, ys])

    def tick(frame):
        # ── Leer ring buffer ──────────────────────────────
        with lock:
            raw0 = np.array(state['ring'][0], dtype=np.float32)
            raw1 = np.array(state['ring'][1], dtype=np.float32)
            db0  = state['spec_buf'][0].copy()
            db1  = state['spec_buf'][1].copy()

        env   = ui['env_mode']
        level = ui['trig_level']
        edge  = ui['trig_edge']

        # ── Señal a mostrar en el overlay ─────────────────
        # Modo onda  → señal bruta (incluye DC real)
        # Modo env   → envolvente (siempre positiva, 0..1)
        proc0 = compute_envelope(raw0) if env else raw0
        proc1 = compute_envelope(raw1) if env else raw1

        # ── TRIGGER ABSOLUTO (sin restar DC) ─────────────
        # El nivel del slider coincide con la posición en pantalla.
        idx = find_trigger(proc0, level, edge)
        if idx is None:
            idx = len(proc0) - DISP_SAMPLES
        idx = max(0, min(idx, len(proc0) - DISP_SAMPLES))

        y0 = proc0[idx : idx + DISP_SAMPLES]
        y1 = proc1[idx : idx + DISP_SAMPLES]
        n  = min(len(y0), len(y1), DISP_SAMPLES)
        y0, y1, tm = y0[:n], y1[:n], t_ms[:n]

        art['line_ov1'].set_data(tm, y0)
        art['line_ov2'].set_data(tm, y1)
        art['poly_ov1'].set_verts([_fill_verts(tm, y0)])
        art['poly_ov2'].set_verts([_fill_verts(tm, y1)])

        # ── DC offset ─────────────────────────────────────
        dc0 = float(np.mean(y0))
        dc1 = float(np.mean(y1))
        art['dc_hline1'].set_ydata([dc0, dc0])
        art['dc_hline2'].set_ydata([dc1, dc1])
        art['dc_txt1'].set_position((tm[-1] * 0.98, dc0))
        art['dc_txt2'].set_position((tm[-1] * 0.98, dc1))
        art['dc_txt1'].set_text(_dc_label(dc0))
        art['dc_txt2'].set_text(_dc_label(dc1))

        # ── Auto-escala suavizada del eje Y overlay ────────
        # Siempre incluye 0 para que la línea de referencia sea visible.
        lo = min(0.0, float(np.min(y0)), float(np.min(y1)))
        hi = max(0.0, float(np.max(y0)), float(np.max(y1)))
        mg = max(0.05, (hi - lo) * 0.12)
        lo -= mg;  hi += mg
        ylim[0] = ylim[0] * (1 - YLIM_ALPHA) + lo * YLIM_ALPHA
        ylim[1] = ylim[1] * (1 - YLIM_ALPHA) + hi * YLIM_ALPHA
        art['ax_ov'].set_ylim(ylim[0], ylim[1])

        # ── Diferencia FFT ────────────────────────────────
        diff = db0[fmask] - db1[fmask]
        fx   = art['freqs_masked']
        art['line_df'].set_data(fx, diff)
        art['poly_df_pos'].set_verts([_fft_fill(fx, np.maximum(diff, 0))])
        art['poly_df_neg'].set_verts([_fft_fill(fx, np.minimum(diff, 0))])

        # ── FFT ───────────────────────────────────────────
        art['line_fft1'].set_ydata(db0[fmask])
        art['line_fft2'].set_ydata(db1[fmask])

        # ── Envelope panel ────────────────────────────────
        t_env_ms = art['t_env_ms']
        ne = min(len(raw0), len(raw1), ENV_DISP_SAMPLES)
        r0 = raw0[-ne:]          # últimas ne muestras brutas
        r1 = raw1[-ne:]
        e0 = compute_envelope(r0)
        e1 = compute_envelope(r1)

        art['line_rect1'].set_data(t_env_ms[:ne], np.abs(r0))
        art['line_rect2'].set_data(t_env_ms[:ne], np.abs(r1))
        art['line_env1'].set_data(t_env_ms[:ne], e0)
        art['line_env2'].set_data(t_env_ms[:ne], e1)
        art['poly_env1'].set_verts([_fill_verts(t_env_ms[:ne], e0)])
        art['poly_env2'].set_verts([_fill_verts(t_env_ms[:ne], e1)])

        # Peak hold Ch1
        pk0 = float(np.max(e0))
        if pk0 >= peak[0]:
            peak[0] = pk0;  peak_tmr[0] = PEAK_HOLD_FRAMES
        else:
            if peak_tmr[0] > 0:
                peak_tmr[0] -= 1
            else:
                peak[0] *= PEAK_DECAY

        # Peak hold Ch2
        pk1 = float(np.max(e1))
        if pk1 >= peak[1]:
            peak[1] = pk1;  peak_tmr[1] = PEAK_HOLD_FRAMES
        else:
            if peak_tmr[1] > 0:
                peak_tmr[1] -= 1
            else:
                peak[1] *= PEAK_DECAY

        art['env_peak1'].set_ydata([peak[0], peak[0]])
        art['env_peak2'].set_ydata([peak[1], peak[1]])

        rms0 = float(np.sqrt(np.mean(r0 ** 2)))
        rms1 = float(np.sqrt(np.mean(r1 ** 2)))
        art['env_txt1'].set_position((t_env_ms[ne - 1] * 0.98, peak[0]))
        art['env_txt2'].set_position((t_env_ms[ne - 1] * 0.98, peak[1]))
        art['env_txt1'].set_text(f'pk {peak[0]:.3f}  rms {rms0:.3f}')
        art['env_txt2'].set_text(f'pk {peak[1]:.3f}  rms {rms1:.3f}')

        # Auto-escala del eje Y del envelope (cubre picos)
        env_hi = max(float(np.max(e0)), float(np.max(e1)), peak[0], peak[1])
        art['ax_env'].set_ylim(-0.05, max(1.05, env_hi * 1.12))

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
          f"({1000 * block_size / sample_rate:.1f} ms/bloque)")
    print(f"  → Overlay     : {DISP_SAMPLES} muestras  "
          f"({1000 * DISP_SAMPLES / sample_rate:.1f} ms)")
    print(f"  → Envelope    : {ENV_DISP_SAMPLES} muestras  "
          f"({1000 * ENV_DISP_SAMPLES / sample_rate:.1f} ms)")
    if n_ch < 2:
        print("  ⚠️  Dispositivo MONO — Ch2 replicará Ch1")
    print("\n  [Cierra la ventana para salir]\n")

    fig, art, fmask, t_ms = build_figure()
    tick = make_tick(art, fmask, t_ms)

    # FuncAnimation se integra con el event loop de Tk/Qt → sliders y
    # botones responden correctamente (a diferencia de new_timer).
    # Guardar la referencia en fig._ani evita que el GC lo destruya.
    fig._ani = FuncAnimation(
        fig, tick,
        interval=REFRESH_MS,
        blit=False,
        cache_frame_data=False,
    )

    stream = sd.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=n_ch,
        blocksize=block_size,
        callback=audio_callback,
        dtype='float32',
    )

    with stream:
        plt.show()   # bloquea hasta cerrar la ventana


if __name__ == '__main__':
    main()
