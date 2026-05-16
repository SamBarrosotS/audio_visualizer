"""
╔══════════════════════════════════════════════════════════╗
║       Dual Channel Audio Monitor — Tiempo Real           ║
║  Overlay · ΔFFT · FFT · Envelopes · Afinador             ║
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
      [Trigger slider]  [Rise/Fall]
    ┌──────────────────┐  ┌──────────────────┐
    │  AFINADOR  Ch1   │  │  AFINADOR  Ch2   │
    │  nota · Hz · ¢   │  │  nota · Hz · ¢   │
    └──────────────────┘  └──────────────────┘
"""

import matplotlib
matplotlib.use('TkAgg')      # cambiar a Qt5Agg si TkAgg falla

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, sosfilt
import threading
from collections import deque

# ═══════════════════════════════════════════════════════
#  PARÁMETROS FIJOS
# ═══════════════════════════════════════════════════════
DISP_SAMPLES     = 2048       # muestras en overlay
ENV_DISP_SAMPLES = 8192       # ventana del envelope follower
TUNE_SAMPLES     = 8192       # ventana del pitch detector
RING_SIZE        = max(DISP_SAMPLES, ENV_DISP_SAMPLES, TUNE_SAMPLES) * 4
FFT_SIZE         = 4096
SMOOTHING        = 0.55       # suavizado exponencial FFT
ENV_CUTOFF_HZ    = 20         # LPF del envelope follower
REFRESH_MS       = 33         # ~30 fps
YLIM_ALPHA       = 0.06       # velocidad de auto-escala del eje Y overlay
PEAK_HOLD_FRAMES = 60         # frames de peak hold (~2 s a 30 fps)
PEAK_DECAY       = 0.97

# ── Trigger ───────────────────────────────────────────
# Cuántas muestras hacia atrás mirar para el trigger.
# Más pequeño = menos latencia pero más probabilidad de no encontrar cruce.
# 3 × DISP_SAMPLES ≈ 128 ms a 48 kHz — suficiente para cualquier frecuencia
# de audio y prácticamente sin latencia perceptible.
TRIG_LOOKBACK    = DISP_SAMPLES * 3

# ── Pitch detector ────────────────────────────────────
TUNE_FMIN        = 20.0       # Hz — subgraves / CV eurorack
TUNE_FMAX        = 5000.0     # Hz — VCO alta frecuencia
TUNE_CONF_THR    = 0.40       # umbral de confianza ACF normalizada
TUNE_SILENCE_RMS = 0.002      # gate de silencio
TUNE_ALPHA       = 0.20       # suavizado EMA de la frecuencia detectada
TUNE_FREEZE_FR   = 18         # frames sin señal antes de borrar nota (~0.6 s)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

BLOCK_CANDIDATES = (256, 512, 1024, 2048)

# ═══════════════════════════════════════════════════════
#  ESTADO COMPARTIDO
# ═══════════════════════════════════════════════════════
state = {}
lock  = threading.Lock()
ui    = dict(trig_level=0.0, trig_edge='rise')


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
#  DISPOSITIVO
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
        padded     = np.zeros(FFT_SIZE, dtype=np.float32)
        n          = min(len(sig), bs)
        padded[:n] = sig[:n] * window[:n]
        fft        = np.abs(np.fft.rfft(padded)) / bs
        new_specs.append(20 * np.log10(fft + 1e-12))

    with lock:
        state['ring'][0].extend(ch0)
        state['ring'][1].extend(ch1)
        sb    = state['spec_buf']
        sb[0] = SMOOTHING * sb[0] + (1 - SMOOTHING) * new_specs[0]
        sb[1] = SMOOTHING * sb[1] + (1 - SMOOTHING) * new_specs[1]

# ═══════════════════════════════════════════════════════
#  TRIGGER  (nivel absoluto, sin restar DC)
# ═══════════════════════════════════════════════════════
def find_trigger(sig, level, edge):
    """
    Busca el cruce de nivel MÁS RECIENTE dentro de una ventana limitada
    (TRIG_LOOKBACK muestras) para minimizar la latencia de visualización.
    Devuelve el índice de inicio de la ventana a mostrar, o None.
    """
    end    = len(sig) - DISP_SAMPLES         # última posición válida de inicio
    start  = max(0, end - TRIG_LOOKBACK)     # no mirar más atrás que esto
    search = sig[start:end]
    if edge == 'rise':
        idx = np.where((search[:-1] < level) & (search[1:] >= level))[0]
    else:
        idx = np.where((search[:-1] > level) & (search[1:] <= level))[0]
    # Usamos el cruce MÁS RECIENTE (idx[-1]) → mínima latencia
    return start + int(idx[-1]) + 1 if len(idx) else None


def compute_envelope(sig):
    return sosfilt(state['sos_env'], np.abs(sig))

# ═══════════════════════════════════════════════════════
#  PITCH DETECTOR — autocorrelación + interpolación
# ═══════════════════════════════════════════════════════
def detect_pitch(sig, sr):
    """
    Autocorrelación MPM con interpolación parabólica sub-muestra.
    Devuelve (freq_hz, confidence) o (None, 0.0).
    """
    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms < TUNE_SILENCE_RMS:
        return None, 0.0

    s = (sig - np.mean(sig)).astype(np.float64)
    s *= np.hanning(len(s))

    n     = len(s)
    fft_n = 1 << (2 * n - 1).bit_length()    # próxima pot. de 2 >= 2n-1
    F     = np.fft.rfft(s, n=fft_n)
    acf   = np.fft.irfft(F * np.conj(F))[:n].real

    if acf[0] < 1e-12:
        return None, 0.0
    acf_norm = acf / acf[0]

    lag_min = max(1, int(sr / TUNE_FMAX))
    lag_max = min(n - 1, int(sr / TUNE_FMIN))
    if lag_min >= lag_max:
        return None, 0.0

    segment = acf_norm[lag_min:lag_max]
    peak_i  = int(np.argmax(segment))
    peak_v  = float(segment[peak_i])

    if peak_v < TUNE_CONF_THR:
        return None, 0.0

    # Interpolación parabólica para resolución sub-muestra
    i = peak_i + lag_min
    if 0 < i < n - 1:
        a, b, c = acf[i - 1], acf[i], acf[i + 1]
        denom   = a - 2 * b + c
        offset  = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        lag     = i + offset
    else:
        lag = float(i)

    return float(sr / lag), peak_v


def freq_to_note(freq, a4=440.0):
    """Devuelve (nota_str, octava, cents_desviacion) o (None, 0, 0.0)."""
    if freq is None or freq <= 0:
        return None, 0, 0.0
    semis  = 12.0 * np.log2(freq / a4)
    midi   = semis + 69.0
    midi_r = round(midi)
    cents  = (midi - midi_r) * 100.0
    note   = NOTE_NAMES[int(midi_r) % 12]
    octave = int(midi_r) // 12 - 1
    return note, octave, float(cents)

# ═══════════════════════════════════════════════════════
#  ESTILO VISUAL
# ═══════════════════════════════════════════════════════
BG = '#0d0d0d'
GR = '#1c1c1c'
ZL = '#2e2e2e'
C1 = '#2dcca0'   # Ch1 — verde agua
C2 = '#f07b50'   # Ch2 — naranja
CD = '#c084fc'   # delta FFT — lila
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
    xs = np.concatenate([t, t[::-1]])
    ys = np.concatenate([y, np.full(len(y), base)])
    return np.column_stack([xs, ys])


def _needle_color(cents):
    ac = abs(cents)
    if ac <=  5: return '#22dd55'   # verde — en tune
    if ac <= 15: return '#aadd22'   # verde-amarillo
    if ac <= 30: return '#ffbb00'   # amarillo-naranja
    return '#ff4422'                # rojo — muy desafinado

# ═══════════════════════════════════════════════════════
#  PANEL AFINADOR
# ═══════════════════════════════════════════════════════
def _build_tuner(fig, rect, ch_label, ch_color):
    """Construye un panel de afinador y devuelve sus artistas dinámicos."""
    ax = fig.add_axes(rect)
    ax.set_facecolor('#080808')
    ax.set_xlim(-58, 58)
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_edgecolor('#252525')
        sp.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Zonas de color (fondo estático — de fuera hacia adentro)
    for x0, x1, fc in [
        (-55,  55, '#180a0a'),   # fondo: rojo muy oscuro
        (-28,  28, '#161408'),   # ±28¢: ámbar oscuro
        ( -8,   8, '#0a1a0d'),   # ±8¢:  verde oscuro
        ( -3,   3, '#0c2210'),   # ±3¢:  verde vivo
    ]:
        ax.add_patch(Rectangle((x0, 0.08), x1 - x0, 0.82,
                                facecolor=fc, edgecolor='none', zorder=1))

    # Marcas de la escala
    for c in [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]:
        is_major = (c % 10 == 0)
        h_frac   = 0.72 if is_major else 0.45
        lw       = 1.4  if c == 0  else (0.9 if is_major else 0.5)
        col      = '#505050' if c == 0 else ('#303030' if is_major else '#202020')
        y0_      = 0.08 + (0.82 - h_frac * 0.82) / 2
        y1_      = y0_ + h_frac * 0.82
        ax.plot([c, c], [y0_, y1_], color=col, linewidth=lw,
                solid_capstyle='butt', zorder=2)

    # Etiquetas de cents
    for c in [-50, -25, 0, 25, 50]:
        lbl = f'{c:+d}' if c != 0 else '±0'
        ax.text(c, 0.065, lbl, color='#2e2e2e', fontsize=6,
                ha='center', va='top', zorder=3)

    # Título del canal
    ax.text(0, 0.975, f'── AFINADOR  {ch_label} ──',
            color=ch_color, fontsize=7.5, ha='center', va='top',
            fontfamily='monospace', fontweight='bold', zorder=3)

    # ── Artistas dinámicos ──────────────────────────────
    # Nota (grande, izquierda)
    note_txt = ax.text(-55, 0.52, '—',
                       color='#2a2a2a', fontsize=28, va='center', ha='left',
                       fontfamily='monospace', fontweight='bold', zorder=5)
    # Frecuencia (bajo la nota)
    hz_txt = ax.text(-55, 0.12, '',
                     color='#282828', fontsize=8, va='bottom', ha='left',
                     fontfamily='monospace', zorder=5)
    # Cents (derecha, grande)
    cents_txt = ax.text(55, 0.52, '',
                        color='#2a2a2a', fontsize=17, va='center', ha='right',
                        fontfamily='monospace', zorder=5)
    # Aguja
    needle, = ax.plot([0, 0], [0.05, 0.95],
                      color='#151515', linewidth=4.5,
                      solid_capstyle='round', zorder=6)
    # Punto en la punta
    needle_dot, = ax.plot([0], [0.95], 'o',
                          color='#151515', markersize=7, zorder=7)

    return dict(note_txt=note_txt, hz_txt=hz_txt,
                cents_txt=cents_txt, needle=needle, needle_dot=needle_dot)


def _update_tuner(t, freq, note, octave, cents):
    """Actualiza los artistas dinámicos de un panel de afinador."""
    if note is None:
        t['note_txt'].set_text('—')
        t['note_txt'].set_color('#2a2a2a')
        t['hz_txt'].set_text('')
        t['cents_txt'].set_text('')
        t['needle'].set_color('#101010')
        t['needle_dot'].set_color('#101010')
        t['needle'].set_xdata([0, 0])
        t['needle_dot'].set_xdata([0])
    else:
        nc = _needle_color(cents)
        t['note_txt'].set_text(f'{note}{octave}')
        t['note_txt'].set_color(nc)
        t['hz_txt'].set_text(f'{freq:.2f} Hz')
        sign = '+' if cents >= 0 else ''
        t['cents_txt'].set_text(f'{sign}{cents:.1f}¢')
        t['cents_txt'].set_color(nc)
        cx = float(np.clip(cents, -50, 50))
        t['needle'].set_xdata([cx, cx])
        t['needle'].set_color(nc)
        t['needle_dot'].set_xdata([cx])
        t['needle_dot'].set_color(nc)

# ═══════════════════════════════════════════════════════
#  FIGURA PRINCIPAL
# ═══════════════════════════════════════════════════════
def build_figure():
    sr       = state['sample_rate']
    freqs    = np.fft.rfftfreq(FFT_SIZE, 1 / sr)
    fmask    = freqs >= 20
    t_ms     = np.linspace(0, DISP_SAMPLES / sr * 1000, DISP_SAMPLES)
    t_env_ms = np.linspace(0, ENV_DISP_SAMPLES / sr * 1000, ENV_DISP_SAMPLES)

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.suptitle(
        f"Dual Channel Audio Monitor — Eurorack  "
        f"[{sr // 1000}kHz · env {ENV_CUTOFF_HZ}Hz · "
        f"tune {TUNE_FMIN:.0f}–{TUNE_FMAX:.0f}Hz]",
        color='#cccccc', fontsize=11, y=0.997)

    # Los paneles principales ocupan el tercio superior
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1],
        hspace=0.44, wspace=0.30,
        left=0.06, right=0.98, top=0.958, bottom=0.305,
    )

    # ── [0,0]  Overlay ────────────────────────────────────────
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

    dc_hline1 = ax_ov.axhline(0, color=C1, linewidth=0.9,
                               linestyle='--', zorder=4, alpha=0.55)
    dc_hline2 = ax_ov.axhline(0, color=C2, linewidth=0.9,
                               linestyle='--', zorder=4, alpha=0.55)
    dc_txt1 = ax_ov.text(t_ms[-1] * 0.98, 0, '', color=C1, fontsize=7,
                          va='center', ha='right', fontfamily='monospace', zorder=6)
    dc_txt2 = ax_ov.text(t_ms[-1] * 0.98, 0, '', color=C2, fontsize=7,
                          va='center', ha='right', fontfamily='monospace', zorder=6)
    poly_ov1 = PolyCollection([_fill_verts(t_ms, np.zeros(DISP_SAMPLES))],
                               facecolors=[C1], alpha=0.07, zorder=2)
    poly_ov2 = PolyCollection([_fill_verts(t_ms, np.zeros(DISP_SAMPLES))],
                               facecolors=[C2], alpha=0.06, zorder=2)
    ax_ov.add_collection(poly_ov1)
    ax_ov.add_collection(poly_ov2)

    # ── [1,0]  Delta FFT ──────────────────────────────────────
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

    # ── [0,1]  FFT superpuestas ────────────────────────────────
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

    # ── [1,1]  Envelope follower ───────────────────────────────
    env_ms_label = f"{ENV_DISP_SAMPLES / sr * 1000:.0f} ms"
    ax_env = fig.add_subplot(gs[1, 1])
    style_ax(ax_env, f"Envelope Follower — Ch1 + Ch2  [{env_ms_label}]", CL)
    ax_env.set_xlim(0, t_env_ms[-1])
    ax_env.set_ylim(-0.05, 1.15)
    ax_env.set_xlabel("Tiempo (ms)")
    ax_env.set_ylabel("Amplitud")
    for yg in [0.25, 0.5, 0.75, 1.0]:
        ax_env.axhline(yg, color=GR, linewidth=0.5)
    ax_env.axhline(0, color=ZL, linewidth=0.8, linestyle='--')
    line_rect1, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C1, linewidth=0.7, zorder=2, alpha=0.18)
    line_rect2, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C2, linewidth=0.7, zorder=2, alpha=0.16)
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
    env_peak1 = ax_env.axhline(0, color=C1, linewidth=1.4,
                                linestyle='--', zorder=5, alpha=0.75)
    env_peak2 = ax_env.axhline(0, color=C2, linewidth=1.4,
                                linestyle='--', zorder=5, alpha=0.75)
    env_txt1 = ax_env.text(t_env_ms[-1] * 0.98, 0.0, '',
                            color=C1, fontsize=7, va='bottom', ha='right',
                            fontfamily='monospace', zorder=6)
    env_txt2 = ax_env.text(t_env_ms[-1] * 0.98, 0.0, '',
                            color=C2, fontsize=7, va='top', ha='right',
                            fontfamily='monospace', zorder=6)
    ax_env.legend(handles=[line_env1, line_env2], loc='upper left', fontsize=7,
                  facecolor='#111111', edgecolor=GR, labelcolor='#aaaaaa')

    # ── Slider de trigger ──────────────────────────────────────
    ax_sl = fig.add_axes([0.06, 0.234, 0.54, 0.022])
    ax_sl.set_facecolor('#111111')
    slider = Slider(ax_sl, 'Trigger ', -1.5, 1.5, valinit=0.0, color=CT)
    slider.label.set_color('#666666')
    slider.label.set_fontsize(8)
    slider.valtext.set_color(CT)
    slider.valtext.set_fontsize(8)

    def on_slider(val):
        ui['trig_level'] = val
        trig_hline.set_ydata([val, val])
        trig_txt.set_position((t_ms[-1] * 0.01, val))
    slider.on_changed(on_slider)

    # ── Botón Rise / Fall ──────────────────────────────────────
    ax_be = fig.add_axes([0.63, 0.222, 0.12, 0.044])
    btn_edge = Button(ax_be, '▲ Rise', color='#141414', hovercolor='#242424')
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

    # ── Afinadores (dos paneles lado a lado) ───────────────────
    # [left, bottom, width, height] en coordenadas de figura
    tune1_art = _build_tuner(fig, [0.06, 0.01, 0.43, 0.195], 'Ch1', C1)
    tune2_art = _build_tuner(fig, [0.51, 0.01, 0.43, 0.195], 'Ch2', C2)

    art = dict(
        ax_ov=ax_ov,
        line_ov1=line_ov1,    line_ov2=line_ov2,
        poly_ov1=poly_ov1,    poly_ov2=poly_ov2,
        trig_hline=trig_hline, trig_txt=trig_txt,
        dc_hline1=dc_hline1,  dc_hline2=dc_hline2,
        dc_txt1=dc_txt1,      dc_txt2=dc_txt2,
        line_df=line_df,
        poly_df_pos=poly_df_pos, poly_df_neg=poly_df_neg,
        line_fft1=line_fft1,  line_fft2=line_fft2,
        ax_env=ax_env,
        line_rect1=line_rect1, line_rect2=line_rect2,
        line_env1=line_env1,   line_env2=line_env2,
        poly_env1=poly_env1,   poly_env2=poly_env2,
        env_peak1=env_peak1,   env_peak2=env_peak2,
        env_txt1=env_txt1,     env_txt2=env_txt2,
        tune1=tune1_art,       tune2=tune2_art,
        freqs_masked=freqs[fmask],
        t_env_ms=t_env_ms,
    )
    return fig, art, fmask, t_ms

# ═══════════════════════════════════════════════════════
#  TICK DE REFRESCO
# ═══════════════════════════════════════════════════════
def make_tick(art, fmask, t_ms):
    sr = state['sample_rate']

    ylim       = [-1.5, 1.5]
    peak       = [0.0, 0.0]
    peak_tmr   = [0, 0]
    pitch_freq = [None, None]   # frecuencia EMA por canal
    pitch_miss = [0, 0]         # frames sin detección

    def _dc_label(v):
        arrow = '▲' if v > 0.01 else ('▼' if v < -0.01 else '●')
        return f'{arrow} DC {v:+.3f}'

    def _fft_fill(x, y):
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([y, np.zeros(len(y))])
        return np.column_stack([xs, ys])

    def _smooth_pitch(ch, raw_freq, conf):
        if raw_freq is not None and conf >= TUNE_CONF_THR:
            pitch_miss[ch] = 0
            if pitch_freq[ch] is None:
                pitch_freq[ch] = raw_freq
            else:
                alpha = min(TUNE_ALPHA * (0.5 + conf), 0.9)
                pitch_freq[ch] = alpha * raw_freq + (1 - alpha) * pitch_freq[ch]
        else:
            pitch_miss[ch] += 1
            if pitch_miss[ch] > TUNE_FREEZE_FR:
                pitch_freq[ch] = None
        return pitch_freq[ch]

    def tick(frame):
        with lock:
            raw0 = np.array(state['ring'][0], dtype=np.float32)
            raw1 = np.array(state['ring'][1], dtype=np.float32)
            db0  = state['spec_buf'][0].copy()
            db1  = state['spec_buf'][1].copy()

        level = ui['trig_level']
        edge  = ui['trig_edge']

        # ── Overlay ──────────────────────────────────────
        # Fallback: si no hay cruce, mostrar las muestras MÁS RECIENTES
        idx = find_trigger(raw0, level, edge)
        idx = idx if idx is not None else len(raw0) - DISP_SAMPLES
        idx = max(0, min(idx, len(raw0) - DISP_SAMPLES))

        y0 = raw0[idx: idx + DISP_SAMPLES]
        y1 = raw1[idx: idx + DISP_SAMPLES]
        n  = min(len(y0), len(y1), DISP_SAMPLES)
        y0, y1, tm = y0[:n], y1[:n], t_ms[:n]

        art['line_ov1'].set_data(tm, y0)
        art['line_ov2'].set_data(tm, y1)
        art['poly_ov1'].set_verts([_fill_verts(tm, y0)])
        art['poly_ov2'].set_verts([_fill_verts(tm, y1)])

        # DC
        dc0 = float(np.mean(y0));  dc1 = float(np.mean(y1))
        art['dc_hline1'].set_ydata([dc0, dc0])
        art['dc_hline2'].set_ydata([dc1, dc1])
        art['dc_txt1'].set_position((tm[-1] * 0.98, dc0))
        art['dc_txt2'].set_position((tm[-1] * 0.98, dc1))
        art['dc_txt1'].set_text(_dc_label(dc0))
        art['dc_txt2'].set_text(_dc_label(dc1))

        # Auto-escala eje Y
        lo = min(0.0, float(np.min(y0)), float(np.min(y1)))
        hi = max(0.0, float(np.max(y0)), float(np.max(y1)))
        mg = max(0.05, (hi - lo) * 0.12)
        lo -= mg;  hi += mg
        ylim[0] = ylim[0] * (1 - YLIM_ALPHA) + lo * YLIM_ALPHA
        ylim[1] = ylim[1] * (1 - YLIM_ALPHA) + hi * YLIM_ALPHA
        art['ax_ov'].set_ylim(ylim[0], ylim[1])

        # ── Delta FFT ────────────────────────────────────
        diff = db0[fmask] - db1[fmask]
        fx   = art['freqs_masked']
        art['line_df'].set_data(fx, diff)
        art['poly_df_pos'].set_verts([_fft_fill(fx, np.maximum(diff, 0))])
        art['poly_df_neg'].set_verts([_fft_fill(fx, np.minimum(diff, 0))])

        # ── FFT ──────────────────────────────────────────
        art['line_fft1'].set_ydata(db0[fmask])
        art['line_fft2'].set_ydata(db1[fmask])

        # ── Envelope follower ─────────────────────────────
        t_env_ms = art['t_env_ms']
        ne = min(len(raw0), len(raw1), ENV_DISP_SAMPLES)
        r0, r1 = raw0[-ne:], raw1[-ne:]
        e0 = compute_envelope(r0)
        e1 = compute_envelope(r1)

        art['line_rect1'].set_data(t_env_ms[:ne], np.abs(r0))
        art['line_rect2'].set_data(t_env_ms[:ne], np.abs(r1))
        art['line_env1'].set_data(t_env_ms[:ne], e0)
        art['line_env2'].set_data(t_env_ms[:ne], e1)
        art['poly_env1'].set_verts([_fill_verts(t_env_ms[:ne], e0)])
        art['poly_env2'].set_verts([_fill_verts(t_env_ms[:ne], e1)])

        for ch, (r, e, ep, et) in enumerate(zip(
            (r0, r1), (e0, e1),
            (art['env_peak1'], art['env_peak2']),
            (art['env_txt1'],  art['env_txt2']),
        )):
            pk_new = float(np.max(e))
            if pk_new >= peak[ch]:
                peak[ch] = pk_new;  peak_tmr[ch] = PEAK_HOLD_FRAMES
            else:
                if peak_tmr[ch] > 0:
                    peak_tmr[ch] -= 1
                else:
                    peak[ch] *= PEAK_DECAY
            ep.set_ydata([peak[ch], peak[ch]])
            rms = float(np.sqrt(np.mean(r ** 2)))
            et.set_text(f'pk {peak[ch]:.3f}  rms {rms:.3f}')
            et.set_position((t_env_ms[ne - 1] * 0.98, peak[ch]))

        env_hi = max(float(np.max(e0)), float(np.max(e1)), peak[0], peak[1])
        art['ax_env'].set_ylim(-0.05, max(1.05, env_hi * 1.12))

        # ── Afinadores ───────────────────────────────────
        for ch, (raw, t_art) in enumerate(zip(
            (raw0, raw1), (art['tune1'], art['tune2'])
        )):
            nt = min(len(raw), TUNE_SAMPLES)
            raw_freq, conf = detect_pitch(raw[-nt:].astype(np.float64), sr)
            freq_s = _smooth_pitch(ch, raw_freq, conf)
            note, octave, cents = freq_to_note(freq_s)
            _update_tuner(t_art, freq_s, note, octave, cents)

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
    print(f"  → Afinador    : {TUNE_SAMPLES} muestras  "
          f"({1000 * TUNE_SAMPLES / sample_rate:.1f} ms)  "
          f"[{TUNE_FMIN:.0f}–{TUNE_FMAX:.0f} Hz]")
    if n_ch < 2:
        print("  ⚠️  Dispositivo MONO — Ch2 replicará Ch1")
    print("\n  [Cierra la ventana para salir]\n")

    fig, art, fmask, t_ms = build_figure()
    tick = make_tick(art, fmask, t_ms)

    # Guardar referencia para evitar que el GC destruya la animación
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
        plt.show()


if __name__ == '__main__':
    main()
