"""
╔══════════════════════════════════════════════════════════════╗
║          Eurorack Dual Channel Audio Monitor                 ║
║   Overlay · ΔFFT · FFT · Envelope Follower · Tuner          ║
╚══════════════════════════════════════════════════════════════╝

Requirements:
    pip install sounddevice numpy matplotlib scipy

Runs on Windows, macOS, and Linux (requires python3-tk on Linux).
The native Tk menu bar integrates automatically with the OS:
  - macOS  → menu appears in the system menu bar at the top
  - Windows/Linux → menu bar inside the application window
"""

# ---------------------------------------------------------------------------
# Backend selection — TkAgg first, Qt5Agg as fallback.
# Must happen before any other matplotlib import.
# ---------------------------------------------------------------------------
import matplotlib
try:
    matplotlib.use("TkAgg")
    import tkinter as _tk_test  # noqa: F401  (verify Tk is available)
except Exception:
    matplotlib.use("Qt5Agg")

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import sys
import platform
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, sosfilt
import threading
from collections import deque

APP_NAME    = "Eurorack Audio Monitor"
APP_VERSION = "1.0.0"
APP_AUTHOR  = "Samuel Barroso Bellido"
APP_URL     = "https://github.com/SamBarrosotS/"
APP_EMAIL   = "your@email.com"

# ---------------------------------------------------------------------------
# Fixed parameters  (can be changed via Edit › Parameters)
# ---------------------------------------------------------------------------
DISP_SAMPLES     = 2048       # overlay window length (samples)
ENV_DISP_SAMPLES = 8192       # envelope follower window (samples)
TUNE_SAMPLES     = 8192       # pitch detector window (samples)
RING_SIZE        = max(DISP_SAMPLES, ENV_DISP_SAMPLES, TUNE_SAMPLES) * 4
FFT_SIZE         = 4096
SMOOTHING        = 0.55       # FFT exponential smoothing coefficient
ENV_CUTOFF_HZ    = 20         # envelope follower LPF cutoff (Hz)
REFRESH_MS       = 33         # animation interval (~30 fps)
PEAK_HOLD_FRAMES = 60         # peak-hold duration (frames, ~2 s at 30 fps)
PEAK_DECAY       = 0.97       # peak-hold decay factor per frame

# Trigger: look back at most this many samples so the display is near-live
TRIG_LOOKBACK    = DISP_SAMPLES * 3   # ~128 ms at 48 kHz

# Pitch detector limits
TUNE_FMIN        = 20.0       # minimum detectable frequency (Hz)
TUNE_FMAX        = 5000.0     # maximum detectable frequency (Hz)
TUNE_CONF_THR    = 0.40       # ACF confidence threshold (0–1)
TUNE_SILENCE_RMS = 0.002      # RMS gate — below this is treated as silence
TUNE_ALPHA       = 0.20       # EMA smoothing for detected frequency
TUNE_FREEZE_FR   = 18         # frames without signal before clearing note

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

BLOCK_CANDIDATES = (256, 512, 1024, 2048)

# Available voltage ranges: (button label, volt_scale, volt_range)
# volt_scale is stored but signal is NOT rescaled — only Y axis limits change.
VOLT_RANGES = [
    ("[-1, 1]",   1,  1),
    ("[-5, 5]",   5,  5),
    ("[-10,10]", 10, 10),
]

# ---------------------------------------------------------------------------
# Shared state  (audio thread ↔ GUI thread)
# ---------------------------------------------------------------------------
state: dict = {}
lock         = threading.Lock()
ui           = dict(
    trig_level        = 0.0,    # trigger level in raw signal units (±1.0 full scale)
    trig_edge         = "rise",
    volt_scale        = 1,      # stored for reference — does NOT rescale signal
    volt_range        = 1,      # Y axis half-range (volts label only)
    envelope_enabled  = False,  # envelope follower off by default (saves CPU)
)


def init_state(sample_rate: int, block_size: int) -> None:
    """Initialise all shared buffers for a given audio configuration."""
    state["sample_rate"] = sample_rate
    state["block_size"]  = block_size
    state["ring"] = [
        deque(np.zeros(RING_SIZE, dtype=np.float32), maxlen=RING_SIZE),
        deque(np.zeros(RING_SIZE, dtype=np.float32), maxlen=RING_SIZE),
    ]
    state["spec_buf"] = [
        np.full(FFT_SIZE // 2 + 1, -90.0),
        np.full(FFT_SIZE // 2 + 1, -90.0),
    ]
    state["window"]  = np.hanning(block_size).astype(np.float32)
    state["sos_env"] = butter(
        2, ENV_CUTOFF_HZ / (sample_rate / 2), btype="low", output="sos")

# ---------------------------------------------------------------------------
# Device negotiation
# ---------------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """Return only devices that have at least one input channel."""
    return [
        {"index": i, **d}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] >= 1
    ]


def negotiate_stream(device_index, n_ch: int):
    """
    Try common sample rates and block sizes until the device accepts one.
    Returns (sample_rate, block_size, dev_info).
    """
    if device_index is not None:
        dev_info = sd.query_devices(device_index, "input")
    else:
        idx = sd.default.device
        dev_info = sd.query_devices(
            idx[0] if isinstance(idx, (list, tuple)) else idx, "input")

    native_sr     = int(dev_info["default_samplerate"])
    sr_candidates = [native_sr] + [r for r in (48000, 44100, 96000)
                                   if r != native_sr]
    for sr in sr_candidates:
        for bs in BLOCK_CANDIDATES:
            try:
                with sd.InputStream(device=device_index, samplerate=sr,
                                    channels=n_ch, blocksize=bs,
                                    dtype="float32"):
                    return sr, bs, dev_info
            except sd.PortAudioError:
                continue
    raise RuntimeError(f"Could not open stream on '{dev_info['name']}'.")

# ---------------------------------------------------------------------------
# GUI device-selection dialog
# ---------------------------------------------------------------------------

def show_device_dialog_terminal() -> tuple[int | None, bool]:
    """Select audio device via terminal input to prevent Wayland/XWayland crashes."""
    print("\n--- [ Audio Input Devices ] ---")
    devices = sd.query_devices()
    input_devices = []
    
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{len(input_devices)}] Index {i}: {dev['name']} (Max Ch: {dev['max_input_channels']})")
            input_devices.append(i)
            
    if not input_devices:
        print("[!] No input devices found.")
        return None, False
        
    try:
        selection = input("\nSelect device index (or press Enter for default): ").strip()
        if selection == "":
            return None, True
        
        idx = int(selection)
        if 0 <= idx < len(input_devices):
            return input_devices[idx], True
        else:
            print("[!] Invalid selection.")
            return None, False
    except (ValueError, KeyboardInterrupt):
        return None, False

def show_device_dialog(parent=None) -> tuple[int | None, bool]:
    """
    Show a modal dialog listing all input devices.
    Returns (device_index, confirmed).
    If the user cancels, confirmed is False.
    """
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Select Input Device")
    win.resizable(False, False)
    win.configure(bg="#1a1a1a")

    # Center on screen
    win.update_idletasks()
    w, h = 520, 340
    x = (win.winfo_screenwidth()  - w) // 2
    y = (win.winfo_screenheight() - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(win, text="Select Audio Input Device",
             bg="#1a1a1a", fg="#cccccc",
             font=("Helvetica", 13, "bold")).pack(pady=(16, 6))

    tk.Label(win, text="★  = stereo (2+ channels)",
             bg="#1a1a1a", fg="#666666",
             font=("Helvetica", 9)).pack()

    frame = tk.Frame(win, bg="#1a1a1a")
    frame.pack(fill="both", expand=True, padx=16, pady=10)

    style = ttk.Style(win)
    style.theme_use("clam")
    style.configure("Dev.Treeview",
                    background="#111111", foreground="#cccccc",
                    fieldbackground="#111111", rowheight=22,
                    font=("Courier", 9))
    style.configure("Dev.Treeview.Heading",
                    background="#222222", foreground="#aaaaaa",
                    font=("Helvetica", 9, "bold"))
    style.map("Dev.Treeview", background=[("selected", "#2a4a3a")])

    cols = ("idx", "name", "ch", "sr")
    tree = ttk.Treeview(frame, columns=cols, show="headings",
                        style="Dev.Treeview", selectmode="browse")
    tree.heading("idx",  text="#")
    tree.heading("name", text="Device Name")
    tree.heading("ch",   text="Ch")
    tree.heading("sr",   text="Sample Rate")
    tree.column("idx",  width=32,  anchor="center", stretch=False)
    tree.column("name", width=310, anchor="w")
    tree.column("ch",   width=40,  anchor="center", stretch=False)
    tree.column("sr",   width=90,  anchor="center", stretch=False)

    sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    devices = list_input_devices()
    default_idx = sd.default.device
    if isinstance(default_idx, (list, tuple)):
        default_idx = default_idx[0]

    first_iid = None
    for dev in devices:
        stereo = "★" if dev["max_input_channels"] >= 2 else " "
        sr     = int(dev["default_samplerate"])
        iid    = tree.insert("", "end",
                             values=(dev["index"],
                                     f"{stereo} {dev['name']}",
                                     dev["max_input_channels"],
                                     f"{sr} Hz"))
        if first_iid is None:
            first_iid = iid
        if dev["index"] == default_idx:
            tree.selection_set(iid)
            tree.see(iid)

    if not tree.selection() and first_iid:
        tree.selection_set(first_iid)

    result = {"index": None, "ok": False}

    def _confirm(*_):
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        result["index"] = int(vals[0])
        result["ok"]    = True
        win.destroy()

    def _cancel(*_):
        result["ok"] = False
        win.destroy()

    tree.bind("<Double-1>", _confirm)
    win.bind("<Return>", _confirm)
    win.bind("<Escape>", _cancel)

    btn_frame = tk.Frame(win, bg="#1a1a1a")
    btn_frame.pack(pady=(0, 14))

    btn_ok = tk.Button(btn_frame, text="Connect",
                       command=_confirm, width=12,
                       bg="#1e4a30", fg="#cccccc",
                       activebackground="#2a6a40",
                       relief="flat", cursor="hand2")
    btn_ok.pack(side="left", padx=6)

    btn_cancel = tk.Button(btn_frame, text="Cancel",
                           command=_cancel, width=12,
                           bg="#3a1a1a", fg="#cccccc",
                           activebackground="#5a2a2a",
                           relief="flat", cursor="hand2")
    btn_cancel.pack(side="left", padx=6)

    if parent:
        win.transient(parent)
    win.grab_set()
    win.wait_window()

    return result["index"], result["ok"]

# ---------------------------------------------------------------------------
# Parameters dialog
# ---------------------------------------------------------------------------

def show_params_dialog(parent, current: dict) -> dict | None:
    """
    Let the user edit runtime parameters.
    Returns updated dict, or None if cancelled.
    """
    win = tk.Toplevel(parent)
    win.title("Edit Parameters")
    win.resizable(False, False)
    win.configure(bg="#1a1a1a")
    win.update_idletasks()
    w, h = 380, 320
    x = (win.winfo_screenwidth()  - w) // 2
    y = (win.winfo_screenheight() - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")

    fields = [
        ("Display samples (overlay)",  "disp_samples",     current["disp_samples"]),
        ("Envelope window (samples)",  "env_disp_samples", current["env_disp_samples"]),
        ("FFT size",                   "fft_size",         current["fft_size"]),
        ("FFT smoothing (0–1)",        "smoothing",        current["smoothing"]),
        ("Env LPF cutoff (Hz)",        "env_cutoff_hz",    current["env_cutoff_hz"]),
        ("Refresh interval (ms)",      "refresh_ms",       current["refresh_ms"]),
        ("Tuner min freq (Hz)",        "tune_fmin",        current["tune_fmin"]),
        ("Tuner max freq (Hz)",        "tune_fmax",        current["tune_fmax"]),
    ]

    entries: dict[str, tk.StringVar] = {}

    for row, (label, key, val) in enumerate(fields):
        tk.Label(win, text=label, bg="#1a1a1a", fg="#aaaaaa",
                 font=("Helvetica", 9), anchor="w").grid(
                 row=row, column=0, padx=16, pady=4, sticky="w")
        var = tk.StringVar(value=str(val))
        entries[key] = var
        tk.Entry(win, textvariable=var, bg="#222222", fg="#eeeeee",
                 insertbackground="white",
                 relief="flat", font=("Courier", 9), width=12).grid(
                 row=row, column=1, padx=16, pady=4)

    result = {"ok": False, "values": None}

    def _confirm():
        try:
            values = {
                "disp_samples":     int(entries["disp_samples"].get()),
                "env_disp_samples": int(entries["env_disp_samples"].get()),
                "fft_size":         int(entries["fft_size"].get()),
                "smoothing":        float(entries["smoothing"].get()),
                "env_cutoff_hz":    float(entries["env_cutoff_hz"].get()),
                "refresh_ms":       int(entries["refresh_ms"].get()),
                "tune_fmin":        float(entries["tune_fmin"].get()),
                "tune_fmax":        float(entries["tune_fmax"].get()),
            }
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Please enter valid numbers.", parent=win)
            return
        result["ok"]     = True
        result["values"] = values
        win.destroy()

    def _cancel():
        win.destroy()

    btn_frame = tk.Frame(win, bg="#1a1a1a")
    btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=14)
    tk.Button(btn_frame, text="Apply & Restart",
              command=_confirm, width=16,
              bg="#1e4a30", fg="#cccccc", activebackground="#2a6a40",
              relief="flat", cursor="hand2").pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel",
              command=_cancel, width=10,
              bg="#3a1a1a", fg="#cccccc", activebackground="#5a2a2a",
              relief="flat", cursor="hand2").pack(side="left", padx=6)

    win.transient(parent)
    win.grab_set()
    win.wait_window()
    return result["values"] if result["ok"] else None

# ---------------------------------------------------------------------------
# Audio callback  (runs in sounddevice's real-time thread)
# ---------------------------------------------------------------------------

def audio_callback(indata, frames, time, status) -> None:
    if status:
        print(f"  [!] {status}", file=sys.stderr)

    bs, window = state["block_size"], state["window"]
    ch0 = indata[:, 0].astype(np.float32)
    ch1 = indata[:, 1 if indata.shape[1] > 1 else 0].astype(np.float32)

    new_specs = []
    for sig in (ch0, ch1):
        padded     = np.zeros(FFT_SIZE, dtype=np.float32)
        n          = min(len(sig), bs)
        padded[:n] = sig[:n] * window[:n]
        spectrum   = np.abs(np.fft.rfft(padded)) / bs
        new_specs.append(20.0 * np.log10(spectrum + 1e-12))

    with lock:
        state["ring"][0].extend(ch0)
        state["ring"][1].extend(ch1)
        sb    = state["spec_buf"]
        sb[0] = SMOOTHING * sb[0] + (1.0 - SMOOTHING) * new_specs[0]
        sb[1] = SMOOTHING * sb[1] + (1.0 - SMOOTHING) * new_specs[1]

# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def find_trigger(sig: np.ndarray, level: float, edge: str) -> int | None:
    """
    Find the most RECENT crossing of `level` within the last TRIG_LOOKBACK
    samples. Searching from the recent past minimises display latency.
    Both `sig` and `level` are in raw normalised units (±1.0 full scale).
    Returns the sample index to start the display window, or None.
    """
    end    = len(sig) - DISP_SAMPLES
    start  = max(0, end - TRIG_LOOKBACK)
    search = sig[start:end]
    if edge == "rise":
        idx = np.where((search[:-1] < level) & (search[1:] >= level))[0]
    else:
        idx = np.where((search[:-1] > level) & (search[1:] <= level))[0]
    return start + int(idx[-1]) + 1 if len(idx) else None


def compute_envelope(sig: np.ndarray) -> np.ndarray:
    """Full-wave rectify then low-pass filter → amplitude envelope."""
    return sosfilt(state["sos_env"], np.abs(sig))

# ---------------------------------------------------------------------------
# Pitch detector  (MPM autocorrelation + parabolic interpolation)
# ---------------------------------------------------------------------------

def detect_pitch(sig: np.ndarray, sr: int) -> tuple[float | None, float]:
    """
    Estimate fundamental frequency via normalised autocorrelation.
    Returns (frequency_hz, confidence) where confidence ∈ [0, 1].
    Returns (None, 0.0) when silent or pitch is ambiguous.
    """
    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms < TUNE_SILENCE_RMS:
        return None, 0.0

    # Remove DC and apply window before ACF
    s     = (sig - np.mean(sig)).astype(np.float64)
    s    *= np.hanning(len(s))
    n     = len(s)

    # Padded FFT for linear (non-circular) autocorrelation
    fft_n = 1 << (2 * n - 1).bit_length()
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

    # Sub-sample accuracy via parabolic interpolation
    i = peak_i + lag_min
    if 0 < i < n - 1:
        a, b, c = acf[i - 1], acf[i], acf[i + 1]
        denom   = a - 2.0 * b + c
        offset  = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        lag     = i + offset
    else:
        lag = float(i)

    return float(sr / lag), peak_v


def freq_to_note(freq: float | None, a4: float = 440.0
                 ) -> tuple[str | None, int, float]:
    """
    Convert a frequency to (note_name, octave, cents_deviation).
    Returns (None, 0, 0.0) for invalid input.
    """
    if freq is None or freq <= 0:
        return None, 0, 0.0
    semis  = 12.0 * np.log2(freq / a4)
    midi   = semis + 69.0
    midi_r = round(midi)
    cents  = (midi - midi_r) * 100.0
    note   = NOTE_NAMES[int(midi_r) % 12]
    octave = int(midi_r) // 12 - 1
    return note, octave, float(cents)

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
BG = "#0d0d0d"
GR = "#1c1c1c"
ZL = "#2e2e2e"
C1 = "#2dcca0"   # Ch1 — teal
C2 = "#f07b50"   # Ch2 — orange
CD = "#c084fc"   # delta FFT — purple
CL = "#facc15"   # envelope — yellow
CT = "#ff4466"   # trigger — red


def _style_ax(ax, title: str, color: str) -> None:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GR)
    ax.tick_params(colors="#555555", labelsize=8)
    ax.set_title(title, color=color, fontsize=9, pad=4)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")


def _fill_verts(t: np.ndarray, y: np.ndarray, base: float = 0.0) -> np.ndarray:
    """Build polygon vertices for a filled waveform area."""
    xs = np.concatenate([t, t[::-1]])
    ys = np.concatenate([y, np.full(len(y), base)])
    return np.column_stack([xs, ys])


def _needle_color(cents: float) -> str:
    ac = abs(cents)
    if ac <=  5: return "#22dd55"   # green  — in tune
    if ac <= 15: return "#aadd22"   # yellow-green
    if ac <= 30: return "#ffbb00"   # amber
    return "#ff4422"                # red — out of tune


def _volt_grid_step(volt_range: int) -> float:
    if volt_range <= 1: return 0.5
    if volt_range <= 5: return 1.0
    return 2.0


def _apply_volt_grid(ax, volt_range: int) -> None:
    ax.set_ylim(-volt_range, volt_range)
    ax.yaxis.set_major_locator(MultipleLocator(_volt_grid_step(volt_range)))


def _apply_env_grid(ax, volt_range: int) -> None:
    ax.set_ylim(-volt_range * 0.04, volt_range * 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(_volt_grid_step(volt_range)))

# ---------------------------------------------------------------------------
# Tuner panel
# ---------------------------------------------------------------------------

def _build_tuner(fig, rect: list, ch_label: str, ch_color: str) -> dict:
    """Build a slim tuner panel. Returns dict of dynamic artists."""
    ax = fig.add_axes(rect)
    ax.set_facecolor("#080808")
    ax.set_xlim(-58, 58)
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_edgecolor("#252525")
        sp.set_linewidth(0.7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Static background colour zones (outer → inner)
    for x0, x1, fc in [
        (-55,  55, "#180a0a"),   # out of tune  (dark red)
        (-28,  28, "#161408"),   # close        (dark amber)
        ( -8,   8, "#0a1a0d"),   # near         (dark green)
        ( -3,   3, "#0c2210"),   # in tune      (green)
    ]:
        ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 1.0,
                                facecolor=fc, edgecolor="none", zorder=1))

    # Scale tick marks
    for c in [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]:
        is_major = (c % 10 == 0)
        h_frac   = 0.65 if is_major else 0.38
        lw       = 1.2  if c == 0  else (0.8 if is_major else 0.45)
        col      = "#505050" if c == 0 else ("#303030" if is_major else "#1e1e1e")
        y0_ = (1.0 - h_frac) / 2
        y1_ = y0_ + h_frac
        ax.plot([c, c], [y0_, y1_], color=col, linewidth=lw,
                solid_capstyle="butt", zorder=2)

    # Scale labels
    for c in [-50, 0, 50]:
        lbl = f"{c:+d}¢" if c != 0 else "0¢"
        ax.text(c, 0.04, lbl, color="#2a2a2a", fontsize=5.5,
                ha="center", va="bottom", zorder=3)

    # Channel header
    ax.text(0, 0.97, f"TUNER  {ch_label}",
            color=ch_color, fontsize=7, ha="center", va="top",
            fontfamily="monospace", fontweight="bold", zorder=3)

    # Dynamic artists
    note_txt  = ax.text(-55, 0.52, "—", color="#2a2a2a", fontsize=17,
                        va="center", ha="left",
                        fontfamily="monospace", fontweight="bold", zorder=5)
    hz_txt    = ax.text(-55, 0.08, "", color="#282828", fontsize=6.5,
                        va="bottom", ha="left", fontfamily="monospace", zorder=5)
    cents_txt = ax.text(55, 0.52, "", color="#2a2a2a", fontsize=12,
                        va="center", ha="right", fontfamily="monospace", zorder=5)
    needle,   = ax.plot([0, 0], [0.08, 0.92], color="#151515",
                        linewidth=3.0, solid_capstyle="round", zorder=6)
    dot,      = ax.plot([0], [0.92], "o", color="#151515",
                        markersize=5, zorder=7)

    return dict(note_txt=note_txt, hz_txt=hz_txt,
                cents_txt=cents_txt, needle=needle, dot=dot)


def _update_tuner(t: dict, freq: float | None,
                  note: str | None, octave: int, cents: float) -> None:
    if note is None:
        t["note_txt"].set_text("—");   t["note_txt"].set_color("#2a2a2a")
        t["hz_txt"].set_text("");       t["cents_txt"].set_text("")
        t["needle"].set_color("#101010"); t["dot"].set_color("#101010")
        t["needle"].set_xdata([0, 0]);  t["dot"].set_xdata([0])
    else:
        nc = _needle_color(cents)
        t["note_txt"].set_text(f"{note}{octave}"); t["note_txt"].set_color(nc)
        t["hz_txt"].set_text(f"{freq:.2f} Hz")
        sign = "+" if cents >= 0 else ""
        t["cents_txt"].set_text(f"{sign}{cents:.1f}¢")
        t["cents_txt"].set_color(nc)
        cx = float(np.clip(cents, -50, 50))
        t["needle"].set_xdata([cx, cx]); t["needle"].set_color(nc)
        t["dot"].set_xdata([cx]);        t["dot"].set_color(nc)

# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(sr: int) -> tuple:
    """
    Build the matplotlib figure with all panels and controls.
    Returns (fig, art, fmask, t_ms).
    """
    freqs    = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    fmask    = freqs >= 20
    t_ms     = np.linspace(0, DISP_SAMPLES / sr * 1000.0, DISP_SAMPLES)
    t_env_ms = np.linspace(0, ENV_DISP_SAMPLES / sr * 1000.0, ENV_DISP_SAMPLES)

    # Vertical layout (figure-normalised coordinates, bottom = 0):
    #   0.01 – 0.11   tuner panels     (h = 0.10)
    #   0.12 – 0.19   controls row     (trigger slider + buttons)
    #   0.20 – 0.958  data panels      (gridspec)
    TUNE_B = 0.01;  TUNE_H = 0.10
    CTRL_B = 0.12;  CTRL_H = 0.022
    GRID_B = 0.20;  GRID_T = 0.958

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.suptitle(
        f"{APP_NAME}  —  Eurorack  [{sr // 1000} kHz]",
        color="#cccccc", fontsize=11, y=0.997)

    gs = fig.add_gridspec(2, 2,
                          height_ratios=[1, 1],
                          hspace=0.44, wspace=0.30,
                          left=0.06, right=0.98,
                          top=GRID_T, bottom=GRID_B)

    # ── [0,0]  Overlay ───────────────────────────────────────────
    ax_ov = fig.add_subplot(gs[0, 0])
    _style_ax(ax_ov, "Ch1 + Ch2 — Overlay  (raw signal + DC offset)", C1)
    ax_ov.set_xlim(0, t_ms[-1])
    ax_ov.set_xlabel("Time (ms)")
    ax_ov.set_ylabel("Amplitude")
    ax_ov.grid(True, color=GR, linewidth=0.5, linestyle="-", zorder=0)
    ax_ov.axhline(0, color=ZL, linewidth=0.9, linestyle="--", zorder=1)
    _apply_volt_grid(ax_ov, 1)

    line_ov1, = ax_ov.plot(t_ms, np.zeros(DISP_SAMPLES),
                            color=C1, linewidth=1.3, zorder=3, label="Ch1")
    line_ov2, = ax_ov.plot(t_ms, np.zeros(DISP_SAMPLES),
                            color=C2, linewidth=1.3, zorder=3,
                            label="Ch2", alpha=0.85)
    trig_hline = ax_ov.axhline(0, color=CT, linewidth=0.9,
                                linestyle=":", zorder=4, alpha=0.9)
    trig_txt   = ax_ov.text(t_ms[-1] * 0.01, 0.0, " ▶",
                             color=CT, fontsize=8, va="center", zorder=5)
    ax_ov.legend(loc="upper right", fontsize=7,
                 facecolor="#111111", edgecolor=GR, labelcolor="#aaaaaa")

    dc_hline1 = ax_ov.axhline(0, color=C1, linewidth=0.8,
                               linestyle="--", zorder=4, alpha=0.55)
    dc_hline2 = ax_ov.axhline(0, color=C2, linewidth=0.8,
                               linestyle="--", zorder=4, alpha=0.55)
    dc_txt1 = ax_ov.text(t_ms[-1] * 0.98, 0.0, "", color=C1, fontsize=7,
                          va="center", ha="right", fontfamily="monospace", zorder=6)
    dc_txt2 = ax_ov.text(t_ms[-1] * 0.98, 0.0, "", color=C2, fontsize=7,
                          va="center", ha="right", fontfamily="monospace", zorder=6)
    poly_ov1 = PolyCollection([_fill_verts(t_ms, np.zeros(DISP_SAMPLES))],
                               facecolors=[C1], alpha=0.07, zorder=2)
    poly_ov2 = PolyCollection([_fill_verts(t_ms, np.zeros(DISP_SAMPLES))],
                               facecolors=[C2], alpha=0.06, zorder=2)
    ax_ov.add_collection(poly_ov1)
    ax_ov.add_collection(poly_ov2)

    # ── [1,0]  Delta FFT ─────────────────────────────────────────
    ax_df = fig.add_subplot(gs[1, 0])
    _style_ax(ax_df, "FFT Ch1 − FFT Ch2  (dB)", CD)
    ax_df.set_xscale("log"); ax_df.set_xlim(20, sr / 2); ax_df.set_ylim(-40, 40)
    ax_df.set_xlabel("Frequency (Hz)"); ax_df.set_ylabel("Difference (dB)")
    for xg in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if xg < sr / 2:
            ax_df.axvline(xg, color=GR, linewidth=0.5, zorder=1)
    ax_df.axhline( 10, color=GR, linewidth=0.5, linestyle="--")
    ax_df.axhline(-10, color=GR, linewidth=0.5, linestyle="--")
    ax_df.axhline(  0, color=ZL, linewidth=0.9, linestyle="--")
    line_df, = ax_df.plot(freqs[fmask], np.zeros(fmask.sum()),
                           color=CD, linewidth=1.2, zorder=3)
    poly_df_pos = PolyCollection([], facecolors=[C1], alpha=0.15, zorder=2)
    poly_df_neg = PolyCollection([], facecolors=[C2], alpha=0.15, zorder=2)
    ax_df.add_collection(poly_df_pos); ax_df.add_collection(poly_df_neg)

    # ── [0,1]  FFT spectrum ───────────────────────────────────────
    ax_fft = fig.add_subplot(gs[0, 1])
    _style_ax(ax_fft, "FFT Spectrum — Ch1 + Ch2", C1)
    ax_fft.set_xscale("log"); ax_fft.set_xlim(20, sr / 2); ax_fft.set_ylim(-90, 6)
    ax_fft.set_xlabel("Frequency (Hz)"); ax_fft.set_ylabel("Level (dB)")
    for xg in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if xg < sr / 2:
            ax_fft.axvline(xg, color=GR, linewidth=0.5, zorder=1)
    ax_fft.axhline(-60, color=GR, linewidth=0.5, linestyle="--")
    ax_fft.axhline(-30, color=GR, linewidth=0.5, linestyle="--")
    ax_fft.axhline(  0, color=ZL, linewidth=0.8, linestyle="--")
    line_fft1, = ax_fft.plot(freqs[fmask], np.full(fmask.sum(), -90.0),
                              color=C1, linewidth=1.2, zorder=3, label="Ch1")
    line_fft2, = ax_fft.plot(freqs[fmask], np.full(fmask.sum(), -90.0),
                              color=C2, linewidth=1.2, zorder=3,
                              label="Ch2", alpha=0.85)
    ax_fft.legend(loc="upper right", fontsize=7,
                  facecolor="#111111", edgecolor=GR, labelcolor="#aaaaaa")

    # ── [1,1]  Envelope follower ──────────────────────────────────
    env_ms_lbl = f"{ENV_DISP_SAMPLES / sr * 1000:.0f} ms"
    ax_env = fig.add_subplot(gs[1, 1])
    _style_ax(ax_env, f"Envelope Follower — Ch1 + Ch2  [{env_ms_lbl}]", CL)
    ax_env.set_xlim(0, t_env_ms[-1])
    ax_env.set_xlabel("Time (ms)"); ax_env.set_ylabel("Amplitude")
    ax_env.grid(True, color=GR, linewidth=0.5, linestyle="-", zorder=0)
    ax_env.axhline(0, color=ZL, linewidth=0.8, linestyle="--", zorder=1)
    _apply_env_grid(ax_env, 1)

    line_rect1, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C1, linewidth=0.7, zorder=2, alpha=0.18)
    line_rect2, = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C2, linewidth=0.7, zorder=2, alpha=0.16)
    line_env1,  = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C1, linewidth=2.0, zorder=4, label="Env Ch1")
    line_env2,  = ax_env.plot(t_env_ms, np.zeros(ENV_DISP_SAMPLES),
                               color=C2, linewidth=2.0, zorder=4,
                               label="Env Ch2", alpha=0.85)
    poly_env1 = PolyCollection([_fill_verts(t_env_ms, np.zeros(ENV_DISP_SAMPLES))],
                                facecolors=[C1], alpha=0.13, zorder=3)
    poly_env2 = PolyCollection([_fill_verts(t_env_ms, np.zeros(ENV_DISP_SAMPLES))],
                                facecolors=[C2], alpha=0.11, zorder=3)
    ax_env.add_collection(poly_env1); ax_env.add_collection(poly_env2)
    env_peak1 = ax_env.axhline(0, color=C1, linewidth=1.3,
                                linestyle="--", zorder=5, alpha=0.75)
    env_peak2 = ax_env.axhline(0, color=C2, linewidth=1.3,
                                linestyle="--", zorder=5, alpha=0.75)
    env_txt1 = ax_env.text(t_env_ms[-1] * 0.98, 0.0, "",
                            color=C1, fontsize=7, va="bottom", ha="right",
                            fontfamily="monospace", zorder=6)
    env_txt2 = ax_env.text(t_env_ms[-1] * 0.98, 0.0, "",
                            color=C2, fontsize=7, va="top", ha="right",
                            fontfamily="monospace", zorder=6)
    ax_env.legend(handles=[line_env1, line_env2], loc="upper left", fontsize=7,
                  facecolor="#111111", edgecolor=GR, labelcolor="#aaaaaa")

    # ── Controls ─────────────────────────────────────────────────
    # Trigger level slider  (raw units, ±1.0)
    ax_sl = fig.add_axes([0.06, CTRL_B, 0.44, CTRL_H])
    ax_sl.set_facecolor("#111111")
    slider = Slider(ax_sl, "Trig ", -1.0, 1.0, valinit=0.0, color=CT)
    slider.label.set_color("#666666"); slider.label.set_fontsize(8)
    slider.valtext.set_color(CT);      slider.valtext.set_fontsize(8)

    def _on_slider(val: float) -> None:
        ui["trig_level"] = val
        trig_hline.set_ydata([val, val])
        trig_txt.set_position((t_ms[-1] * 0.01, val))
    slider.on_changed(_on_slider)

    # Rise / Fall edge button
    ax_be = fig.add_axes([0.52, CTRL_B - 0.005, 0.09, CTRL_H + 0.012])
    btn_edge = Button(ax_be, "▲ Rise", color="#141414", hovercolor="#242424")
    btn_edge.label.set_color(CT); btn_edge.label.set_fontsize(8)
    btn_edge.label.set_fontfamily("monospace")

    def _toggle_edge(_) -> None:
        if ui["trig_edge"] == "rise":
            ui["trig_edge"] = "fall"; btn_edge.label.set_text("▼ Fall")
        else:
            ui["trig_edge"] = "rise"; btn_edge.label.set_text("▲ Rise")
    btn_edge.on_clicked(_toggle_edge)

    # Voltage range buttons
    btn_volt_refs = []
    VBTN_Y = CTRL_B - 0.005;  VBTN_H = CTRL_H + 0.012
    vbtn_x = [0.63, 0.73, 0.83]

    def _make_volt_cb(scale: int, vrange: int, btn_idx: int):
        def _cb(_) -> None:
            ui["volt_scale"] = scale
            ui["volt_range"] = vrange
            # Only axis limits change — signal is never rescaled
            _apply_volt_grid(ax_ov, vrange)
            _apply_env_grid(ax_env, vrange)
            for i, b in enumerate(btn_volt_refs):
                active = (i == btn_idx)
                b.ax.set_facecolor("#2a1a00" if active else "#141414")
            fig.canvas.draw_idle()
        return _cb

    for bi, (lbl, sc, vr) in enumerate(VOLT_RANGES):
        ax_vb = fig.add_axes([vbtn_x[bi], VBTN_Y, 0.09, VBTN_H])
        b = Button(ax_vb, lbl, color="#141414", hovercolor="#242424")
        b.label.set_color("#ffcc44"); b.label.set_fontsize(7.5)
        b.label.set_fontfamily("monospace")
        btn_volt_refs.append(b)

    for bi, (_, sc, vr) in enumerate(VOLT_RANGES):
        btn_volt_refs[bi].on_clicked(_make_volt_cb(sc, vr, bi))

    btn_volt_refs[0].ax.set_facecolor("#2a1a00")  # default: [-1, 1] active

    # Tuner panels
    tune1 = _build_tuner(fig, [0.06, TUNE_B, 0.43, TUNE_H], "Ch1", C1)
    tune2 = _build_tuner(fig, [0.51, TUNE_B, 0.43, TUNE_H], "Ch2", C2)

    art = dict(
        ax_ov=ax_ov,         ax_env=ax_env,
        line_ov1=line_ov1,   line_ov2=line_ov2,
        poly_ov1=poly_ov1,   poly_ov2=poly_ov2,
        trig_hline=trig_hline, trig_txt=trig_txt,
        dc_hline1=dc_hline1, dc_hline2=dc_hline2,
        dc_txt1=dc_txt1,     dc_txt2=dc_txt2,
        line_df=line_df,
        poly_df_pos=poly_df_pos, poly_df_neg=poly_df_neg,
        line_fft1=line_fft1, line_fft2=line_fft2,
        line_rect1=line_rect1, line_rect2=line_rect2,
        line_env1=line_env1,   line_env2=line_env2,
        poly_env1=poly_env1,   poly_env2=poly_env2,
        env_peak1=env_peak1,   env_peak2=env_peak2,
        env_txt1=env_txt1,     env_txt2=env_txt2,
        tune1=tune1,           tune2=tune2,
        freqs_masked=freqs[fmask],
        t_env_ms=t_env_ms,
    )
    slider=slider,
    btn_edge=btn_edge,
    btn_volt_refs=btn_volt_refs,
    fig._gui_widgets = [slider, btn_edge, btn_volt_refs]
    return fig, art, fmask, t_ms

# ---------------------------------------------------------------------------
# Animation tick  (called by FuncAnimation every REFRESH_MS)
# ---------------------------------------------------------------------------

def make_tick(art: dict, fmask: np.ndarray, t_ms: np.ndarray):
    """Return the animation update function (closure over shared art dict)."""
    sr = state["sample_rate"]

    peak       = [0.0, 0.0]
    peak_tmr   = [0, 0]
    pitch_freq = [None, None]   # EMA-smoothed frequency per channel
    pitch_miss = [0, 0]         # consecutive frames without detection

    def _dc_label(v: float) -> str:
        arrow = "▲" if v > 0.01 else ("▼" if v < -0.01 else "●")
        return f"{arrow} DC {v:+.3f}"

    def _fft_fill(x, y):
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([y, np.zeros(len(y))])
        return np.column_stack([xs, ys])

    def _smooth_pitch(ch: int, raw_freq: float | None, conf: float
                      ) -> float | None:
        if raw_freq is not None and conf >= TUNE_CONF_THR:
            pitch_miss[ch] = 0
            if pitch_freq[ch] is None:
                pitch_freq[ch] = raw_freq
            else:
                alpha = min(TUNE_ALPHA * (0.5 + conf), 0.9)
                pitch_freq[ch] = (alpha * raw_freq
                                  + (1.0 - alpha) * pitch_freq[ch])
        else:
            pitch_miss[ch] += 1
            if pitch_miss[ch] > TUNE_FREEZE_FR:
                pitch_freq[ch] = None
        return pitch_freq[ch]

    def tick(frame: int) -> None:
        # Read shared buffers atomically
        with lock:
            raw0 = np.array(state["ring"][0], dtype=np.float32)
            raw1 = np.array(state["ring"][1], dtype=np.float32)
            db0  = state["spec_buf"][0].copy()
            db1  = state["spec_buf"][1].copy()

        # ── Overlay ──────────────────────────────────────────
        level = ui["trig_level"]   # raw units (±1.0)
        edge  = ui["trig_edge"]

        idx = find_trigger(raw0, level, edge)
        idx = idx if idx is not None else len(raw0) - DISP_SAMPLES
        idx = max(0, min(idx, len(raw0) - DISP_SAMPLES))

        y0 = raw0[idx: idx + DISP_SAMPLES]
        y1 = raw1[idx: idx + DISP_SAMPLES]
        n  = min(len(y0), len(y1), DISP_SAMPLES)
        y0, y1, tm = y0[:n], y1[:n], t_ms[:n]

        art["line_ov1"].set_data(tm, y0)
        art["line_ov2"].set_data(tm, y1)
        art["poly_ov1"].set_verts([_fill_verts(tm, y0)])
        art["poly_ov2"].set_verts([_fill_verts(tm, y1)])

        # DC offset lines — mean of the visible window
        dc0 = float(np.mean(y0))
        dc1 = float(np.mean(y1))
        art["dc_hline1"].set_ydata([dc0, dc0])
        art["dc_hline2"].set_ydata([dc1, dc1])
        art["dc_txt1"].set_position((tm[-1] * 0.98, dc0))
        art["dc_txt2"].set_position((tm[-1] * 0.98, dc1))
        art["dc_txt1"].set_text(_dc_label(dc0))
        art["dc_txt2"].set_text(_dc_label(dc1))

        # ── Delta FFT ────────────────────────────────────────
        diff = db0[fmask] - db1[fmask]
        fx   = art["freqs_masked"]
        art["line_df"].set_data(fx, diff)
        art["poly_df_pos"].set_verts([_fft_fill(fx, np.maximum(diff, 0))])
        art["poly_df_neg"].set_verts([_fft_fill(fx, np.minimum(diff, 0))])

        # ── FFT spectrum ─────────────────────────────────────
        art["line_fft1"].set_ydata(db0[fmask])
        art["line_fft2"].set_ydata(db1[fmask])

        # ── Envelope follower ─────────────────────────────────
        t_env_ms = art["t_env_ms"]
        ne  = min(len(raw0), len(raw1), ENV_DISP_SAMPLES)
        r0, r1 = raw0[-ne:], raw1[-ne:]
        e0 = compute_envelope(r0)
        e1 = compute_envelope(r1)

        art["line_rect1"].set_data(t_env_ms[:ne], np.abs(r0))
        art["line_rect2"].set_data(t_env_ms[:ne], np.abs(r1))
        art["line_env1"].set_data(t_env_ms[:ne], e0)
        art["line_env2"].set_data(t_env_ms[:ne], e1)
        art["poly_env1"].set_verts([_fill_verts(t_env_ms[:ne], e0)])
        art["poly_env2"].set_verts([_fill_verts(t_env_ms[:ne], e1)])

        for ch, (r, e, ep, et) in enumerate(zip(
            (np.abs(r0), np.abs(r1)), (e0, e1),
            (art["env_peak1"], art["env_peak2"]),
            (art["env_txt1"],  art["env_txt2"]),
        )):
            pk_new = float(np.max(e))
            if pk_new >= peak[ch]:
                peak[ch] = pk_new; peak_tmr[ch] = PEAK_HOLD_FRAMES
            else:
                if peak_tmr[ch] > 0:
                    peak_tmr[ch] -= 1
                else:
                    peak[ch] *= PEAK_DECAY
            ep.set_ydata([peak[ch], peak[ch]])
            rms = float(np.sqrt(np.mean(r ** 2)))
            et.set_text(f"pk {peak[ch]:.3f}  rms {rms:.3f}")
            et.set_position((t_env_ms[ne - 1] * 0.98, peak[ch]))

        # ── Tuners ───────────────────────────────────────────
        for ch, (raw, t_art) in enumerate(zip(
            (raw0, raw1), (art["tune1"], art["tune2"])
        )):
            nt = min(len(raw), TUNE_SAMPLES)
            raw_freq, conf = detect_pitch(raw[-nt:].astype(np.float64), sr)
            freq_s = _smooth_pitch(ch, raw_freq, conf)
            note, octave, cents = freq_to_note(freq_s)
            _update_tuner(t_art, freq_s, note, octave, cents)

    return tick

# ---------------------------------------------------------------------------
# Native menu bar  (attached to the Tk root window)
# ---------------------------------------------------------------------------

def attach_menubar(root: tk.Tk, fig, stream_holder: list,
                   device_index_holder: list) -> None:
    """
    Build and attach the native OS menu bar to the Tk window.
    stream_holder[0] and device_index_holder[0] hold mutable references
    that can be swapped when the user changes input.
    """
    menubar = tk.Menu(root, bg="#1a1a1a", fg="#cccccc",
                      activebackground="#2a4a3a", activeforeground="#ffffff",
                      relief="flat")

    # ── File ────────────────────────────────────────────────────
    file_menu = tk.Menu(menubar, tearoff=0,
                        bg="#1a1a1a", fg="#cccccc",
                        activebackground="#2a4a3a")
    file_menu.add_command(
        label="Exit",
        accelerator="Ctrl+Q",
        command=lambda: _quit(root))
    root.bind_all("<Control-q>", lambda _: _quit(root))
    menubar.add_cascade(label="File", menu=file_menu)

    # ── Edit ────────────────────────────────────────────────────
    edit_menu = tk.Menu(menubar, tearoff=0,
                        bg="#1a1a1a", fg="#cccccc",
                        activebackground="#2a4a3a")
    edit_menu.add_command(
        label="Parameters…",
        accelerator="Ctrl+,",
        command=lambda: _edit_params(root))
    root.bind_all("<Control-comma>", lambda _: _edit_params(root))
    menubar.add_cascade(label="Edit", menu=edit_menu)

    # ── Inputs ──────────────────────────────────────────────────
    inputs_menu = tk.Menu(menubar, tearoff=0,
                          bg="#1a1a1a", fg="#cccccc",
                          activebackground="#2a4a3a")
    inputs_menu.add_command(
        label="Select Device…",
        accelerator="Ctrl+D",
        command=lambda: _change_device(root, stream_holder,
                                       device_index_holder, fig))
    root.bind_all("<Control-d>", lambda _: _change_device(
        root, stream_holder, device_index_holder, fig))
    menubar.add_cascade(label="Inputs", menu=inputs_menu)

    # ── Help ────────────────────────────────────────────────────
    help_menu = tk.Menu(menubar, tearoff=0,
                        bg="#1a1a1a", fg="#cccccc",
                        activebackground="#2a4a3a")
    help_menu.add_command(label="View on GitHub",
                          command=lambda: _open_url(APP_URL))
    help_menu.add_command(label="Report an Issue",
                          command=lambda: _open_url(
                              APP_URL + "/issues"))
    help_menu.add_separator()
    help_menu.add_command(label="Contact",
                          command=lambda: _show_contact(root))
    help_menu.add_separator()
    help_menu.add_command(label="About",
                          command=lambda: _show_about(root))
    menubar.add_cascade(label="Help", menu=help_menu)

    root.config(menu=menubar)


def _quit(root: tk.Tk) -> None:
    plt.close("all")
    try:
        root.quit()
        root.destroy()
    except Exception:
        pass
    sys.exit(0)


def _open_url(url: str) -> None:
    import webbrowser
    webbrowser.open(url)


def _show_about(root: tk.Tk) -> None:
    messagebox.showinfo(
        "About",
        f"{APP_NAME}\n"
        f"Version {APP_VERSION}\n\n"
        f"A real-time dual-channel audio monitor\n"
        f"designed for Eurorack modular synthesis.\n\n"
        f"Author : {APP_AUTHOR}\n"
        f"GitHub : {APP_URL}\n\n"
        f"Python {sys.version.split()[0]}  ·  "
        f"{platform.system()} {platform.release()}",
        parent=root)


def _show_contact(root: tk.Tk) -> None:
    messagebox.showinfo(
        "Contact",
        f"Email   : {APP_EMAIL}\n"
        f"GitHub  : {APP_URL}\n\n"
        f"Feel free to open an issue or send a pull request.",
        parent=root)


def _edit_params(root: tk.Tk) -> None:
    current = dict(
        disp_samples     = DISP_SAMPLES,
        env_disp_samples = ENV_DISP_SAMPLES,
        fft_size         = FFT_SIZE,
        smoothing        = SMOOTHING,
        env_cutoff_hz    = ENV_CUTOFF_HZ,
        refresh_ms       = REFRESH_MS,
        tune_fmin        = TUNE_FMIN,
        tune_fmax        = TUNE_FMAX,
    )
    result = show_params_dialog(root, current)
    if result:
        messagebox.showinfo(
            "Parameters",
            "Changes will take effect after restarting the application.",
            parent=root)


def _change_device(root: tk.Tk, stream_holder: list,
                   device_index_holder: list, fig) -> None:
    """Stop the current stream, let user pick a new device, restart."""
    dev_idx, ok = show_device_dialog(root)
    if not ok:
        return

    # Stop old stream
    old_stream = stream_holder[0]
    if old_stream is not None:
        old_stream.stop()
        old_stream.close()

    try:
        dev_info = sd.query_devices(dev_idx, "input")
        n_ch     = min(2, dev_info["max_input_channels"])
        sr, bs, dev_info = negotiate_stream(dev_idx, n_ch)
        init_state(sr, bs)
        new_stream = sd.InputStream(
            device=dev_idx, samplerate=sr, channels=n_ch,
            blocksize=bs, callback=audio_callback, dtype="float32")
        new_stream.start()
        stream_holder[0]       = new_stream
        device_index_holder[0] = dev_idx
        fig.suptitle(
            f"{APP_NAME}  —  Eurorack  [{sr // 1000} kHz]  "
            f"·  {dev_info['name']}",
            color="#cccccc", fontsize=11, y=0.997)
        fig.canvas.draw_idle()
    except Exception as exc:
        messagebox.showerror("Device Error", str(exc), parent=root)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    is_linux = sys.platform.startswith("linux")

    if is_linux:
        dev_idx, ok = show_device_dialog_terminal()
        if not ok:
            print("Selection canceled. Exiting.")
            sys.exit(0)
        # Inicializamos un root oculto que Matplotlib heredará de forma segura
        root = tk.Tk()
        root.withdraw()
    else:
        # Comportamiento original para Windows/macOS
        root = tk.Tk()
        root.withdraw()
        dev_idx, ok = show_device_dialog(root)
        if not ok:
            root.destroy()
            sys.exit(0)

    # --- Negotiate audio stream ---
    dev_info = sd.query_devices(dev_idx, "input") if dev_idx is not None \
               else sd.query_devices(sd.default.device[0]
                                     if isinstance(sd.default.device, (list, tuple))
                                     else sd.default.device, "input")
    n_ch = min(2, dev_info["max_input_channels"])

    print(f"  Device     : {dev_info['name']}")
    print(f"  Channels   : {n_ch}")

    sr, bs, dev_info = negotiate_stream(dev_idx, n_ch)
    init_state(sr, bs)

    print(f"  Sample rate: {sr} Hz")
    print(f"  Block size : {bs} samples  ({1000 * bs / sr:.1f} ms)")
    if n_ch < 2:
        print("  ⚠  Mono device — Ch2 mirrors Ch1")

    # --- Build matplotlib figure ---
    fig, art, fmask, t_ms = build_figure(sr)
    tick = make_tick(art, fmask, t_ms)

    # FuncAnimation integrates with Tk's event loop — sliders/buttons work.
    # Store reference in fig to prevent garbage collection.
    fig._ani = FuncAnimation(fig, tick,
                              interval=REFRESH_MS,
                              blit=False,
                              cache_frame_data=False)

    # --- Attach native menu bar ---
    stream_holder       = [None]
    device_index_holder = [dev_idx]

    stream = sd.InputStream(
        device=dev_idx, samplerate=sr, channels=n_ch,
        blocksize=bs, callback=audio_callback, dtype="float32")
    stream.start()
    stream_holder[0] = stream

    try:
        root = fig.canvas.manager.window   # underlying Tk root window
        attach_menubar(root, fig, stream_holder, device_index_holder)
        root.protocol("WM_DELETE_WINDOW", lambda: _quit(root))
    except Exception:
        pass   # non-Tk backend — menu not available, but app still works

    plt.show()

    # --- Cleanup ---
    if stream_holder[0] is not None:
        stream_holder[0].stop()
        stream_holder[0].close()

    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
