import os
import sys
import gc
import ctypes
from ctypes import wintypes
import tkinter as tk
import tracemalloc

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gui import DrawingApp


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_private_mb():
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)

        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.PrivateUsage / (1024 * 1024)
    except Exception:
        pass

    # Fallback: tracemalloc only measures Python-managed allocations.
    current, _ = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


def draw_sample_letter(app):
    # Draw a deterministic "A" shape directly on the hidden PIL image.
    w = app.canvas_width
    h = app.canvas_height

    left = int(w * 0.30)
    right = int(w * 0.70)
    top = int(h * 0.20)
    bottom = int(h * 0.80)
    mid_y = int(h * 0.52)

    app.draw.line([(left, bottom), (w // 2, top)], fill=0, width=24)
    app.draw.line([(w // 2, top), (right, bottom)], fill=0, width=24)
    app.draw.line([(left + 20, mid_y), (right - 20, mid_y)], fill=0, width=20)


def main():
    iterations = 300
    report_every = 50
    allowed_growth_mb = 80.0

    if not tracemalloc.is_tracing():
        tracemalloc.start()

    root = tk.Tk()
    root.withdraw()
    app = DrawingApp(root)

    try:
        gc.collect()
        mem_start = get_process_private_mb()
        peak_mem = mem_start

        print(f"[MEM] Start private memory: {mem_start:.2f} MB")

        for i in range(1, iterations + 1):
            app.clear_canvas()
            draw_sample_letter(app)
            app.predict()
            root.update_idletasks()

            if i % report_every == 0:
                gc.collect()
                current = get_process_private_mb()
                peak_mem = max(peak_mem, current)
                print(f"[MEM] Iteration {i:03d}: {current:.2f} MB")

        gc.collect()
        mem_before_shutdown = get_process_private_mb()
        print(f"[MEM] Before shutdown: {mem_before_shutdown:.2f} MB")

        app.on_close()
        gc.collect()
        mem_after_shutdown = get_process_private_mb()
        print(f"[MEM] After shutdown: {mem_after_shutdown:.2f} MB")

        growth_peak = peak_mem - mem_start
        growth_end = mem_after_shutdown - mem_start

        print(f"[MEM] Peak growth: {growth_peak:.2f} MB")
        print(f"[MEM] End growth:  {growth_end:.2f} MB")

        if growth_peak > allowed_growth_mb:
            print("[WARN] Peak memory growth exceeded threshold.")
        else:
            print("[OK] Peak memory growth is within threshold.")

        if growth_end > (allowed_growth_mb / 2):
            print("[WARN] Memory remained high after shutdown.")
        else:
            print("[OK] Memory recovered well after shutdown.")

    finally:
        # Extra safety in case test exits early
        try:
            app.shutdown_resources()
        except Exception:
            pass
        try:
            if root.winfo_exists():
                root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
