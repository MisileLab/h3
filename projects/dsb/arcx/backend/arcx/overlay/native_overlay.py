"""Lightweight native overlay (no Overwolf required).

Creates a frameless, always-on-top Tk window that polls the local API and
shows EV values, recommendation, and lets the user change risk profile.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from typing import Optional

import httpx

from arcx.config import config


class NativeOverlayApp:
    """Simple native overlay backed by the ArcX API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        poll_interval: float = 0.5,
    ):
        self.base_url = base_url or f"http://{config.api.host}:{config.api.port}"
        self.poll_interval = poll_interval

        self.client = httpx.Client(base_url=self.base_url, timeout=2.0)
        self.queue: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.running = True

        self.run_id: Optional[str] = None
        self.run_started_at: Optional[float] = None
        self.current_risk = config.inference.risk_profile

        # UI setup
        self.root = tk.Tk()
        self.root.title("ArcX Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.configure(bg="#0f1117")

        self._drag_offset: tuple[int, int] | None = None

        self._build_ui()
        self._make_draggable(self.header_frame)

        # Background polling thread
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()

        # Periodic queue processing
        self.root.after(100, self._process_queue)

    def _build_ui(self):
        self.header_frame = tk.Frame(self.root, bg="#0f1117")
        self.header_frame.pack(fill="x", padx=8, pady=(6, 2))

        self.title_label = tk.Label(
            self.header_frame,
            text="ArcX EV",
            fg="#e5e7eb",
            bg="#0f1117",
            font=("Segoe UI", 12, "bold"),
        )
        self.title_label.pack(side="left")

        self.status_var = tk.StringVar(value="Connecting...")
        self.status_label = tk.Label(
            self.header_frame,
            textvariable=self.status_var,
            fg="#9ca3af",
            bg="#0f1117",
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side="left", padx=(8, 0))

        self.close_btn = tk.Button(
            self.header_frame,
            text="X",
            command=self.close,
            fg="#e5e7eb",
            bg="#1f2937",
            activebackground="#4b5563",
            relief="flat",
            padx=6,
            pady=2,
            font=("Segoe UI", 9, "bold"),
        )
        self.close_btn.pack(side="right")

        body = tk.Frame(self.root, bg="#0f1117")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.recommendation = tk.Label(
            body,
            text="--",
            fg="#0f1117",
            bg="#facc15",
            font=("Segoe UI", 12, "bold"),
            width=24,
            padx=6,
            pady=6,
        )
        self.recommendation.pack(fill="x", pady=(0, 6))

        values_frame = tk.Frame(body, bg="#0f1117")
        values_frame.pack(fill="x")

        self.ev_stay_var = tk.StringVar(value="stay: --")
        self.ev_extract_var = tk.StringVar(value="extract: --")
        self.delta_var = tk.StringVar(value="Delta EV: --")

        tk.Label(
            values_frame,
            textvariable=self.ev_stay_var,
            fg="#a5b4fc",
            bg="#0f1117",
            font=("Segoe UI", 11),
        ).pack(anchor="w")
        tk.Label(
            values_frame,
            textvariable=self.ev_extract_var,
            fg="#fca5a5",
            bg="#0f1117",
            font=("Segoe UI", 11),
        ).pack(anchor="w")
        tk.Label(
            values_frame,
            textvariable=self.delta_var,
            fg="#fef3c7",
            bg="#0f1117",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(2, 0))

        # Risk profile buttons
        risk_frame = tk.Frame(body, bg="#0f1117")
        risk_frame.pack(fill="x", pady=(8, 4))
        tk.Label(
            risk_frame,
            text="Risk:",
            fg="#9ca3af",
            bg="#0f1117",
            font=("Segoe UI", 10),
        ).pack(side="left")
        for risk in ("safe", "neutral", "aggressive"):
            btn = tk.Button(
                risk_frame,
                text=risk.title(),
                command=lambda r=risk: self.set_risk(r),
                fg="#e5e7eb",
                bg="#1f2937",
                activebackground="#4b5563",
                relief="flat",
                padx=8,
                pady=4,
                font=("Segoe UI", 9),
            )
            btn.pack(side="left", padx=4)
            if risk == self.current_risk:
                btn.configure(bg="#2563eb")
            btn._risk_value = risk  # type: ignore[attr-defined]
            btn._base_bg = btn.cget("bg")  # type: ignore[attr-defined]
            setattr(self, f"risk_btn_{risk}", btn)

        # Run control buttons
        control_frame = tk.Frame(body, bg="#0f1117")
        control_frame.pack(fill="x", pady=(4, 0))

        self.run_status_var = tk.StringVar(value="Run: idle")
        tk.Label(
            control_frame,
            textvariable=self.run_status_var,
            fg="#9ca3af",
            bg="#0f1117",
            font=("Segoe UI", 9),
        ).pack(side="left")

        self.start_btn = tk.Button(
            control_frame,
            text="Start",
            command=self.start_run,
            fg="#e5e7eb",
            bg="#16a34a",
            activebackground="#22c55e",
            relief="flat",
            padx=8,
            pady=4,
            font=("Segoe UI", 9, "bold"),
        )
        self.start_btn.pack(side="right", padx=(4, 0))

        self.stop_btn = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_run,
            fg="#e5e7eb",
            bg="#dc2626",
            activebackground="#ef4444",
            relief="flat",
            padx=8,
            pady=4,
            font=("Segoe UI", 9, "bold"),
            state="disabled",
        )
        self.stop_btn.pack(side="right")

    def _make_draggable(self, widget: tk.Widget):
        widget.bind("<ButtonPress-1>", self._on_drag_start)
        widget.bind("<B1-Motion>", self._on_drag_motion)

    def _on_drag_start(self, event):
        self._drag_offset = (event.x, event.y)

    def _on_drag_motion(self, event):
        if self._drag_offset is None:
            return
        x = self.root.winfo_pointerx() - self._drag_offset[0]
        y = self.root.winfo_pointery() - self._drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def _poll_loop(self):
        while self.running:
            try:
                resp = self.client.get("/ev")
                if resp.status_code == 200:
                    self.queue.put(("ev", resp.json()))
                    self.status_var.set("Live")
                else:
                    self.queue.put(("error", f"{resp.status_code}: {resp.text}"))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("error", str(exc)))
            time.sleep(self.poll_interval)

    def _process_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "ev":
                    self._render_ev(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self._render_error(str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def _render_ev(self, ev_payload: dict):
        ev_stay = ev_payload.get("ev_stay")
        ev_extract = ev_payload.get("ev_extract")
        delta = ev_payload.get("delta_ev")
        recommendation = ev_payload.get("recommendation", "--")
        color = ev_payload.get("color", "yellow")
        message = ev_payload.get("message", "")

        self.ev_stay_var.set(f"stay: {ev_stay:.1f}" if ev_stay is not None else "stay: --")
        self.ev_extract_var.set(
            f"extract: {ev_extract:.1f}" if ev_extract is not None else "extract: --"
        )
        self.delta_var.set(
            f"Delta EV: {delta:.1f}" if delta is not None else "Delta EV: --"
        )

        bg_map = {"green": "#22c55e", "red": "#ef4444", "yellow": "#facc15"}
        self.recommendation.configure(
            text=message or recommendation,
            bg=bg_map.get(color, "#facc15"),
        )
        self.status_var.set(f"{recommendation} | risk={ev_payload.get('risk_profile')}")

    def _render_error(self, message: str):
        self.status_var.set(message)
        self.recommendation.configure(text=message, bg="#facc15")

    def set_risk(self, risk: str):
        try:
            resp = self.client.post("/config", json={"risk_profile": risk})
            resp.raise_for_status()
            self.current_risk = risk
            for name in ("safe", "neutral", "aggressive"):
                btn: tk.Button = getattr(self, f"risk_btn_{name}")
                btn.configure(bg="#2563eb" if name == risk else btn._base_bg)  # type: ignore[attr-defined]
            self.status_var.set(f"Risk set to {risk}")
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Risk update failed: {exc}")

    def start_run(self):
        if self.run_id:
            self.status_var.set("Run already active")
            return
        try:
            resp = self.client.post("/run/start", json={"map_id": "unknown"})
            resp.raise_for_status()
            data = resp.json()
            self.run_id = data.get("run_id")
            self.run_started_at = time.time()
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.run_status_var.set(f"Run: {self.run_id}")
            self.status_var.set("Capture started")
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Start failed: {exc}")

    def stop_run(self):
        if not self.run_id:
            self.status_var.set("No active run")
            return
        total_time = 0.0
        if self.run_started_at:
            total_time = time.time() - self.run_started_at
        payload = {
            "run_id": self.run_id,
            "final_loot_value": 0.0,
            "total_time_sec": total_time,
            "success": True,
            "action_taken": "extract",
        }
        try:
            resp = self.client.post("/run/end", json=payload)
            resp.raise_for_status()
            self.status_var.set("Run ended")
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"End failed: {exc}")
        finally:
            self.run_id = None
            self.run_started_at = None
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.run_status_var.set("Run: idle")

    def close(self):
        self.running = False
        try:
            self.client.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_overlay():
    """Entrypoint to launch the overlay."""
    app = NativeOverlayApp()
    app.run()


if __name__ == "__main__":
    run_overlay()
