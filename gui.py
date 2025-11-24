# main_gui.py
import tkinter as tk
from tkinter import ttk, Canvas, PhotoImage
import customtkinter as ctk
import requests
import threading
import json
import time
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import websockets
import asyncio

# Configuration
API_BASE = "http://localhost:8080"
WS_URL = "ws://localhost:8080/voice-stream/default_user"
THEME = {
    "primary": "#2A2F3D",
    "secondary": "#3D4455",
    "accent": "#00C9B8",
    "text": "#FFFFFF",
    "warning": "#FF6B6B"
}

class NeuroInterface:
    """Simulated neural interface controller"""
    def __init__(self):
        self.cognitive_load = 0.0
        self.attention_level = 0.0
        self.engagement = 0.0
        
    def update_stats(self):
        """Simulate neural data updates"""
        self.cognitive_load = np.random.uniform(0, 1)
        self.attention_level = np.random.uniform(0.5, 1)
        self.engagement = np.random.uniform(0.3, 0.9)

class ARCanvas(ctk.CTkCanvas):
    """Augmented Reality canvas with OpenCV integration"""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.cap = cv2.VideoCapture(0)
        self.ar_overlays = []
        self.configure(width=640, height=480)
        
        self.bind("<Configure>", self._resize)
        self._update_frame()

    def _resize(self, event):
        self.width = event.width
        self.height = event.height

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.img_tk = ImageTk.PhotoImage(image=img)
            self.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.after(30, self._update_frame)

    def add_overlay(self, text, position):
        self.ar_overlays.append((text, position))
        
class MainApplication(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.neuro_interface = NeuroInterface()
        self.setup_ui()
        self.start_background_tasks()

    def setup_ui(self):
        self.title("NeuroAI Interface v2.0")
        self.geometry("1440x900")
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=THEME["primary"])
        
        # Main grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create interface sections
        self.create_neural_panel()
        self.create_main_display()
        self.create_ar_panel()
        self.create_quantum_bar()

    def create_neural_panel(self):
        """Left neural control panel"""
        self.neural_frame = ctk.CTkFrame(self, width=300, fg_color=THEME["secondary"])
        self.neural_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        
        # Neural metrics
        self.brain_wave_chart = self.create_wave_visualization()
        self.cognitive_load_bar = ctk.CTkProgressBar(self.neural_frame, orientation="vertical")
        self.attention_meter = ctk.CTkProgressBar(self.neural_frame, orientation="vertical")
        
        # Layout neural components
        self.brain_wave_chart.grid(row=0, column=0, pady=20)
        self.cognitive_load_bar.grid(row=1, column=0, pady=10)
        self.attention_meter.grid(row=2, column=0, pady=10)

    def create_main_display(self):
        """Central holographic display area"""
        self.main_frame = ctk.CTkFrame(self, fg_color=THEME["primary"])
        self.main_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        
        # 3D Avatar Display (Simulated)
        self.avatar_canvas = ctk.CTkCanvas(self.main_frame, bg=THEME["primary"])
        self.avatar_canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_avatar()

        # Quantum prediction feed
        self.prediction_feed = ctk.CTkTextbox(self.main_frame, height=100)
        self.prediction_feed.pack(fill=tk.X, pady=10)

    def create_ar_panel(self):
        """Right AR panel with camera feed"""
        self.ar_frame = ctk.CTkFrame(self, width=400, fg_color=THEME["secondary"])
        self.ar_frame.grid(row=0, column=2, sticky="nswe", padx=10, pady=10)
        
        self.ar_canvas = ARCanvas(self.ar_frame)
        self.ar_canvas.pack(fill=tk.BOTH, expand=True)

    def create_quantum_bar(self):
        """Bottom prediction bar"""
        self.quantum_bar = ctk.CTkFrame(self, height=80, fg_color=THEME["secondary"])
        self.quantum_bar.grid(row=1, column=0, columnspan=3, sticky="we", padx=10, pady=10)
        
        self.prediction_buttons = [
            ctk.CTkButton(self.quantum_bar, text=f"Prediction {i+1}", 
                         fg_color=THEME["accent"], hover_color=THEME["warning"])
            for i in range(3)
        ]
        for btn in self.prediction_buttons:
            btn.pack(side=tk.LEFT, padx=10)

    def create_wave_visualization(self):
        """Brain wave visualization using matplotlib"""
        fig = Figure(figsize=(4, 2), dpi=100)
        ax = fig.add_subplot(111)
        x = np.linspace(0, 4*np.pi, 100)
        y = np.sin(x)
        ax.plot(x, y, color=THEME["accent"])
        ax.axis("off")
        
        canvas = FigureCanvasTkAgg(fig, master=self.neural_frame)
        return canvas.get_tk_widget()

    def draw_avatar(self):
        """Simulated 3D avatar using basic shapes"""
        canvas = self.avatar_canvas
        canvas.delete("all")
        
        # Draw avatar base
        canvas.create_oval(100, 50, 300, 250, fill=THEME["accent"], outline="")
        canvas.create_arc(100, 100, 300, 300, start=0, extent=180, fill=THEME["warning"])
        
        # Animate with simple transform
        self.after(1000, self.draw_avatar)

    def start_background_tasks(self):
        """Start background updates"""
        threading.Thread(target=self.update_neural_stats, daemon=True).start()
        threading.Thread(target=self.fetch_predictions, daemon=True).start()
        threading.Thread(target=self.start_websocket).start()

    def update_neural_stats(self):
        """Update neural interface metrics"""
        while True:
            self.neuro_interface.update_stats()
            self.cognitive_load_bar.set(self.neuro_interface.cognitive_load)
            self.attention_meter.set(self.neuro_interface.attention_level)
            time.sleep(1)

    def fetch_predictions(self):
        """Get quantum predictions from AI backend"""
        while True:
            try:
                response = requests.post(f"{API_BASE}/predict", json={
                    "context": "current_state"
                })
                predictions = response.json().get("predictions", [])
                self.update_prediction_buttons(predictions)
            except Exception as e:
                print(f"Prediction error: {e}")
            time.sleep(5)

    def update_prediction_buttons(self, predictions):
        """Update quantum prediction buttons"""
        for btn, text in zip(self.prediction_buttons, predictions[:3]):
            btn.configure(text=text)

    def start_websocket(self):
        """Handle WebSocket communication"""
        asyncio.run(self.websocket_handler())

    async def websocket_handler(self):
        async with websockets.connect(WS_URL) as ws:
            while True:
                try:
                    message = await ws.recv()
                    self.process_ws_message(message)
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break

    def process_ws_message(self, message):
        """Process incoming WebSocket messages"""
        data = json.loads(message)
        if data["type"] == "voice_response":
            self.prediction_feed.insert(tk.END, f"AI: {data['content']}\n")

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()