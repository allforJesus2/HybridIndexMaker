import tkinter as tk
from tkinter import ttk

class ProgressWindow:
    def __init__(self, parent, total_pages):
        self.window = tk.Toplevel(parent)
        self.window.title("Processing Pages")
        self.window.transient(parent)
        self.window.grab_set()

        # Center the window
        window_width = 300
        window_height = 150
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Progress label
        self.label = ttk.Label(self.window, text="Processing page: 0/" + str(total_pages))
        self.label.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.window, length=200, mode='determinate', maximum=total_pages)
        self.progress.pack(pady=10)

        # Cancel button
        self.cancelled = False
        self.cancel_button = ttk.Button(self.window, text="Cancel", command=self.cancel)
        self.cancel_button.pack(pady=10)

        # Prevent window from being closed with X button
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

    def update(self, current_page):
        self.label.config(text=f"Processing page: {current_page}/{self.progress['maximum']}")
        self.progress['value'] = current_page
        self.window.update()

    def cancel(self):
        self.cancelled = True
        self.cancel_button.config(state='disabled')
        self.label.config(text="Cancelling...")

    def destroy(self):
        self.window.destroy() 