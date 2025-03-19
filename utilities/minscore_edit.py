import tkinter as tk
from tkinter import ttk, messagebox


class SliderApp:
    def __init__(self, parent_window, labels_scores, callback=None, title="Slider Application"):
        self.parent_window = parent_window
        self.labels_scores = labels_scores
        self.sliders = {}
        self.callback = callback
        self.original_values = labels_scores.copy()

        # Configure the parent window
        self.parent_window.title(title)
        self.parent_window.minsize(400, 300)

        # Create main frame
        self.main_frame = ttk.Frame(parent_window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create fixed header frame
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill="x", pady=(0, 5))

        # Create header labels
        ttk.Label(self.header_frame, text="Label", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(5, 0))
        ttk.Label(self.header_frame, text="Value", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(215, 0))
        ttk.Label(self.header_frame, text="Score", font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(165, 0))

        # Create separator after header
        ttk.Separator(self.main_frame, orient='horizontal').pack(fill='x', pady=5)

        # Create the scrollable middle section
        self.create_scrollable_content()

        # Create separator before footer
        ttk.Separator(self.main_frame, orient='horizontal').pack(fill='x', pady=5)

        # Create fixed footer frame
        self.footer_frame = ttk.Frame(self.main_frame)
        self.footer_frame.pack(fill="x", pady=(5, 0))

        # Create Set All frame with entry and button
        set_all_frame = ttk.Frame(self.footer_frame)
        set_all_frame.pack(side="left", padx=5)

        ttk.Label(set_all_frame, text="Set all to:").pack(side="left", padx=(0, 5))
        self.set_all_value = ttk.Entry(set_all_frame, width=6)
        self.set_all_value.pack(side="left", padx=(0, 5))
        self.set_all_value.insert(0, "0.5")  # Default value
        ttk.Button(set_all_frame, text="Set All", command=self.set_all_values).pack(side="left")

        # Create button frame for other controls
        button_frame = ttk.Frame(self.footer_frame)
        button_frame.pack(side="right", padx=5)

        # Add buttons
        ttk.Button(button_frame, text="Reset All", command=self.reset_values).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_values).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save and Close", command=self.save_and_close).pack(side="left", padx=5)

    def create_scrollable_content(self):
        # Create a frame to hold the canvas and scrollbar
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        # Create a canvas with a scrollbar
        self.canvas = tk.Canvas(self.canvas_frame)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Add mousewheel scrolling only to the canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Create sliders for each label-score pair
        for i, (label, score) in enumerate(self.labels_scores.items()):
            # Create a frame for each row
            row_frame = ttk.Frame(self.scrollable_frame)
            row_frame.pack(fill="x", padx=5, pady=2)

            # Create a label for the slider
            label_widget = ttk.Label(row_frame, text=label, width=20)
            label_widget.pack(side="left")

            # Create a slider for adjusting the score
            self.sliders[label] = ttk.Scale(
                row_frame,
                from_=0,
                to=1,
                orient=tk.HORIZONTAL,
                length=200
            )
            self.sliders[label].set(score)
            self.sliders[label].pack(side="left", padx=(5, 5))

            # Create a label to display the current score value
            self.sliders[label + "_display"] = ttk.Label(
                row_frame,
                text=f"{score:.2f}",
                width=8
            )
            self.sliders[label + "_display"].pack(side="left")

            # Bind the slider change events
            self.sliders[label].bind("<Motion>", lambda event, lbl=label: self.update_display(lbl))
            self.sliders[label].bind("<ButtonRelease-1>", lambda event, lbl=label: self.update_display(lbl))

    def set_all_values(self):
        try:
            value = float(self.set_all_value.get())
            if 0 <= value <= 1:
                for label in self.labels_scores.keys():
                    self.sliders[label].set(value)
                    self.update_display(label)
            else:
                messagebox.showerror("Error", "Please enter a value between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def _on_mousewheel(self, event):
        if self.canvas.winfo_exists():  # Check if canvas still exists
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_display(self, label):
        current_value = round(self.sliders[label].get(), 2)
        self.sliders[label + "_display"].config(text=f"{current_value:.2f}")

    def reset_values(self):
        for label, original_value in self.original_values.items():
            self.sliders[label].set(original_value)
            self.update_display(label)

    def save_values(self):
        saved_values = self._get_current_values()
        if self.callback:
            self.callback(saved_values)
        messagebox.showinfo("Success", "Values have been saved!")

    def save_and_close(self):
        saved_values = self._get_current_values()
        if self.callback:
            self.callback(saved_values)
        self.parent_window.destroy()

    def _get_current_values(self):
        return {
            label: round(slider.get(), 2)
            for label, slider in self.sliders.items()
            if not label.endswith("_display")
        }


def example_callback(values):
    print("Saved values:", values)


if __name__ == "__main__":
    root = tk.Tk()
    labels_scores = {
        "Quality Score": 0.5,
        "Performance Rating": 0.75,
        "User Satisfaction": 0.25,
        "Reliability Index": 0.8,
        "Cost Efficiency": 0.6,
        "Speed Score": 0.4,
        "Accuracy Rating": 0.9,
        "Innovation Index": 0.3,
        "Customer Feedback": 0.7,
        "Market Impact": 0.55
    }
    app = SliderApp(root, labels_scores, callback=example_callback, title="Score Adjustment Tool")
    root.mainloop()