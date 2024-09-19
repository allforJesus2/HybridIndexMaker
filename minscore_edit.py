import tkinter as tk
from tkinter import ttk

class SliderApp:
    def __init__(self, parent_window, labels_scores, callback=None):
        self.parent_window = parent_window
        self.labels_scores = labels_scores
        self.sliders = {}
        self.callback = callback

        # Create a canvas with a scrollbar
        self.canvas = tk.Canvas(parent_window)
        self.scrollbar = ttk.Scrollbar(parent_window, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Set the canvas height to 600px
        self.canvas.config(height=600)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Create sliders for each label-score pair
        for i, (label, score) in enumerate(labels_scores.items()):
            # Create a label for the slider
            label_widget = ttk.Label(self.scrollable_frame, text=label)
            label_widget.grid(row=i, column=0, sticky="ew", padx=5, pady=5)

            # Create a slider for adjusting the score
            self.sliders[label] = ttk.Scale(self.scrollable_frame, from_=0, to=1, orient=tk.HORIZONTAL, length=200)
            self.sliders[label].set(score)
            self.sliders[label].grid(row=i, column=1, sticky="ew", padx=5, pady=5)

            # Create a label to display the current score value
            self.sliders[label + "_display"] = ttk.Label(self.scrollable_frame, text=str(score))
            self.sliders[label + "_display"].grid(row=i, column=2, sticky="ew", padx=5, pady=5)

            # Bind the slider change event to update the display label
            self.sliders[label].bind("<Motion>", lambda event, label=label: self.update_display(label))
            self.sliders[label].bind("<ButtonRelease-1>", lambda event, label=label: self.update_display(label))

        # Add a Save and Close button
        save_and_close_button = ttk.Button(self.scrollable_frame, text="Save and Close", command=self.save_and_close)
        save_and_close_button.grid(row=len(labels_scores), column=1, sticky="ew", padx=5, pady=10)

    def update_display(self, label):
        current_value = round(self.sliders[label].get(), 2)
        self.sliders[label + "_display"].config(text=str(current_value))

    def save_and_close(self):
        # Collect and print slider values (simulating saving)
        saved_values = {}
        for label, slider in self.sliders.items():
            if not label.endswith("_display"):  # Skip display labels
                saved_values[label] = round(slider.get(), 2)
        print("Saved values:", saved_values)
        if self.callback:
            self.callback(saved_values)

        # Close the application
        self.parent_window.destroy()
# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    labels_scores = {"Label 1": 0.5, "Label 2": 0.75, "Label 3": 0.25}
    app = SliderApp(root, labels_scores)
    root.mainloop()
