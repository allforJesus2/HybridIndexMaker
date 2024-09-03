import tkinter as tk


class SliderApp:
    def __init__(self, parent_window, labels_scores, callback=None):
        self.parent_window = parent_window
        self.labels_scores = labels_scores
        self.sliders = {}
        self.callback = callback
        # Configure the grid to allow resizing
        for i in range(len(labels_scores)):
            parent_window.rowconfigure(i, weight=1)
            parent_window.columnconfigure(0, weight=1)
            parent_window.columnconfigure(1, weight=1)
            parent_window.columnconfigure(2, weight=1)

        # Create sliders for each label-score pair
        for i, (label, score) in enumerate(labels_scores.items()):
            # Create a label for the slider
            label_widget = tk.Label(parent_window, text=label)
            label_widget.grid(row=i, column=0, sticky="ew")  # Use "ew" for east-west expansion

            # Create a slider for adjusting the score
            self.sliders[label] = tk.Scale(parent_window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
            self.sliders[label].set(score)
            self.sliders[label].grid(row=i, column=1, sticky="ew")  # Use "ew" for east-west expansion

            # Create a label to display the current score value
            self.sliders[label + "_display"] = tk.Label(parent_window, text=str(score))
            self.sliders[label + "_display"].grid(row=i, column=2, sticky="ew")  # Use "ew" for east-west expansion

            # Bind the slider change event to update the display label
            def update_display(event, label=label):
                current_value = self.sliders[label].get()
                self.sliders[label + "_display"].config(text=str(current_value))

            self.sliders[label].bind("<ButtonRelease-1>", update_display)

        # Add a Save and Close button
        save_and_close_button = tk.Button(parent_window, text="Save and Close", command=self.save_and_close)
        save_and_close_button.grid(row=len(labels_scores), column=1, sticky="ew")

    def save_and_close(self):
        # Collect and print slider values (simulating saving)
        saved_values = {}
        for label, slider in self.sliders.items():
            if not label.endswith("_display"):  # Skip display labels
                saved_values[label] = slider.get()
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
