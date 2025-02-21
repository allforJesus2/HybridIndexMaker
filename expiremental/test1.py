import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import os


class ImageOrientationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Orientation Analyzer")
        self.root.geometry("800x600")

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Browse button
        self.browse_button = ttk.Button(self.main_frame, text="Browse for Image", command=self.browse_image)
        self.browse_button.grid(row=0, column=0, pady=10)

        # File path label
        self.file_path_var = tk.StringVar()
        self.file_path_label = ttk.Label(self.main_frame, textvariable=self.file_path_var, wraplength=700)
        self.file_path_label.grid(row=1, column=0, pady=5)

        # Image preview frame
        self.preview_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="solid")
        self.preview_frame.grid(row=2, column=0, pady=10)

        # Image preview label
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(padx=10, pady=10)

        # Results label
        self.results_var = tk.StringVar()
        self.results_label = ttk.Label(self.main_frame, textvariable=self.results_var,
                                       font=('Arial', 12, 'bold'))
        self.results_label.grid(row=3, column=0, pady=10)

        # Initialize variables
        self.current_image = None
        self.photo_image = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.file_path_var.set(f"Selected file: {file_path}")
            self.load_and_analyze_image(file_path)

    def load_and_analyze_image(self, file_path):
        # Load and display image
        self.current_image = Image.open(file_path)

        # Resize image for preview while maintaining aspect ratio
        display_size = (400, 400)
        preview_image = self.resize_image(self.current_image, display_size)

        # Convert to PhotoImage for display
        self.photo_image = ImageTk.PhotoImage(preview_image)
        self.preview_label.configure(image=self.photo_image)

        # Analyze orientation
        angle = self.get_image_orientation(self.current_image)

        # Update results
        dimensions = self.current_image.size
        orientation = "Landscape" if dimensions[0] >= dimensions[1] else "Portrait"
        self.results_var.set(
            f"Dimensions: {dimensions[0]}x{dimensions[1]} ({orientation})\n"
            f"The Arrow points: {angle}"
        )

    def resize_image(self, image, max_size):
        ratio = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def get_image_orientation(self, image):
        # Convert to grayscale and numpy array
        img_gray = image.convert('L')
        pixel_array = np.array(img_gray)

        # Get dimensions
        height, width = pixel_array.shape

        if width >= height:  # Landscape or square
            # Split into left and right halves
            left_half = pixel_array[:, :width // 2]
            right_half = pixel_array[:, width // 2:]

            # Calculate average darkness (lower values are darker)
            left_darkness = np.mean(left_half)
            right_darkness = np.mean(right_half)

            # Return angle based on which side is lighter
            return 'left' if left_darkness > right_darkness else 'right'

        else:  # Portrait
            # Split into top and bottom halves
            top_half = pixel_array[:height // 2, :]
            bottom_half = pixel_array[height // 2:, :]

            # Calculate average darkness
            top_darkness = np.mean(top_half)
            bottom_darkness = np.mean(bottom_half)

            # Return angle based on which side is lighter
            return 'down' if bottom_darkness > top_darkness else 'up'


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageOrientationGUI(root)
    root.mainloop()