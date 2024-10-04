import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import cv2
import os


class ImageEditor:
    def __init__(self, root, image=None):
        self.root = root
        self.root.title("Image Editor")
        self.current_file_path = image

        # Image handling variables
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.current_file_path = None  # Store the path of the loaded image
        self.brightness = 1.0
        self.contrast = 1.0
        self.erosion = 0
        self.dilation = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.view_x = 0
        self.view_y = 0

        self.setup_ui()

        if self.current_file_path:
            self.original_image = Image.open(self.current_file_path)
            self.reset_values()
            self.update_image()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side='left', fill='y', padx=(0, 10))

        # Load image button
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(pady=(0, 10))

        # Brightness control
        ttk.Label(control_frame, text="Brightness:").pack()
        self.brightness_scale = ttk.Scale(control_frame, from_=0.0, to=2.0, value=1.0,
                                          orient='horizontal', command=self.update_image)
        self.brightness_scale.pack(fill='x', pady=(0, 10))

        # Contrast control
        ttk.Label(control_frame, text="Contrast:").pack()
        self.contrast_scale = ttk.Scale(control_frame, from_=0.0, to=10.0, value=1.0,
                                        orient='horizontal', command=self.update_image)
        self.contrast_scale.pack(fill='x', pady=(0, 10))

        # Erosion control
        ttk.Label(control_frame, text="Erosion:").pack()
        self.erosion_scale = ttk.Scale(control_frame, from_=0, to=10, value=0,
                                       orient='horizontal', command=self.update_image)
        self.erosion_scale.pack(fill='x', pady=(0, 10))

        # Dilation control
        ttk.Label(control_frame, text="Dilation:").pack()
        self.dilation_scale = ttk.Scale(control_frame, from_=0, to=10, value=0,
                                        orient='horizontal', command=self.update_image)
        self.dilation_scale.pack(fill='x', pady=(0, 10))

        # Add Batch Apply button
        ttk.Button(control_frame, text="Batch Apply", command=self.batch_apply_settings).pack(pady=5)

        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill='x', pady=10)

        # Apply Changes button
        ttk.Button(control_frame, text="Apply Changes", command=self.apply_changes).pack(pady=5)

        # Save As button
        ttk.Button(control_frame, text="Save As...", command=self.save_as).pack(pady=5)

        # Reset Values button
        ttk.Button(control_frame, text="Reset Values", command=self.reset_values).pack(pady=5)

        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, width=640, height=480, bg='gray')
        self.canvas.pack(side='left')

        # Bind mouse events for panning
        self.canvas.bind('<ButtonPress-1>', self.start_pan)
        self.canvas.bind('<B1-Motion>', self.pan)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")])
        if file_path:
            self.current_file_path = file_path
            self.original_image = Image.open(file_path)
            self.reset_values()
            self.update_image()

    def save_as(self):
        """Save the current image to a new file"""
        if self.display_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"),
                           ("JPEG files", "*.jpg"),
                           ("All files", "*.*")],
                initialfile=os.path.basename(self.current_file_path) if self.current_file_path else None
            )
            if file_path:
                self.display_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved as {file_path}")

    def apply_changes(self):
        """Apply current modifications by overwriting the original file"""
        if not self.current_file_path or not self.display_image:
            messagebox.showwarning("Warning", "No image loaded or no changes to apply.")
            return

        try:

            # Save the modified image over the original file
            self.display_image.save(self.current_file_path)

            # Update the original image in memory
            self.original_image = self.display_image.copy()

            # Reset all adjustment values
            self.reset_values()

            messagebox.showinfo("Success",
                                f"Changes applied and saved to {self.current_file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save changes: {str(e)}")

    def reset_values(self):
        """Reset all adjustment values to their defaults"""
        self.brightness_scale.set(1.0)
        self.contrast_scale.set(1.0)
        self.erosion_scale.set(0)
        self.dilation_scale.set(0)
        self.update_image()

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        # Update view position
        delta_x = event.x - self.pan_start_x
        delta_y = event.y - self.pan_start_y
        self.view_x -= delta_x
        self.view_y -= delta_y
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def apply_morphological_ops(self, img, erosion_size, dilation_size):
        # Convert PIL image to OpenCV format
        img_array = np.array(img)

        # Create kernels for morphological operations
        if erosion_size > 0:
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            img_array = cv2.erode(img_array, kernel, iterations=1)

        if dilation_size > 0:
            kernel = np.ones((dilation_size, dilation_size), np.uint8)
            img_array = cv2.dilate(img_array, kernel, iterations=1)

        # Convert back to PIL image
        return Image.fromarray(img_array)

    def batch_apply_settings(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            return

        # Get current adjustment values
        brightness = self.brightness_scale.get()
        contrast = self.contrast_scale.get()
        erosion = int(self.erosion_scale.get())
        dilation = int(self.dilation_scale.get())

        processed_count = 0
        error_count = 0

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                file_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(file_path)

                    # Apply brightness and contrast
                    if brightness != 1.0:
                        enhancer = ImageEnhance.Brightness(img)
                        img = enhancer.enhance(brightness)
                    if contrast != 1.0:
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(contrast)

                    # Apply morphological operations
                    if erosion > 0 or dilation > 0:
                        img = self.apply_morphological_ops(img, erosion, dilation)

                    # Save the modified image
                    output_path = os.path.join(folder_path, filename)
                    img.save(output_path)
                    print(f'done processing {output_path}')
                    processed_count += 1

                except Exception as e:
                    error_count += 1
                    print(f"Error processing {filename}: {str(e)}")

        messagebox.showinfo("Batch Processing Complete",
                            f"Processed {processed_count} images.\n{error_count} errors occurred.")

    def update_image(self, *args):
        if self.original_image is None:
            return

        # Get current adjustment values
        brightness = self.brightness_scale.get()
        contrast = self.contrast_scale.get()
        erosion = int(self.erosion_scale.get())
        dilation = int(self.dilation_scale.get())

        # Apply adjustments
        img = self.original_image.copy()

        # Apply brightness and contrast
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

        # Apply morphological operations
        if erosion > 0 or dilation > 0:
            img = self.apply_morphological_ops(img, erosion, dilation)

        # Convert to PhotoImage for display
        self.display_image = img
        self.photo = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(640 // 2, 480 // 2, image=self.photo, anchor='center')


def main():
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
