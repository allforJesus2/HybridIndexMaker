import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import cv2
import os


class ImageEditor:
    def __init__(self, root, image=None, folder=None):
        self.root = root
        self.root.title("Image Editor")
        self.current_file_path = image
        self.folder_path = folder

        # Image handling variables
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.brightness = 1.0
        self.contrast = 1.0
        self.erosion = 0
        self.dilation = 0
        self.rotation = 0

        # Zoom and pan variables
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_panning = False

        # Color variables
        self.color_tolerance = 30
        self.selected_color = None
        self.is_bw = False
        self.color_preview_label = None
        self.color_operation = "No color change"

        self.setup_menu()
        self.setup_ui()

        if self.current_file_path:
            self.load_image_file(self.current_file_path)

    def setup_menu(self):
        """Setup menu bar with File and Edit menus"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Save As...", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Apply Changes", command=self.apply_changes)
        edit_menu.add_command(label="Batch Apply", command=self.batch_apply_settings)
        edit_menu.add_command(label="Reset Values", command=self.reset_values)

        # View menu
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        view_menu.add_command(label="Reset Pan", command=self.reset_pan)

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Control panel with fixed width
        control_frame = ttk.Frame(main_frame, width=250)  # Fixed width of 250 pixels
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        control_frame.pack_propagate(False)  # Prevent frame from resizing

        # Zoom control
        zoom_frame = ttk.LabelFrame(control_frame, text="Zoom", padding=5)
        zoom_frame.pack(fill='x', pady=(0, 10))
        self.zoom_scale_widget = ttk.Scale(zoom_frame, from_=0.1, to=5.0, value=1.0,
                                         orient='horizontal', command=self.update_image)
        self.zoom_scale_widget.pack(fill='x')

        # Rotation control
        rotation_frame = ttk.LabelFrame(control_frame, text="Rotation", padding=5)
        rotation_frame.pack(fill='x', pady=(0, 10))

        # Rotation scale
        ttk.Label(rotation_frame, text="Angle:").pack()
        self.rotation_scale = ttk.Scale(rotation_frame, from_=-180, to=180, value=0,
                                        orient='horizontal', command=self.update_image)
        self.rotation_scale.pack(fill='x')

        # Quick rotation buttons
        quick_rotation_frame = ttk.Frame(rotation_frame)
        quick_rotation_frame.pack(fill='x', pady=5)

        ttk.Button(quick_rotation_frame, text="90°",
                   command=lambda: self.quick_rotate(90)).pack(side='left', expand=True, padx=2)
        ttk.Button(quick_rotation_frame, text="180°",
                   command=lambda: self.quick_rotate(180)).pack(side='left', expand=True, padx=2)
        ttk.Button(quick_rotation_frame, text="270°",
                   command=lambda: self.quick_rotate(270)).pack(side='left', expand=True, padx=2)

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

        # Color operations section
        color_frame = ttk.LabelFrame(control_frame, text="Color Operations", padding=5)
        color_frame.pack(fill='x', pady=5)

        # Color operation dropdown
        self.color_operation_var = tk.StringVar(value="No color change")
        ttk.Label(color_frame, text="Color Operation:").pack(pady=(0, 5))
        self.color_operation_dropdown = ttk.Combobox(
            color_frame,
            textvariable=self.color_operation_var,
            values=["No color change", "Replace Color", "Isolate Color"],
            state="readonly"
        )
        self.color_operation_dropdown.pack(fill='x', pady=(0, 5))
        self.color_operation_dropdown.bind('<<ComboboxSelected>>', self.update_image)

        # Color preview frame
        preview_frame = ttk.Frame(color_frame)
        preview_frame.pack(fill='x', pady=5)

        # Color preview label
        self.color_preview_label = ttk.Label(preview_frame, text="Hover Color: RGB(-, -, -)")
        self.color_preview_label.pack(side='left')

        # Color preview box
        self.color_preview_box = tk.Canvas(preview_frame, width=20, height=20)
        self.color_preview_box.pack(side='left', padx=5)

        # RGB entry fields for target color
        target_frame = ttk.LabelFrame(color_frame, text="Target Color")
        target_frame.pack(fill='x', pady=5)

        rgb_frame = ttk.Frame(target_frame)
        rgb_frame.pack(fill='x')

        self.r_var = tk.StringVar(value="0")
        self.g_var = tk.StringVar(value="0")
        self.b_var = tk.StringVar(value="0")

        ttk.Label(rgb_frame, text="R:").pack(side='left')
        ttk.Entry(rgb_frame, width=4, textvariable=self.r_var).pack(side='left', padx=2)
        ttk.Label(rgb_frame, text="G:").pack(side='left')
        ttk.Entry(rgb_frame, width=4, textvariable=self.g_var).pack(side='left', padx=2)
        ttk.Label(rgb_frame, text="B:").pack(side='left')
        ttk.Entry(rgb_frame, width=4, textvariable=self.b_var).pack(side='left', padx=2)

        # RGB entry fields for replacement color
        replace_frame = ttk.LabelFrame(color_frame, text="Replacement Color")
        replace_frame.pack(fill='x', pady=5)

        replace_rgb_frame = ttk.Frame(replace_frame)
        replace_rgb_frame.pack(fill='x')

        self.replace_r_var = tk.StringVar(value="0")
        self.replace_g_var = tk.StringVar(value="0")
        self.replace_b_var = tk.StringVar(value="0")

        ttk.Label(replace_rgb_frame, text="R:").pack(side='left')
        ttk.Entry(replace_rgb_frame, width=4, textvariable=self.replace_r_var).pack(side='left', padx=2)
        ttk.Label(replace_rgb_frame, text="G:").pack(side='left')
        ttk.Entry(replace_rgb_frame, width=4, textvariable=self.replace_g_var).pack(side='left', padx=2)
        ttk.Label(replace_rgb_frame, text="B:").pack(side='left')
        ttk.Entry(replace_rgb_frame, width=4, textvariable=self.replace_b_var).pack(side='left', padx=2)

        # Color tolerance slider
        ttk.Label(color_frame, text="Color Tolerance:").pack()
        self.tolerance_scale = ttk.Scale(color_frame, from_=0, to=255, value=30,
                                         orient='horizontal', command=self.update_image)
        self.tolerance_scale.pack(fill='x')

        # Sample color button
        ttk.Button(color_frame, text="Sample Color", command=self.enable_color_sampling).pack(pady=5)

        # Black and White toggle
        self.bw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Black & White", variable=self.bw_var,
                        command=self.update_image).pack(pady=5)

        # Rest of the buttons
        ttk.Button(control_frame, text="Batch Apply", command=self.batch_apply_settings).pack(pady=5)
        ttk.Button(control_frame, text="Apply Changes", command=self.apply_changes).pack(pady=5)
        ttk.Button(control_frame, text="Save As...", command=self.save_as).pack(pady=5)
        ttk.Button(control_frame, text="Reset Values", command=self.reset_values).pack(pady=5)

        # Canvas for image display with scrollbars
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side='left', expand=True, fill='both')

        # Add scrollbars
        x_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal')
        y_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical')
        self.canvas = tk.Canvas(canvas_frame, width=640, height=480,
                                xscrollcommand=x_scrollbar.set,
                                yscrollcommand=y_scrollbar.set,
                                bg='gray')

        # Configure scrollbars
        x_scrollbar.config(command=self.canvas.xview)
        y_scrollbar.config(command=self.canvas.yview)

        # Pack scrollbars and canvas
        x_scrollbar.pack(side='bottom', fill='x')
        y_scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', expand=True, fill='both')

        # Bind mouse events for zoom and pan
        self.canvas.bind('<ButtonPress-1>', self.start_pan)
        self.canvas.bind('<B1-Motion>', self.pan)
        self.canvas.bind('<ButtonRelease-1>', self.stop_pan)
        self.canvas.bind('<MouseWheel>', self.zoom_with_mouse)  # Windows
        self.canvas.bind('<Button-4>', self.zoom_with_mouse)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.zoom_with_mouse)  # Linux scroll down

    def quick_rotate(self, angle):
        """Quickly rotate image by a multiple of 90 degrees"""
        self.rotation_scale.set(angle)
        self.update_image()

    def rotate_image(self, img, angle):
        """Rotate image with optimization for 90-degree multiples"""
        if angle == 0:
            return img

        # Check if angle is a multiple of 90
        if angle % 90 == 0:
            # Convert angle to range 0-360 and then to number of 90-degree rotations
            rotations = (int(angle) % 360) // 90
            # Use fast rotation for 90-degree multiples
            return img.rotate(angle, expand=True, resample=Image.NEAREST)
        else:
            # Use regular rotation for arbitrary angles
            return img.rotate(angle, expand=True, resample=Image.BICUBIC)

    def zoom_with_mouse(self, event):
        """Handle mouse wheel zoom events"""
        if not self.display_image:
            return

        # Get the current cursor position
        cursor_x = self.canvas.canvasx(event.x)
        cursor_y = self.canvas.canvasy(event.y)

        # Determine zoom direction
        if event.num == 5 or event.delta < 0:  # Zoom out
            self.zoom_scale = max(0.1, self.zoom_scale - 0.1)
        else:  # Zoom in
            self.zoom_scale = min(5.0, self.zoom_scale + 0.1)

        self.zoom_scale_widget.set(self.zoom_scale)
        self.update_image()

    def start_pan(self, event):
        """Start panning the image"""
        self.is_panning = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.canvas.config(cursor="fleur")

    def pan(self, event):
        """Pan the image"""
        if not self.is_panning:
            return

        # Calculate the distance moved
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y

        # Update pan position
        self.pan_x += dx
        self.pan_y += dy

        # Update drag start position
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        self.update_image()

    def stop_pan(self, event):
        """Stop panning the image"""
        self.is_panning = False
        self.canvas.config(cursor="")

    def reset_zoom(self):
        """Reset zoom to default"""
        self.zoom_scale = 1.0
        self.zoom_scale_widget.set(1.0)
        self.update_image()

    def reset_pan(self):
        """Reset pan position"""
        self.pan_x = 0
        self.pan_y = 0
        self.update_image()

    def update_image(self, *args):
        if self.original_image is None:
            return

        # Get current adjustment values
        brightness = self.brightness_scale.get()
        contrast = self.contrast_scale.get()
        erosion = int(self.erosion_scale.get())
        dilation = int(self.dilation_scale.get())
        rotation = float(self.rotation_scale.get())

        # Apply adjustments
        img = self.original_image.copy()

        # Apply rotation first
        if rotation != 0:
            img = self.rotate_image(img, rotation)

        # Apply other adjustments (brightness, contrast, etc.)
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

        # Apply morphological operations
        if erosion > 0 or dilation > 0:
            img = self.apply_morphological_ops(img, erosion, dilation)

        # Apply color operations
        if self.color_operation_var.get() != "No color change":
            if self.selected_color or all(v.get().isdigit() for v in [self.r_var, self.g_var, self.b_var]):
                img = self.process_color(img)

        # Convert to black and white if enabled
        if self.bw_var.get():
            img = self.convert_to_bw(img)

        # Apply zoom
        if self.zoom_scale != 1.0:
            new_width = int(img.width * self.zoom_scale)
            new_height = int(img.height * self.zoom_scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage for display
        self.display_image = img
        self.photo = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.delete("all")

        # Calculate center position with pan offset
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = self.photo.width()
        image_height = self.photo.height()

        x = (canvas_width - image_width) // 2 + self.pan_x
        y = (canvas_height - image_height) // 2 + self.pan_y

        # Create image on canvas
        self.canvas.create_image(x, y, image=self.photo, anchor='nw')

        # Update canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def reset_values(self):
        """Reset all adjustment values to their defaults"""
        self.brightness_scale.set(1.0)
        self.contrast_scale.set(1.0)
        self.erosion_scale.set(0)
        self.dilation_scale.set(0)
        self.rotation_scale.set(0)  # Reset rotation
        self.tolerance_scale.set(30)
        self.r_var.set("0")
        self.g_var.set("0")
        self.b_var.set("0")
        self.replace_r_var.set("0")
        self.replace_g_var.set("0")
        self.replace_b_var.set("0")
        self.selected_color = None
        self.bw_var.set(False)
        self.color_operation_var.set("No color change")
        self.update_image()

    def process_color(self, img):
        """Process image colors based on selected operation"""
        if not self.selected_color and not all(v.get().isdigit() for v in [self.r_var, self.g_var, self.b_var]):
            return img

        # Get RGB values and tolerance
        r = int(self.r_var.get())
        g = int(self.g_var.get())
        b = int(self.b_var.get())

        # Get replacement color values
        replace_r = int(self.replace_r_var.get())
        replace_g = int(self.replace_g_var.get())
        replace_b = int(self.replace_b_var.get())

        tolerance = self.tolerance_scale.get()

        # Convert PIL image to OpenCV format
        img_array = np.array(img)

        # Ensure the image is in BGR format for OpenCV
        if len(img_array.shape) == 2:  # Handle grayscale images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # Handle RGBA images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # Handle RGB images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Create bounds for color range (in BGR format for OpenCV)
        lower_bound = np.array([max(0, x - tolerance) for x in [b, g, r]], dtype=np.uint8)
        upper_bound = np.array([min(255, x + tolerance) for x in [b, g, r]], dtype=np.uint8)

        # Create mask for colors within tolerance range
        mask = cv2.inRange(img_array, lower_bound, upper_bound)

        # Create RGBA image for transparency
        rgba = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)

        operation = self.color_operation_var.get()

        if operation == "Replace Color":
            # Replace the selected color with the replacement color
            rgba[mask > 0] = [replace_b, replace_g, replace_r, 255]
        elif operation == "Isolate Color":
            # Replace all colors except the target color with the replacement color
            non_matching_pixels = cv2.bitwise_not(mask)
            rgba[non_matching_pixels > 0] = [replace_b, replace_g, replace_r, 255]

        # Convert back to PIL image
        return Image.fromarray(rgba)

    def remove_color(self, img):
        """Remove or replace selected color from image within tolerance range"""
        if not self.color_removal_var.get():  # Check if color removal is enabled
            return img

        if not self.selected_color and not all(v.get().isdigit() for v in [self.r_var, self.g_var, self.b_var]):
            return img

        # Get RGB values and tolerance
        r = int(self.r_var.get())
        g = int(self.g_var.get())
        b = int(self.b_var.get())

        # Get replacement color values
        replace_r = int(self.replace_r_var.get())
        replace_g = int(self.replace_g_var.get())
        replace_b = int(self.replace_b_var.get())

        tolerance = self.tolerance_scale.get()

        # Convert PIL image to OpenCV format
        img_array = np.array(img)

        # Ensure the image is in BGR format for OpenCV
        if len(img_array.shape) == 2:  # Handle grayscale images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # Handle RGBA images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # Handle RGB images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Create bounds for color range (in BGR format for OpenCV)
        lower_bound = np.array([max(0, x - tolerance) for x in [b, g, r]], dtype=np.uint8)
        upper_bound = np.array([min(255, x + tolerance) for x in [b, g, r]], dtype=np.uint8)

        # Create mask for colors within tolerance range
        mask = cv2.inRange(img_array, lower_bound, upper_bound)

        # Create RGBA image for transparency
        rgba = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGBA)

        # Replace colors instead of making them transparent
        rgba[mask > 0] = [replace_b, replace_g, replace_r, 255]

        # Convert back to PIL image
        return Image.fromarray(rgba)

    def enable_color_sampling(self):
        """Enable color sampling mode"""
        self.color_sampling_enabled = True
        self.canvas.config(cursor="crosshair")
        # Bind color sampling events
        self.canvas.bind('<Button-1>', self.sample_color)
        self.canvas.bind('<Motion>', self.preview_color)  # Add hover event

    def disable_color_sampling(self):
        """Disable color sampling mode"""
        self.color_sampling_enabled = False
        self.canvas.config(cursor="")
        self.canvas.unbind('<Button-1>')
        self.canvas.unbind('<Motion>')
        self.color_preview_label.configure(text="Hover Color: RGB(-, -, -)")
        self.color_preview_box.delete("all")

    def preview_color(self, event):
        """Preview color under cursor"""
        if self.display_image:
            try:
                # Convert canvas coordinates to image coordinates
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                img_width, img_height = self.display_image.size

                # Calculate scaling factors
                scale_x = img_width / canvas_width
                scale_y = img_height / canvas_height

                # Calculate actual image coordinates
                x = int(event.x * scale_x)
                y = int(event.y * scale_y)

                # Ensure coordinates are within image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))

                # Get pixel color
                r, g, b = self.display_image.convert('RGB').getpixel((x, y))

                # Update color preview label
                self.color_preview_label.configure(text=f"Hover Color: RGB({r}, {g}, {b})")

                # Update color preview box
                self.color_preview_box.delete("all")
                self.color_preview_box.configure(bg=f'#{r:02x}{g:02x}{b:02x}')

            except Exception as e:
                print(f"Error previewing color: {str(e)}")

    def sample_color(self, event):
        """Sample color from the clicked pixel"""
        if self.display_image:
            # Convert canvas coordinates to image coordinates
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = self.display_image.size

            # Calculate scaling factors
            scale_x = img_width / canvas_width
            scale_y = img_height / canvas_height

            # Calculate actual image coordinates
            x = int(event.x * scale_x)
            y = int(event.y * scale_y)

            # Ensure coordinates are within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))

            # Get pixel color
            try:
                r, g, b = self.display_image.convert('RGB').getpixel((x, y))
                self.r_var.set(str(r))
                self.g_var.set(str(g))
                self.b_var.set(str(b))
                self.selected_color = (r, g, b)

                # Disable color sampling mode
                self.disable_color_sampling()

                # Update image with new color selection
                self.update_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to sample color at selected position: {e}")
                print(e)

    def convert_to_bw(self, img):
        """Convert image to black and white"""
        return img.convert('L').convert('RGB')

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
        """
        Batch process images with specified adjustments and save to output folder.
        Includes progress tracking, error handling, and input validation.
        """
        # Validate or get input folder
        folder_path = self.folder_path or filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            return

        # Get and validate output folder
        output_folder = filedialog.askdirectory(title="Select Output Folder")
        if not output_folder:
            return

        # Check for input/output folder conflict
        if output_folder == folder_path:
            if not messagebox.askyesno("Warning",
                                       "Input folder is the same as output. Images will be overwritten. Continue?"):
                return

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Get current adjustment values with input validation
        adjustments = {
            'brightness': max(0.0, float(self.brightness_scale.get())),
            'contrast': max(0.0, float(self.contrast_scale.get())),
            'erosion': max(0, int(self.erosion_scale.get())),
            'dilation': max(0, int(self.dilation_scale.get())),
            'rotation': float(self.rotation_scale.get()),  # Add rotation to adjustments
            'color_operation': self.color_operation_var.get(),
            'target_color': {
                'r': int(self.r_var.get() or 0),
                'g': int(self.g_var.get() or 0),
                'b': int(self.b_var.get() or 0)
            },
            'replacement_color': {
                'r': int(self.replace_r_var.get() or 0),
                'g': int(self.replace_g_var.get() or 0),
                'b': int(self.replace_b_var.get() or 0)
            },
            'color_tolerance': self.tolerance_scale.get(),
            'is_bw': self.bw_var.get()
        }

        # Supported image formats
        SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

        # Initialize progress tracking
        image_files = [f for f in os.listdir(folder_path)
                       if os.path.splitext(f.lower())[1] in SUPPORTED_FORMATS]
        total_files = len(image_files)

        if not total_files:
            messagebox.showinfo("No Images Found",
                                f"No supported images found in {folder_path}")
            return

        processed_count = 0
        error_count = 0
        error_log = []

        # Create progress bar
        progress_window = tk.Toplevel()
        progress_window.title("Processing Images")
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window,
                                       variable=progress_var,
                                       maximum=total_files)
        progress_bar.pack(pady=10, padx=10, fill='x')
        progress_label = ttk.Label(progress_window, text="Processing...")
        progress_label.pack(pady=5)

        try:
            for index, filename in enumerate(image_files):
                file_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_folder, filename)

                # Update progress
                progress_var.set(index)
                progress_label.config(text=f"Processing: {filename}")
                progress_window.update()

                try:
                    with Image.open(file_path) as img:
                        # Convert to RGB if necessary
                        if img.mode not in ('RGB', 'L'):
                            img = img.convert('RGB')

                        # Apply adjustments only if they differ from default values
                        if adjustments['brightness'] != 1.0:
                            img = ImageEnhance.Brightness(img).enhance(
                                adjustments['brightness'])
                        if adjustments['contrast'] != 1.0:
                            img = ImageEnhance.Contrast(img).enhance(
                                adjustments['contrast'])

                        # Apply morphological operations if needed
                        if adjustments['erosion'] > 0 or adjustments['dilation'] > 0:
                            img = self.apply_morphological_ops(
                                img,
                                adjustments['erosion'],
                                adjustments['dilation']
                            )

                        # Apply color operations if enabled
                        if adjustments['color_operation'] != "No color change":
                            # Temporarily store the current settings
                            current_color_op = self.color_operation_var.get()
                            current_selected_color = self.selected_color

                            # Set the processing parameters
                            self.color_operation_var.set(adjustments['color_operation'])
                            self.selected_color = (
                                adjustments['target_color']['r'],
                                adjustments['target_color']['g'],
                                adjustments['target_color']['b']
                            )

                            # Process the color
                            img = self.process_color(img)

                            # Restore the original settings
                            self.color_operation_var.set(current_color_op)
                            self.selected_color = current_selected_color

                        # Apply black and white if enabled
                        if adjustments['is_bw']:
                            img = self.convert_to_bw(img)

                        # Preserve original image format and metadata
                        img.save(output_path,
                                 format=img.format,
                                 quality=95,
                                 exif=img.info.get('exif'))

                        processed_count += 1

                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing {filename}: {str(e)}"
                    error_log.append(error_msg)
                    print(error_msg)  # Console logging

        finally:
            # Close progress window
            progress_window.destroy()

            # Show completion message with details
            completion_msg = (
                f"Processing complete!\n"
                f"Successfully processed: {processed_count} images\n"
                f"Errors: {error_count}"
            )

            if error_log:
                # Save error log if there were any errors
                log_path = os.path.join(output_folder, "processing_errors.log")
                with open(log_path, 'w') as f:
                    f.write('\n'.join(error_log))
                completion_msg += f"\nError details saved to: {log_path}"

            messagebox.showinfo("Batch Processing Complete", completion_msg)

def main():
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
