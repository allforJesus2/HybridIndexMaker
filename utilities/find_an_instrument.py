import os.path
import tkinter as tk
from os import startfile
import math
from PIL import Image, ImageDraw, ImageTk, ImageFont
from tkinter import filedialog, colorchooser, ttk, messagebox
import time
import threading

class CropWindow(tk.Toplevel):
    def __init__(self, parent, image_path, coords, initial_width, initial_height, index=None, total=None):
        super().__init__(parent)

        self.image_path = image_path
        self.coords = coords
        self.full_image = Image.open(image_path)
        self.initial_width = initial_width
        self.initial_height = initial_height
        self.index = index
        self.total = total

        # Calculate center of the bounding box
        self.center_x = (coords[0] + coords[2]) // 2
        self.center_y = (coords[1] + coords[3]) // 2

        # Create control frame at the top
        control_frame = tk.Frame(self)
        control_frame.pack(fill='x', padx=5, pady=5)

        # Step size entry
        tk.Label(control_frame, text="Step (px):").pack(side=tk.LEFT)
        self.step_entry = tk.Entry(control_frame, width=8)
        self.step_entry.insert(0, "500")
        self.step_entry.pack(side=tk.LEFT, padx=5)

        # Zoom buttons
        tk.Button(control_frame, text="-", command=self.decrease_size).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="+", command=self.increase_size).pack(side=tk.LEFT, padx=2)

        # Open original image button
        tk.Button(control_frame, text="Open Original",
                  command=lambda: startfile(self.image_path)).pack(side=tk.RIGHT, padx=5)

        # Add new button to show highlighted region
        tk.Button(control_frame, text="Show Region in Full Image",
                  command=self.show_highlighted_region).pack(side=tk.RIGHT, padx=5)

        # Image label
        self.image_label = tk.Label(self)
        self.image_label.pack(expand=True, fill='both')

        # Set window title
        filename = os.path.basename(image_path)
        if index is not None and total is not None:
            self.title(f"{filename} - Region at {coords} [{index+1}/{total}]")
        else:
            self.title(f"{filename} - Region at {coords}")

        # Update the image
        self.update_crop(initial_width, initial_height)

    def show_highlighted_region(self):
        """Show the original image with the current crop region highlighted"""
        try:
            # Create a copy of the original image
            img_copy = self.full_image.copy()
            draw = ImageDraw.Draw(img_copy)

            # Calculate current crop coordinates
            left = max(0, self.center_x - self.current_width // 2)
            top = max(0, self.center_y - self.current_height // 2)
            right = min(self.full_image.width, left + self.current_width)
            bottom = min(self.full_image.height, top + self.current_height)

            # Draw red rectangle around the current crop region
            draw.rectangle([left, top, right, bottom], outline='red', width=5)

            # Create a temporary file with a .png extension
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                img_copy.save(temp_path)
                startfile(temp_path)

            # Schedule the temporary file for deletion after a delay
            # This gives the system time to open the file before deleting it
            def delete_temp_file():
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass

            self.after(1000, delete_temp_file)  # Delete after 1 second

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show highlighted region: {str(e)}")

    def get_step_size(self):
        try:
            return int(self.step_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid step size. Using default 500px.")
            self.step_entry.delete(0, tk.END)
            self.step_entry.insert(0, "500")
            return 500

    def increase_size(self):
        step = self.get_step_size()
        self.update_crop(self.current_width + step, self.current_height + step)

    def decrease_size(self):
        step = self.get_step_size()
        new_width = max(100, self.current_width - step)  # Minimum width of 100px
        new_height = max(100, self.current_height - step)  # Minimum height of 100px
        self.update_crop(new_width, new_height)

    def update_crop(self, width, height):
        # Store current dimensions
        self.current_width = width
        self.current_height = height

        # Calculate crop coordinates
        left = max(0, self.center_x - width // 2)
        top = max(0, self.center_y - height // 2)
        right = min(self.full_image.width, left + width)
        bottom = min(self.full_image.height, top + height)

        # Crop the region
        region = self.full_image.crop((left, top, right, bottom))

        # Convert to PhotoImage and update label
        photo = ImageTk.PhotoImage(region)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference!

class FindAnInstrumentApp:
    def __init__(self, root, img_path=''):
        self.root = root
        root.title("FAIA (find an instrument app)")

        # Initialize tracking variables
        self.cycling = False
        self.cycle_thread = None
        self.current_index = 0
        self.current_image = None
        self.region_windows = []
        self.region_images = []
        self.single_iteration = False
        
        # Add variables for image navigation
        self.parsed_images = []  # Will store [image_path, coords_strs] pairs
        self.current_image_index = 0
        
        # Track the last opened image file
        self.last_opened_image = None
        self.last_opened_image_process = None

        # Create main frames with clear separation
        self.file_frame = tk.LabelFrame(root, text="File Management", padx=10, pady=5)
        self.input_frame = tk.LabelFrame(root, text="Input Data", padx=10, pady=5)
        self.visualization_frame = tk.LabelFrame(root, text="Visualization Options", padx=10, pady=5)
        self.region_frame = tk.LabelFrame(root, text="Region Management", padx=10, pady=5)
        
        # Add new frame for image navigation
        self.image_nav_frame = tk.LabelFrame(root, text="Image Navigation", padx=10, pady=5)

        # === File Management Frame ===
        # Image path selection
        path_frame = tk.Frame(self.file_frame)
        tk.Label(path_frame, text="Image Path:").pack(side=tk.LEFT)
        self.image_path_entry = tk.Entry(path_frame, width=50)
        self.image_path_entry.pack(side=tk.LEFT, padx=5)
        self.image_path_entry.insert(tk.END, img_path)
        tk.Button(path_frame, text="Browse", command=self.browse_image).pack(side=tk.LEFT, padx=5)
        path_frame.pack(fill='x', pady=5)

        # Output directory selection
        output_frame = tk.Frame(self.file_frame)
        tk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame, width=50)
        self.output_dir_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.LEFT, padx=5)
        output_frame.pack(fill='x', pady=5)

        # === Input Data Frame ===
        # Combined input section
        tk.Label(self.input_frame, text="Combined Path and Tensor:").pack(anchor='w')
        self.combined_entry = tk.Text(self.input_frame, height=5, width=70)
        self.combined_entry.pack(fill='x', pady=5)

        # Parse buttons
        parse_buttons_frame = tk.Frame(self.input_frame)
        tk.Button(parse_buttons_frame, text="Parse & Overlay Indicators",
                  command=self.parse_combined_input).pack(side=tk.LEFT, padx=5)
        tk.Button(parse_buttons_frame, text="Parse & Open Crop Regions",
                  command=self.parse_and_show_regions).pack(side=tk.LEFT, padx=5)
        # Add paste button
        tk.Button(parse_buttons_frame, text="Paste",
                  command=self.paste_clipboard).pack(side=tk.LEFT, padx=5)
        parse_buttons_frame.pack(pady=5)

        # Coordinate input and labeling options
        coords_frame = tk.Frame(self.input_frame)

        # Left side - coordinates
        coords_left = tk.Frame(coords_frame)
        tk.Label(coords_left, text="Coordinates ([x1, y1, x2, y2]):").pack(anchor='w')
        self.coords_text = tk.Text(coords_left, height=8, width=30)
        self.coords_text.pack(pady=5)
        coords_left.pack(side=tk.LEFT, padx=10)

        # Right side - labels
        label_frame = tk.Frame(coords_frame)
        tk.Label(label_frame, text="Label Options:").pack(anchor='w')
        self.label_type_combo = tk.StringVar()
        self.label_type_combobox = ttk.Combobox(label_frame, textvariable=self.label_type_combo,
                                                values=["Index", "Coordinates", "Custom"],
                                                state='readonly')
        self.label_type_combobox.pack(pady=5)
        self.label_type_combo.set("Index")

        tk.Label(label_frame, text="Custom Labels:").pack(anchor='w')
        self.custom_labels_text = tk.Text(label_frame, height=5, width=20)
        self.custom_labels_text.pack(pady=5)
        label_frame.pack(side=tk.LEFT, padx=10)

        coords_frame.pack(pady=5)

        # === Visualization Options Frame ===
        options_frame = tk.Frame(self.visualization_frame)

        # Visual style options
        style_frame = tk.Frame(options_frame)
        # Color selection
        tk.Label(style_frame, text="Color:").pack(side=tk.LEFT)
        self.color_combo = tk.StringVar()
        self.color_combobox = ttk.Combobox(style_frame, textvariable=self.color_combo,
                                           values=["Red", "HotPink", "Green", "LawnGreen", "Blue", "Orange", "Yellow"],
                                           state='readonly', width=10)
        self.color_combobox.pack(side=tk.LEFT, padx=5)
        self.color_combo.set("Red")

        # Indicator type
        tk.Label(style_frame, text="Type:").pack(side=tk.LEFT, padx=(10, 0))
        self.indicator_combo = tk.StringVar()
        self.indicator_combobox = ttk.Combobox(style_frame, textvariable=self.indicator_combo,
                                               values=["Square", "Circle", "Crosshair", "Corner Lines"],
                                               state='readonly', width=10)
        self.indicator_combobox.pack(side=tk.LEFT, padx=5)
        self.indicator_combo.set("Square")

        # Line thickness
        tk.Label(style_frame, text="Thickness:").pack(side=tk.LEFT, padx=(10, 0))
        self.thickness_entry = tk.Entry(style_frame, width=5)
        self.thickness_entry.insert(0, '5')
        self.thickness_entry.pack(side=tk.LEFT, padx=5)

        style_frame.pack(pady=5)

        # Draw button
        tk.Button(options_frame, text="Draw Indicators",
                  command=self.draw_indicators).pack(pady=5)

        options_frame.pack(fill='x')
        
        # === Image Navigation Frame ===
        nav_frame = tk.Frame(self.image_nav_frame)
        
        # Add image navigation status label
        self.image_nav_status = tk.StringVar()
        self.image_nav_status.set("No images parsed")
        tk.Label(nav_frame, textvariable=self.image_nav_status, width=30).pack(side=tk.TOP, pady=5)
        
        # Add navigation buttons
        tk.Button(nav_frame, text="◀ Previous Image", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(nav_frame, text="Next Image ▶", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(nav_frame, text="Process Current Image", 
                  command=self.process_current_image).pack(side=tk.LEFT, padx=5, pady=5)
        
        nav_frame.pack(fill='x')

        # === Region Management Frame ===
        # Region size controls
        size_frame = tk.Frame(self.region_frame)
        tk.Label(size_frame, text="Region Width:").pack(side=tk.LEFT)
        self.region_width_entry = tk.Entry(size_frame, width=8)
        self.region_width_entry.insert(0, "200")
        self.region_width_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(size_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        self.region_height_entry = tk.Entry(size_frame, width=8)
        self.region_height_entry.insert(0, "200")
        self.region_height_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(size_frame, text="Cycle Delay (sec):").pack(side=tk.LEFT, padx=(10, 0))
        self.cycle_delay_entry = tk.Entry(size_frame, width=5)
        self.cycle_delay_entry.insert(0, "1.0")
        self.cycle_delay_entry.pack(side=tk.LEFT, padx=5)
        size_frame.pack(pady=5)

        # Region control buttons
        button_frame = tk.Frame(self.region_frame)
        
        # Add step back and forward buttons
        self.step_back_button = tk.Button(button_frame, text="◀ Step Back",
                                         command=self.step_back)
        self.step_back_button.pack(side=tk.LEFT, padx=5)
        
        self.step_forward_button = tk.Button(button_frame, text="Step Forward ▶",
                                            command=self.step_forward)
        self.step_forward_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Show Region",
                  command=self.show_region).pack(side=tk.LEFT, padx=5)
        self.cycle_button = tk.Button(button_frame, text="Start Cycling",
                                      command=self.toggle_cycling)
        self.cycle_button.pack(side=tk.LEFT, padx=5)
        
        # Add a new button for single iteration cycling
        self.single_cycle_button = tk.Button(button_frame, text="Cycle Once",
                                            command=self.start_single_iteration)
        self.single_cycle_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Close All Windows",
                  command=self.close_all_windows).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save All Windows",
                  command=self.save_all_windows).pack(side=tk.LEFT, padx=5)
        button_frame.pack(pady=5)

        # Pack main frames
        self.file_frame.pack(fill='x', padx=10, pady=5)
        self.input_frame.pack(fill='x', padx=10, pady=5)
        self.visualization_frame.pack(fill='x', padx=10, pady=5)
        self.image_nav_frame.pack(fill='x', padx=10, pady=5)  # Add the new navigation frame
        self.region_frame.pack(fill='x', padx=10, pady=5)

        # Add current window index for stepping
        self.current_window_index = 0

    def toggle_cycling(self):
        """Toggle the cycling of windows"""
        if not self.cycling:
            if not self.region_windows:
                messagebox.showinfo("Info", "No windows to cycle through!")
                return

            try:
                delay = float(self.cycle_delay_entry.get())
                if delay <= 0:
                    raise ValueError("Delay must be positive")
            except ValueError as e:
                messagebox.showerror("Error", "Invalid delay value. Please enter a positive number.")
                return

            self.cycling = True
            self.cycle_button.config(text="Stop Cycling")
            
            # Add a new attribute to track single iteration mode
            self.single_iteration = False
            
            self.cycle_thread = threading.Thread(target=self.cycle_windows)
            self.cycle_thread.daemon = True
            self.cycle_thread.start()
        else:
            self.cycling = False
            self.cycle_button.config(text="Start Cycling")

    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(tk.END, dir_path)

    def get_output_directory(self, image_path):
        """Get the output directory, creating it if necessary"""
        # Check if custom output directory is specified
        custom_dir = self.output_dir_entry.get().strip()
        if custom_dir:
            output_dir = custom_dir
        else:
            # Use default "Images with Indicators" in parent directory
            parent_dir = os.path.dirname(image_path)
            output_dir = os.path.join(parent_dir, "Images with Indicators")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # Modified show_region method for FindAnInstrumentApp class
    def show_region(self):
        try:
            # Get the current image path
            image_path = self.image_path_entry.get().strip()
            if not image_path or not os.path.exists(image_path):
                print("Please load an image first")
                return

            # Get the selected coordinates
            coords_text = self.coords_text.get("1.0", tk.END).strip()
            if not coords_text:
                print("Please enter coordinates first")
                return

            # Get region size
            region_width = int(self.region_width_entry.get())
            region_height = int(self.region_height_entry.get())

            # Clear previous region images and windows lists
            #self.region_images.clear()
            #self.region_windows.clear()

            # Process coordinates
            coords_lines = coords_text.split('\n')
            for i, coords_str in enumerate(coords_lines):
                if coords_str.startswith('tensor'):
                    remove = ['(', ')', '[', ']', ' ', 'tensor']
                    for rm in remove:
                        coords_str = coords_str.replace(rm, '')
                    coords = list(map(round, map(float, coords_str.split(','))))
                else:
                    coords = list(map(int, coords_str.strip('[]').split(',')))

                # Create a new CropWindow with index information
                window = CropWindow(self.root, image_path, coords, region_width, region_height, 
                                   index=len(self.region_windows), total=len(coords_lines))
                self.region_windows.append(window)

                # Store the initial cropped region for saving
                center_x = (coords[0] + coords[2]) // 2
                center_y = (coords[1] + coords[3]) // 2
                left = max(0, center_x - region_width // 2)
                top = max(0, center_y - region_height // 2)
                right = min(Image.open(image_path).width, left + region_width)
                bottom = min(Image.open(image_path).height, top + region_height)
                region = Image.open(image_path).crop((left, top, right, bottom))
                self.region_images.append(region)

            # Set the current window index to the first window
            self.current_window_index = 0

        except Exception as e:
            print(f"Error showing region: {e}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def save_all_windows(self):
        """Save all opened region windows as PNG files using their current crop state"""
        if not self.region_windows:
            messagebox.showinfo("Info", "No region windows to save!")
            return

        # Ask user for save directory
        save_dir = filedialog.askdirectory(title="Select Directory to Save Images")
        if not save_dir:  # User cancelled
            return

        # Create a base filename from the original image
        base_filename = os.path.splitext(os.path.basename(self.image_path_entry.get()))[0]

        # Save current state of each window
        saved_count = 0
        for i, window in enumerate(self.region_windows):
            if isinstance(window, CropWindow) and window.winfo_exists():
                try:
                    # Get the current cropped region from the window
                    left = max(0, window.center_x - window.current_width // 2)
                    top = max(0, window.center_y - window.current_height // 2)
                    right = min(window.full_image.width, left + window.current_width)
                    bottom = min(window.full_image.height, top + window.current_height)

                    # Crop the region with current dimensions
                    current_region = window.full_image.crop((left, top, right, bottom))

                    # Create filename with index and dimensions
                    filename = f"{base_filename}_region_{i + 1}_{window.current_width}x{window.current_height}.png"
                    filepath = os.path.join(save_dir, filename)

                    # Save the image
                    current_region.save(filepath, "PNG")
                    saved_count += 1

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save region {i + 1}: {str(e)}")

        if saved_count > 0:
            messagebox.showinfo("Success", f"Saved {saved_count} region images to {save_dir}")
        else:
            messagebox.showwarning("Warning", "No images were saved")

    def close_all_windows(self):
        """Close all opened region windows and clean up resources"""
        self.cycling = False  # Stop cycling if it's running
        if hasattr(self, 'cycle_button'):
            self.cycle_button.config(text="Start Cycling")

        # Close each window and clean up resources
        for window in self.region_windows:
            try:
                if isinstance(window, CropWindow) and window.winfo_exists():
                    # Close the window
                    window.destroy()
            except Exception as e:
                print(f"Error closing window: {e}")

        # Clear all lists
        self.region_windows.clear()
        self.region_images.clear()

    def cycle_windows(self):
        """Cycle through the windows with delay"""
        try:
            delay = float(self.cycle_delay_entry.get())
        except ValueError:
            delay = 1.0
        
        # Track if we've completed a full cycle
        completed_windows = set()

        while self.cycling and self.region_windows:
            # Reset index if we've reached the end
            if self.current_window_index >= len(self.region_windows):
                self.current_window_index = 0
                
                # If we're in single iteration mode and have seen all windows, stop cycling
                if self.single_iteration and len(completed_windows) == len(self.region_windows):
                    self.cycling = False
                    # Update button text on the main thread
                    self.root.after(0, lambda: self.cycle_button.config(text="Start Cycling"))
                    break

            # Focus the window at the current index
            self.focus_window_at_index(self.current_window_index)
            
            # Add window to completed set
            if self.current_window_index < len(self.region_windows):
                window = self.region_windows[self.current_window_index]
                completed_windows.add(id(window))
            
            # Increment the index
            self.current_window_index += 1

            # Stop cycling if no windows left
            if not self.region_windows:
                self.toggle_cycling()
                break

            time.sleep(delay)

    def reset_index(self):
        """Reset the current index to 0"""
        self.current_index = 0

    def get_label_for_box(self, index, coords, custom_labels):
        label_type = self.label_type_combo.get()

        if label_type == "Index":
            # Use the accumulated index instead of the local index
            return str(self.current_index + index + 1)
        elif label_type == "Coordinates":
            return f"({coords[0]},{coords[1]})"
        elif label_type == "Custom":
            if index < len(custom_labels):
                return custom_labels[index]
            return str(self.current_index + index + 1)
        return str(self.current_index + index + 1)


    def draw_label(self, draw, coords, label, color, font_size=20):
        try:
            # Try to load Arial font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Calculate label position (above the box)
            x = (coords[0] + coords[2]) // 2  # Center of box
            y = coords[1] - font_size - 5  # Above the box

            # Draw text with background for better visibility
            text_bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                           fill='white')
            draw.text((x, y), label, fill=color, font=font, anchor="ms")
        except Exception as e:
            print(f"Error drawing label: {e}")

    def parse_and_show_regions(self):
        """New method to parse the combined input and show regions for all coordinates"""
        combined_text = self.combined_entry.get("1.0", tk.END).strip()
        try:
            path_tensors = {}
            combined_texts = combined_text.split('\n')
            for combined_text in combined_texts:
                # Split the input at 'tensor' keyword
                if 'tensor' in combined_text:
                    parts = combined_text.split('tensor')
                    if len(parts) != 2:
                        raise ValueError("Invalid format. Expected 'path tensor[coordinates]'")
                    image_path = parts[0].strip()
                    coords_str = 'tensor' + parts[1].strip()
                    if path_tensors.get(image_path):
                        path_tensors[image_path].append(coords_str)
                    else:
                        path_tensors[image_path] = [coords_str]
                else:
                    parts = combined_text.split('[')
                    if len(parts) != 2:
                        raise ValueError("Invalid format. Expected 'path tensor[coordinates]'")
                    image_path = parts[0].strip()
                    coords_str = '[' + parts[1].strip()
                    if path_tensors.get(image_path):
                        path_tensors[image_path].append(coords_str)
                    else:
                        path_tensors[image_path] = [coords_str]

            for image_path, coords_strs in path_tensors.items():
                self.image_path_entry.delete(0, tk.END)
                self.image_path_entry.insert(tk.END, image_path)

                # Clean and set the coordinates
                coords_str = '\n'.join(coords_strs)
                self.coords_text.delete("1.0", tk.END)
                self.coords_text.insert(tk.END, coords_str)

                # Show regions for all coordinates
                self.show_region()

        except Exception as e:
            print(f"Error parsing combined input: {e}")

    def draw_crosshair(self, draw, bbox, color, thickness):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = min(x2 - x1, y2 - y1) // 3  # Size of crosshair relative to bbox

        # Vertical line
        draw.line([(center_x, center_y - size), (center_x, center_y + size)],
                  fill=color, width=thickness)
        # Horizontal line
        draw.line([(center_x - size, center_y), (center_x + size, center_y)],
                  fill=color, width=thickness)

    def draw_corner_lines(self, draw, bbox, color, thickness):
        x1, y1, x2, y2 = bbox
        # Calculate line length as a fraction of the box size
        line_length = min(x2 - x1, y2 - y1) // 3

        # Calculate 45-degree offset using line length
        offset = int(line_length / math.sqrt(2))

        # Top-left corner
        draw.line([(x1, y1), (x1 - offset, y1 - offset)], fill=color, width=thickness)
        # Top-right corner
        draw.line([(x2, y1), (x2 + offset, y1 - offset)], fill=color, width=thickness)
        # Bottom-left corner
        draw.line([(x1, y2), (x1 - offset, y2 + offset)], fill=color, width=thickness)
        # Bottom-right corner
        draw.line([(x2, y2), (x2 + offset, y2 + offset)], fill=color, width=thickness)

    def draw_circumscribed_circle(self, draw, bbox, color, thickness):
        x1, y1, x2, y2 = bbox
        # Calculate center of the box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate radius based on the diagonal of the box
        # This ensures the circle will fully contain the box
        width = x2 - x1
        height = y2 - y1
        radius = math.sqrt(width ** 2 + height ** 2) / 2

        # Calculate the circle's bounding box
        circle_x1 = center_x - radius
        circle_y1 = center_y - radius
        circle_x2 = center_x + radius
        circle_y2 = center_y + radius

        draw.ellipse([circle_x1, circle_y1, circle_x2, circle_y2],
                     outline=color, width=thickness)

    def parse_combined_input(self):
        """Parse the combined input and store image paths and coordinates for navigation"""
        self.reset_index()
        combined_text = self.combined_entry.get("1.0", tk.END).strip()
        
        # Clear previous parsed data
        self.parsed_images = []
        self.current_image_index = 0
        
        try:
            path_tensors = {}
            combined_texts = combined_text.split('\n')
            for combined_text in combined_texts:
                # Split the input at 'tensor' keyword
                if 'tensor' in combined_text:
                    parts = combined_text.split('tensor')
                    if len(parts) != 2:
                        raise ValueError("Invalid format. Expected 'path tensor[coordinates]'")
                    image_path = parts[0].strip()
                    coords_str = 'tensor' + parts[1].strip()
                    if path_tensors.get(image_path):
                        path_tensors[image_path].append(coords_str)
                    else:
                        path_tensors[image_path] = [coords_str]
                else:
                    parts = combined_text.split('[')
                    if len(parts) != 2:
                        raise ValueError("Invalid format. Expected 'path tensor[coordinates]'")
                    image_path = parts[0].strip()
                    coords_str = '[' + parts[1].strip()
                    if path_tensors.get(image_path):
                        path_tensors[image_path].append(coords_str)
                    else:
                        path_tensors[image_path] = [coords_str]

            # Store parsed data for navigation
            for image_path, coords_strs in path_tensors.items():
                self.parsed_images.append([image_path, coords_strs])
                
            # If we have parsed images, display the first one
            if self.parsed_images:
                self.current_image_index = 0
                self.update_current_image_display()
                messagebox.showinfo("Success", f"Parsed {len(self.parsed_images)} images. Use navigation buttons to browse and process them.")
            else:
                messagebox.showinfo("Info", "No valid image data found in the input.")

        except Exception as e:
            messagebox.showerror("Error", f"Error parsing combined input: {str(e)}")

    def browse_image(self):
        self.image_path_entry.delete(0, tk.END)
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        self.image_path_entry.insert(tk.END, file_path)

    def draw_indicators(self):
        try:
            image_path = self.image_path_entry.get().strip()
            coords_lines = self.coords_text.get("1.0", tk.END).strip().split('\n')
            custom_labels = self.custom_labels_text.get("1.0", tk.END).strip().split('\n')
            color = self.color_combobox.get()
            thickness = int(self.thickness_entry.get())
            indicator_type = self.indicator_combobox.get()

            if not image_path:
                messagebox.showerror("Error", "Please select an image file.")
                return

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            for i, coords_str in enumerate(coords_lines):
                if coords_str.startswith('tensor'):
                    remove = ['(', ')', '[', ']', ' ', 'tensor']
                    for rm in remove:
                        coords_str = coords_str.replace(rm, '')
                    coords = list(map(round, map(float, coords_str.split(','))))
                else:
                    coords = list(map(int, coords_str.strip('[]').split(',')))

                if indicator_type == "Square":
                    draw.rectangle(coords, outline=color, width=thickness)
                elif indicator_type == "Circle":
                    self.draw_circumscribed_circle(draw, coords, color, thickness)
                elif indicator_type == "Crosshair":
                    self.draw_crosshair(draw, coords, color, thickness)
                elif indicator_type == "Corner Lines":
                    self.draw_corner_lines(draw, coords, color, thickness)

                # Draw label for the box
                label = self.get_label_for_box(i, coords, custom_labels)
                self.draw_label(draw, coords, label, color)

            # Get output directory and save the image
            output_dir = self.get_output_directory(image_path)
            filename = os.path.basename(image_path)
            modified_image_path = os.path.join(output_dir, filename)

            img.save(modified_image_path)
            
            # Close any previously opened image
            self.close_last_opened_image()
            
            # Open the new image and store a reference to it
            self.last_opened_image = modified_image_path
            import subprocess
            self.last_opened_image_process = subprocess.Popen(["start", "", modified_image_path], 
                                                            shell=True)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def paste_clipboard(self):
        """Paste clipboard contents into the combined input area"""
        try:
            # Get clipboard content
            clipboard_content = self.root.clipboard_get()
            self.combined_entry.delete(1.0, tk.END)
            # Insert at current cursor position or replace selected text
            self.combined_entry.insert(tk.INSERT, clipboard_content)
        except tk.TclError:
            # This happens when clipboard is empty or contains non-text data
            messagebox.showinfo("Clipboard Empty", "No text found in clipboard.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to paste from clipboard: {str(e)}")

    def start_single_iteration(self):
        """Start cycling through windows once and then stop"""
        if not self.cycling:
            if not self.region_windows:
                messagebox.showinfo("Info", "No windows to cycle through!")
                return

            try:
                delay = float(self.cycle_delay_entry.get())
                if delay <= 0:
                    raise ValueError("Delay must be positive")
            except ValueError as e:
                messagebox.showerror("Error", "Invalid delay value. Please enter a positive number.")
                return

            self.cycling = True
            self.single_iteration = True  # Set single iteration mode
            self.cycle_button.config(text="Stop Cycling")
            
            self.cycle_thread = threading.Thread(target=self.cycle_windows)
            self.cycle_thread.daemon = True
            self.cycle_thread.start()
        else:
            self.cycling = False
            self.cycle_button.config(text="Start Cycling")

    def step_forward(self):
        """Step forward to the next window"""
        if not self.region_windows:
            messagebox.showinfo("Info", "No windows to navigate!")
            return
        
        # Increment the index, wrapping around if necessary
        self.current_window_index = (self.current_window_index + 1) % len(self.region_windows)
        self.focus_window_at_index(self.current_window_index)

    def step_back(self):
        """Step back to the previous window"""
        if not self.region_windows:
            messagebox.showinfo("Info", "No windows to navigate!")
            return
        
        # Decrement the index, wrapping around if necessary
        self.current_window_index = (self.current_window_index - 1) % len(self.region_windows)
        self.focus_window_at_index(self.current_window_index)

    def focus_window_at_index(self, index):
        """Focus the window at the given index"""
        if 0 <= index < len(self.region_windows):
            window = self.region_windows[index]
            if isinstance(window, CropWindow) and window.winfo_exists():
                try:
                    window.lift()
                    window.focus_force()
                    # Update window title to show current position
                    filename = os.path.basename(window.image_path)
                    window.title(f"{filename} - Region at {window.coords} [{index+1}/{len(self.region_windows)}]")
                except Exception as e:
                    print(f"Error focusing window: {e}")
                    # Remove invalid window
                    self.region_windows.pop(index)
                    # Adjust current index if needed
                    if self.current_window_index >= len(self.region_windows):
                        self.current_window_index = max(0, len(self.region_windows) - 1)
            else:
                # Remove destroyed or invalid window
                self.region_windows.pop(index)
                # Adjust current index if needed
                if self.current_window_index >= len(self.region_windows):
                    self.current_window_index = max(0, len(self.region_windows) - 1)

    # Add new methods for image navigation
    def previous_image(self):
        """Navigate to the previous image in the parsed list"""
        if not self.parsed_images:
            messagebox.showinfo("Info", "No images parsed. Please parse combined input first.")
            return
            
        # Close any open region windows
        self.close_all_windows()
        
        # Close any previously opened image
        self.close_last_opened_image()
            
        # Decrement the index, wrapping around if necessary
        self.current_image_index = (self.current_image_index - 1) % len(self.parsed_images)
        self.update_current_image_display()
        
    def next_image(self):
        """Navigate to the next image in the parsed list"""
        if not self.parsed_images:
            messagebox.showinfo("Info", "No images parsed. Please parse combined input first.")
            return
            
        # Close any open region windows
        self.close_all_windows()
        
        # Close any previously opened image
        self.close_last_opened_image()
            
        # Increment the index, wrapping around if necessary
        self.current_image_index = (self.current_image_index + 1) % len(self.parsed_images)
        self.update_current_image_display()
        
    def update_current_image_display(self):
        """Update the UI to display the current image and its coordinates"""
        if not self.parsed_images or self.current_image_index >= len(self.parsed_images):
            return
            
        # Get current image data
        image_path, coords_strs = self.parsed_images[self.current_image_index]
        
        # Update image path entry
        self.image_path_entry.delete(0, tk.END)
        self.image_path_entry.insert(tk.END, image_path)
        
        # Update coordinates text
        self.coords_text.delete("1.0", tk.END)
        self.coords_text.insert(tk.END, '\n'.join(coords_strs))
        
        # Update navigation status
        self.image_nav_status.set(f"Image {self.current_image_index + 1} of {len(self.parsed_images)}: {os.path.basename(image_path)}")
        
    def process_current_image(self):
        """Process the currently displayed image (draw indicators or show regions)"""
        if not self.parsed_images:
            messagebox.showinfo("Info", "No images parsed. Please parse combined input first.")
            return
            
        # Close any previously opened image
        self.close_last_opened_image()
            
        # Draw indicators for the current image
        self.draw_indicators()

    def close_last_opened_image(self):
        """Close the last opened image file if it exists"""
        if self.last_opened_image_process is not None:
            try:
                # Try to terminate the process
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    # On Windows, use taskkill to close the image viewer
                    subprocess.run(["taskkill", "/F", "/PID", str(self.last_opened_image_process.pid)], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # On other platforms, try to terminate the process
                    self.last_opened_image_process.terminate()
                    
                self.last_opened_image_process = None
            except Exception as e:
                print(f"Error closing last image: {e}")
                
        # Also clear the reference to the last opened image path
        self.last_opened_image = None

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = FindAnInstrumentApp(root)
    root.mainloop()