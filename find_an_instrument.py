import os.path
import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
from os import startfile
import math
from PIL import Image, ImageDraw, ImageTk, ImageFont
from tkinter import filedialog, colorchooser, ttk, messagebox


class FindAnInstrumentApp:
    def __init__(self, root, img_path=''):
        self.root = root
        root.title("FAIA (find an instrument app)")

        # Initialize tracking variables
        self.current_index = 0
        self.current_image = None
        self.region_windows = []  # List to keep track of region windows
        self.region_images = []
        # Create main frames for better organization
        self.input_frame = tk.Frame(root)
        self.controls_frame = tk.Frame(root)
        self.options_frame = tk.Frame(root)

        # === Input Frame ===
        # Combined input section
        tk.Label(self.input_frame, text="Combined Path and Tensor:").pack(anchor='w', padx=10, pady=(10, 0))
        self.combined_entry = tk.Text(self.input_frame, height=5, width=70)
        self.combined_entry.pack(fill='x', padx=10, pady=5)

        # Parse buttons frame
        parse_buttons_frame = tk.Frame(self.input_frame)
        tk.Button(parse_buttons_frame, text="Parse Input and Draw Indicators",
                  command=self.parse_combined_input).pack(side=tk.LEFT, padx=5)
        tk.Button(parse_buttons_frame, text="Parse Input and See Regions",
                  command=self.parse_and_show_regions).pack(side=tk.LEFT, padx=5)
        #tk.Button(parse_buttons_frame, text="Reset Index",
        #          command=self.reset_index).pack(side=tk.LEFT, padx=5)
        parse_buttons_frame.pack(pady=10)

        # Image path section
        path_frame = tk.Frame(self.input_frame)
        tk.Label(path_frame, text="Image Path:").pack(side=tk.LEFT, padx=(10, 5))
        self.image_path_entry = tk.Entry(path_frame, width=50)
        self.image_path_entry.pack(side=tk.LEFT, padx=5)
        self.image_path_entry.insert(tk.END, img_path)
        tk.Button(path_frame, text="Browse",
                  command=self.browse_image).pack(side=tk.LEFT, padx=5)
        path_frame.pack(fill='x', pady=10)

        # Output directory section
        output_frame = tk.Frame(self.input_frame)
        tk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=(10, 5))
        self.output_dir_entry = tk.Entry(output_frame, width=50)
        self.output_dir_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(output_frame, text="Browse",
                  command=self.browse_output_dir).pack(side=tk.LEFT, padx=5)
        output_frame.pack(fill='x', pady=10)

        # === Controls Frame ===
        # Coordinates and labels section
        coords_frame = tk.Frame(self.controls_frame)

        # Coordinates input
        coords_left_frame = tk.Frame(coords_frame)
        tk.Label(coords_left_frame, text="Coordinates ([x1, y1, x2, y2], one per line):").pack(anchor='w')
        self.coords_text = tk.Text(coords_left_frame, height=10, width=30)
        self.coords_text.pack(pady=5)
        coords_left_frame.pack(side=tk.LEFT, padx=10)

        # Label options
        label_frame = tk.Frame(coords_frame)
        tk.Label(label_frame, text="Label Type:").pack(anchor='w')
        self.label_type_combo = tk.StringVar()
        self.label_type_combobox = ttk.Combobox(label_frame, textvariable=self.label_type_combo,
                                                values=["Index", "Coordinates", "Custom"],
                                                state='readonly')
        self.label_type_combobox.pack(pady=5)
        self.label_type_combo.set("Index")

        tk.Label(label_frame, text="Custom Labels (one per line):").pack(anchor='w', pady=(10, 0))
        self.custom_labels_text = tk.Text(label_frame, height=10, width=20)
        self.custom_labels_text.pack(pady=5)
        label_frame.pack(side=tk.LEFT, padx=10)

        coords_frame.pack(pady=10)

        # === Options Frame ===
        # Visual options
        options_top_frame = tk.Frame(self.options_frame)

        # Color selection
        color_frame = tk.Frame(options_top_frame)
        tk.Label(color_frame, text="Color:").pack(side=tk.LEFT, padx=5)
        self.color_combo = tk.StringVar()
        self.color_combobox = ttk.Combobox(color_frame, textvariable=self.color_combo,
                                           values=["Red", "HotPink", "Green", "LawnGreen", "LimeGreen",
                                                   "GreenYellow", "Blue", "Orange", "Yellow"],
                                           state='readonly', width=15)
        self.color_combobox.pack(side=tk.LEFT)
        self.color_combo.set("Red")
        color_frame.pack(side=tk.LEFT, padx=10)

        # Indicator type selection
        indicator_frame = tk.Frame(options_top_frame)
        tk.Label(indicator_frame, text="Indicator Type:").pack(side=tk.LEFT, padx=5)
        self.indicator_combo = tk.StringVar()
        self.indicator_combobox = ttk.Combobox(indicator_frame, textvariable=self.indicator_combo,
                                               values=["Square", "Circle", "Crosshair", "Corner Lines"],
                                               state='readonly', width=15)
        self.indicator_combobox.pack(side=tk.LEFT)
        self.indicator_combo.set("Square")
        indicator_frame.pack(side=tk.LEFT, padx=10)

        # Line thickness and Draw button frame
        thickness_frame = tk.Frame(options_top_frame)
        tk.Label(thickness_frame, text="Line Thickness:").pack(side=tk.LEFT, padx=5)
        self.thickness_entry = tk.Entry(thickness_frame, width=5)
        self.thickness_entry.insert(0, '5')
        self.thickness_entry.pack(side=tk.LEFT)

        # Draw button immediately after line thickness
        tk.Button(thickness_frame, text="Draw Indicators",
                  command=self.draw_indicators).pack(side=tk.LEFT, padx=20)

        thickness_frame.pack(side=tk.LEFT, padx=10)
        options_top_frame.pack(pady=10)

        # Region options
        region_frame = tk.Frame(self.options_frame)
        tk.Label(region_frame, text="Region Width:").pack(side=tk.LEFT, padx=5)
        self.region_width_entry = tk.Entry(region_frame, width=10)
        self.region_width_entry.insert(0, "200")
        self.region_width_entry.pack(side=tk.LEFT)

        tk.Label(region_frame, text="Height:").pack(side=tk.LEFT, padx=5)
        self.region_height_entry = tk.Entry(region_frame, width=10)
        self.region_height_entry.insert(0, "200")
        self.region_height_entry.pack(side=tk.LEFT)

        tk.Button(region_frame, text="See Region",
                  command=self.show_region).pack(side=tk.LEFT, padx=20)

        # Add Close All Windows button
        tk.Button(region_frame, text="Close All Windows",
                  command=self.close_all_windows).pack(side=tk.LEFT, padx=20)
        # Add Save All Windows button
        tk.Button(region_frame, text="Save All Windows",
                  command=self.save_all_windows).pack(side=tk.LEFT, padx=20)
        region_frame.pack(pady=10)

        # Pack main frames
        self.input_frame.pack(fill='x', padx=10, pady=5)
        self.controls_frame.pack(fill='x', padx=10, pady=5)
        self.options_frame.pack(fill='x', padx=10, pady=5)


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

    def save_all_windows(self):
        """Save all opened region windows as PNG files"""
        if not self.region_windows or not self.region_images:
            tk.messagebox.showinfo("Info", "No region windows to save!")
            return

        # Ask user for save directory
        save_dir = filedialog.askdirectory(title="Select Directory to Save Images")
        if not save_dir:  # User cancelled
            return

        # Create a base filename from the original image
        base_filename = os.path.splitext(os.path.basename(self.image_path_entry.get()))[0]

        # Save each region image
        for i, image in enumerate(self.region_images):
            if image:
                # Create filename with index
                filename = f"{base_filename}_region_{i + 1}.png"
                filepath = os.path.join(save_dir, filename)

                # Save the image
                try:
                    image.save(filepath, "PNG")
                except Exception as e:
                    tk.messagebox.showerror("Error", f"Failed to save region {i + 1}: {str(e)}")

        tk.messagebox.showinfo("Success", f"Saved {len(self.region_images)} region images to {save_dir}")

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

            # Open the image
            img = Image.open(image_path)

            # Clear previous region images
            self.region_images.clear()

            # Process coordinates
            coords_lines = coords_text.split('\n')
            for coords_str in coords_lines:
                if coords_str.startswith('tensor'):
                    remove = ['(', ')', '[', ']', ' ', 'tensor']
                    for rm in remove:
                        coords_str = coords_str.replace(rm, '')
                    coords = list(map(round, map(float, coords_str.split(','))))
                else:
                    coords = list(map(int, coords_str.strip('[]').split(',')))

                # Calculate the center of the bounding box
                center_x = (coords[0] + coords[2]) // 2
                center_y = (coords[1] + coords[3]) // 2

                # Calculate crop coordinates
                left = max(0, center_x - region_width // 2)
                top = max(0, center_y - region_height // 2)
                right = min(img.width, left + region_width)
                bottom = min(img.height, top + region_height)

                # Crop the region
                region = img.crop((left, top, right, bottom))
                self.region_images.append(region)  # Store the region image

                # Create a new window for this region
                region_window = tk.Toplevel(self.root)
                self.region_windows.append(region_window)
                filename = os.path.basename(image_path)
                region_window.title(f"{filename} - Region at {coords}")

                # Convert PIL image to PhotoImage
                photo = ImageTk.PhotoImage(region)

                # Create label and display image
                label = tk.Label(region_window, image=photo)
                label.image = photo  # Keep a reference!
                label.pack()

        except Exception as e:
            print(f"Error showing region: {e}")

    def close_all_windows(self):
        """Close all opened region windows"""
        for window in self.region_windows:
            if window.winfo_exists():  # Check if window still exists
                window.destroy()
        self.region_windows.clear()  # Clear the list after closing all windows
        self.region_images.clear()  # Clear the stored images

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
        self.reset_index()
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

                # Draw indicators
                self.draw_indicators()

                # Update the current_index based on the number of coordinates processed
                self.current_index += len(coords_strs)

        except Exception as e:
            print(f"Error parsing combined input: {e}")
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
            startfile(modified_image_path)


        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = FindAnInstrumentApp(root)
    root.mainloop()