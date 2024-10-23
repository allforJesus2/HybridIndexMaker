import os.path
import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
from os import startfile
import math
from PIL import Image, ImageDraw, ImageTk, ImageFont


class FindAnInstrumentApp:
    def __init__(self, root, img_path=''):
        self.root = root
        root.title("FAIA (find an instrument app)")

        # Add current_index to track accumulated index across images
        self.current_index = 0

        # Store the current image
        self.current_image = None

        # Combined input section
        self.combined_label = tk.Label(root, text="Combined Path and Tensor:")
        self.combined_entry = tk.Text(root, height=5, width=70)

        # Create a frame for the parse buttons
        self.parse_buttons_frame = tk.Frame(root)
        self.parse_button = tk.Button(self.parse_buttons_frame, text="Parse Input and Draw Indicators", command=self.parse_combined_input)
        self.parse_regions_button = tk.Button(self.parse_buttons_frame, text="Parse Input and See Regions",
                                              command=self.parse_and_show_regions)

        # Add reset index button
        self.reset_index_button = tk.Button(self.parse_buttons_frame, text="Reset Index", command=self.reset_index)

        # Original widgets
        self.image_path_label = tk.Label(root, text="Image Path:")
        self.image_path_entry = tk.Entry(root, width=50)
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_image)

        self.coords_label = tk.Label(root, text="Coordinates ([x1, y1, x2, y2], one per line):")
        self.coords_text = tk.Text(root, height=10, width=30)

        # Add label type selection
        self.label_type_label = tk.Label(root, text="Label Type:")
        self.label_type_combo = tk.StringVar()
        self.label_type_combobox = ttk.Combobox(root, textvariable=self.label_type_combo,
                                                values=["Index", "Coordinates", "Custom"],
                                                state='readonly')
        self.label_type_combobox.current(0)

        # Add custom label input
        self.custom_labels_label = tk.Label(root, text="Custom Labels (one per line):")
        self.custom_labels_text = tk.Text(root, height=10, width=20)

        self.color_label = tk.Label(root, text="Color:")
        self.color_combo = tk.StringVar()
        self.color_combobox = ttk.Combobox(root, textvariable=self.color_combo,
                                           values=["Red", "HotPink", "Green", "LawnGreen", "LimeGreen", "GreenYellow",
                                                   "Blue", "Orange", "Yellow"],
                                           state='readonly')
        self.color_combobox.current(0)

        # Region size inputs
        self.region_frame = tk.Frame(root)
        self.region_width_label = tk.Label(self.region_frame, text="Region Width:")
        self.region_width_entry = tk.Entry(self.region_frame, width=10)
        self.region_width_entry.insert(0, "200")
        self.region_height_label = tk.Label(self.region_frame, text="Height:")
        self.region_height_entry = tk.Entry(self.region_frame, width=10)
        self.region_height_entry.insert(0, "200")

        # See Region button
        self.see_region_button = tk.Button(self.region_frame, text="See Region", command=self.show_region)

        # Indicator type dropdown
        self.indicator_label = tk.Label(root, text="Indicator Type:")
        self.indicator_combo = tk.StringVar()
        self.indicator_combobox = ttk.Combobox(root, textvariable=self.indicator_combo,
                                               values=["Square", "Circle", "Crosshair", "Corner Lines"],
                                               state='readonly')
        self.indicator_combobox.current(0)

        self.thickness_label = tk.Label(root, text="Line Thickness:")
        self.thickness_entry = tk.Entry(root, width=10)
        self.thickness_entry.insert(tk.END, '5')

        self.apply_button = tk.Button(root, text="Draw Indicators", command=self.draw_indicators)

        # Layout - Combined input section
        self.combined_label.grid(row=0, column=0, padx=10, pady=10)
        self.combined_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=10)

        # Layout - Parse buttons
        self.parse_buttons_frame.grid(row=0, column=3, padx=10, pady=10)
        self.parse_button.pack(side=tk.LEFT, padx=5)
        self.parse_regions_button.pack(side=tk.LEFT, padx=5)
        self.reset_index_button.pack(side=tk.LEFT, padx=5)  # Add reset button to layout

        # Rest of the layout remains the same...
        self.image_path_label.grid(row=1, column=0, padx=10, pady=10)
        self.image_path_entry.grid(row=1, column=1, padx=10, pady=10)
        self.browse_button.grid(row=1, column=2, padx=10, pady=10)
        self.image_path_entry.insert(tk.END, img_path)

        self.coords_label.grid(row=2, column=0, padx=10, pady=10)
        self.coords_text.grid(row=2, column=1, rowspan=3, padx=10, pady=10)

        self.label_type_label.grid(row=2, column=2, padx=10, pady=10)
        self.label_type_combobox.grid(row=2, column=3, padx=10, pady=10)

        self.custom_labels_label.grid(row=3, column=2, padx=10, pady=10)
        self.custom_labels_text.grid(row=3, column=3, rowspan=2, padx=10, pady=10)

        self.color_label.grid(row=5, column=0, padx=10, pady=10)
        self.color_combobox.grid(row=5, column=1, padx=10, pady=10)

        self.region_frame.grid(row=6, column=0, columnspan=4, pady=10)
        self.region_width_label.pack(side=tk.LEFT, padx=5)
        self.region_width_entry.pack(side=tk.LEFT, padx=5)
        self.region_height_label.pack(side=tk.LEFT, padx=5)
        self.region_height_entry.pack(side=tk.LEFT, padx=5)
        self.see_region_button.pack(side=tk.LEFT, padx=20)

        self.indicator_label.grid(row=7, column=0, padx=10, pady=10)
        self.indicator_combobox.grid(row=7, column=1, padx=10, pady=10)

        self.thickness_label.grid(row=8, column=0, padx=10, pady=10)
        self.thickness_entry.grid(row=8, column=1, padx=10, pady=10)

        self.apply_button.grid(row=9, columnspan=4, pady=20)

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

                # Create a new window for this region
                region_window = tk.Toplevel(self.root)
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

            modified_image = os.path.basename(image_path)
            # Save and display modified image
            img.save(modified_image)
            startfile(modified_image)
        except Exception as e:
            print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = FindAnInstrumentApp(root)
    root.mainloop()