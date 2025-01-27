import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
try:
    from detecto_gui.generate_random_training_data import *
except:
    from generate_random_training_data import *


def label_to_entry(label):
    # Get the base name of the label without the colon
    base_name = label.split(':')[0]

    # Convert the base name to lowercase, replace spaces with underscores, remove parentheses, and replace hyphens with underscores
    attribute_name = re.sub(r'[^a-z0-9_]+', '_', base_name.lower()).strip('_')

    return attribute_name + "_entry"


class SplatImagesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Splat Images Parameters")
        self.root.geometry("800x900")
        self.entries = {}
        self.checkbuttons = {}
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Style configuration
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10))

        # Folder selection
        self.create_folder_section(main_frame)

        # Image parameters
        self.create_image_params_section(main_frame)

        # Color parameters
        self.create_color_params_section(main_frame)

        # Kernel patch parameters
        self.create_kernel_patch_section(main_frame)

        # Transformation parameters
        self.create_transform_params_section(main_frame)

        # Buttons
        self.create_buttons(main_frame)

    def create_kernel_patch_section(self, parent):
        patch_frame = ttk.LabelFrame(parent, text="Kernel Patch Parameters", padding="10")
        patch_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        params = [
            ("Min Kernel Patches:", "0"),
            ("Max Kernel Patches:", "5"),
        ]

        for i, (label, default) in enumerate(params):
            setattr(self,
                    label_to_entry(label),
                    self.labeled_entry(patch_frame, label, default, i))

    def create_folder_section(self, parent):
        folder_frame = ttk.LabelFrame(parent, text="Folder Selection", padding="10")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.kernel_folder_entry = self.labeled_entry_with_browse(folder_frame, "Kernel Folder:", 0)
        self.kernel_patches_folder_entry = self.labeled_entry_with_browse(folder_frame, "Kernel Patches Folder:", 1)
        self.not_kernel_folder_entry = self.labeled_entry_with_browse(folder_frame, "NOT Kernel Folder:", 2)
        self.output_folder_entry = self.labeled_entry_with_browse(folder_frame, "Output Folder:", 3)

    def create_image_params_section(self, parent):
        image_frame = ttk.LabelFrame(parent, text="Image Parameters", padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        params = [
            ("Number of Output Images:", "10"),
            ("Splats per Image:", "100"),
            ("Low Variance:", "0.1"),
            ("Validation Split:", "0.1"),  # New parameter

        ]

        for i, (label, default) in enumerate(params):
            setattr(self,
                    label_to_entry(label),
                    self.labeled_entry(image_frame, label, default, i))

    def create_color_params_section(self, parent):
        color_frame = ttk.LabelFrame(parent, text="Background Canvas Parameters", padding="10")
        color_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        params = [
            ("Output tile size (px):", "1300"),
            ("Brightness Range Min:", "0.5"), ("Brightness Range Max:", "1.5"),
            ("Contrast Range Min:", "0.5"), ("Contrast Range Max:", "1.5"),
            ("Hue Range Min:", "0"), ("Hue Range Max:", "180"),
            ("Invert Probability:", "0.0"),
            ("Blur Probability:", "0.2"),
            ("Blur Max:", "1"),

        ]

        for i, (label, default) in enumerate(params):

            # Set the attribute on self with the formatted name
            setattr(self, label_to_entry(label),
                    self.labeled_entry(color_frame, label, default, i))

    def create_transform_params_section(self, parent):
        transform_frame = ttk.LabelFrame(parent, text="Transformation Parameters", padding="10")
        transform_frame.grid(row=2, column=0, columnspan=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        params = [
            ("NOT Weight:", "0.3"),
            ("Horizontal Flip Probability:", "0.5"),
            ("Vertical Flip Probability:", "0.5"),
            ("Rotate Probability:","0.5"),
            ("Black Threshold (0-255):", "200"),
            ("Color Probability:", "0.0"),
            ("Min Scale:", "0.9"),
            ("Max Scale:", "1.1"),
            ("Bbox Scale Min:", "1.0"),
            ("Bbox Scale Max:", "1.0")
        ]

        for i, (label, default) in enumerate(params):
            # Set the attribute on self with the formatted name
            setattr(self, label_to_entry(label),
                    self.labeled_entry(transform_frame, label, default, i))

        self.uniform_scale_var = tk.BooleanVar()
        uniform_scale_cb = ttk.Checkbutton(transform_frame, text="Uniform Scale", variable=self.uniform_scale_var)
        uniform_scale_cb.grid(row=len(params), column=0, sticky=tk.W)
        self.checkbuttons['uniform_scale'] = self.uniform_scale_var

        self.yolo_var = tk.BooleanVar()
        yolo_cb = ttk.Checkbutton(transform_frame, text="YOLO Format", variable=self.yolo_var)
        yolo_cb.grid(row=len(params), column=1, sticky=tk.W)
        self.checkbuttons['yolo'] = self.yolo_var

    def create_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.E, tk.W), pady=10)

        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Load Settings", command=self.load_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Delete Output Files", command=self.delete_output_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Generate", command=self.on_generate).pack(side=tk.RIGHT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT)

    def delete_output_files(self):
        if not self.output_folder_entry.get():
            messagebox.showerror("Error", "Please select an output folder first")
            return

        if messagebox.askyesno("Confirm Delete",
                               "Are you sure you want to delete all PNG and XML files in the output folder and Validation subfolder?"):
            output_dir = self.output_folder_entry.get()
            validation_dir = os.path.join(output_dir, "Validation")
            deleted_count = 0

            # Delete from main output directory
            for file in os.listdir(output_dir):
                if file.lower().endswith(('.png', '.xml')):
                    os.remove(os.path.join(output_dir, file))
                    deleted_count += 1

            # Delete from Validation subfolder if it exists
            if os.path.exists(validation_dir):
                for file in os.listdir(validation_dir):
                    if file.lower().endswith(('.png', '.xml')):
                        os.remove(os.path.join(validation_dir, file))
                        deleted_count += 1

            messagebox.showinfo("Complete", f"Deleted {deleted_count} files")

    def labeled_entry(self, parent, label_text, default_value, row, width=20):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
        entry = ttk.Entry(parent, width=width)
        entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        entry.insert(0, default_value)
        entry_name = label_to_entry(label_text)
        self.entries[entry_name] = entry
        return entry


    def labeled_entry_with_browse(self, parent, label_text, row, width=50):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
        entry = ttk.Entry(parent, width=width)
        entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Button(parent, text="Browse", command=lambda: self.select_folder(entry)).grid(row=row, column=2, padx=(5, 0), pady=2)
        entry_name = label_to_entry(label_text)
        self.entries[entry_name] = entry
        return entry

    def select_folder(self, entry):
        folder = filedialog.askdirectory(title="Select folder")
        if folder:
            entry.delete(0, tk.END)
            entry.insert(0, folder)


    def get_current_settings(self):
        settings = {}
        for name, entry in self.entries.items():
            settings[name] = entry.get()
        for name, var in self.checkbuttons.items():
            settings[name] = var.get()
        return settings

    def apply_settings(self, settings):
        for name, value in settings.items():
            if name in self.entries:
                self.entries[name].delete(0, tk.END)
                self.entries[name].insert(0, value)
            elif name in self.checkbuttons:
                self.checkbuttons[name].set(value)

    def save_settings(self):
        settings = self.get_current_settings()
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if filename:
            with open(filename, "w") as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Settings Saved", f"Settings saved to {filename}")

    def load_settings(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                with open(filename, "r") as f:
                    settings = json.load(f)
                self.apply_settings(settings)
            except Exception as e:
                messagebox.showerror("Error Loading Settings", str(e))

    def on_generate(self):
        settings = self.get_current_settings()
        try:
            # Convert settings to appropriate types
            count = int(settings['number_of_output_images_entry'])
            splats_per_image = int(settings['splats_per_image_entry'])
            not_weight = float(settings['not_weight_entry'])
            yolo = self.yolo_var.get()
            horizontal_flip_p = float(settings['horizontal_flip_probability_entry'])
            vertical_flip_p = float(settings['vertical_flip_probability_entry'])
            min_scale = float(settings['min_scale_entry'])
            max_scale = float(settings['max_scale_entry'])
            uniform_scale = self.uniform_scale_var.get()
            output_size = int(settings['output_tile_size_px_entry'])
            black_threshold = int(settings['black_threshold_0_255_entry'])
            brightness_range_min = float(settings['brightness_range_min_entry'])
            brightness_range_max = float(settings['brightness_range_max_entry'])
            contrast_range_min = float(settings['contrast_range_min_entry'])
            contrast_range_max = float(settings['contrast_range_max_entry'])
            hue_range_min = int(settings['hue_range_min_entry'])
            hue_range_max = int(settings['hue_range_max_entry'])
            invert_probability = float(settings['invert_probability_entry'])
            blur_probability = float(settings['blur_probability_entry'])
            p_rotate = float(settings['rotate_probability_entry'])
            blur_max = int(settings['blur_max_entry'])
            color_probability = float(settings['color_probability_entry'])
            low_variance = float(settings['low_variance_entry'])
            validation_split = float(settings['validation_split_entry'])
            bbox_scale_min = float(settings['bbox_scale_min_entry'])
            bbox_scale_max = float(settings['bbox_scale_max_entry'])
            min_kernel_patches = int(settings['min_kernel_patches_entry'])
            max_kernel_patches = int(settings['max_kernel_patches_entry'])

            # Create progress bar window
            self.progress_window = tk.Toplevel(self.root)
            self.progress_window.title("Generating Images")
            self.progress_window.geometry("300x150")

            # Make the progress window modal
            self.progress_window.transient(self.root)
            self.progress_window.grab_set()

            # Progress label
            self.progress_label = ttk.Label(self.progress_window, text="Generating images...\n0 of {count}")
            self.progress_label.pack(pady=10)

            # Progress bar
            self.progress_bar = ttk.Progressbar(
                self.progress_window,
                length=200,
                mode='determinate'
            )
            self.progress_bar.pack(pady=10)

            # Cancel button
            self.cancel_button = ttk.Button(
                self.progress_window,
                text="Cancel",
                command=lambda: setattr(self, 'generation_cancelled', True)
            )
            self.cancel_button.pack(pady=10)

            # Initialize cancel flag
            self.generation_cancelled = False

            # Create Validation subfolder
            of = settings['output_folder_entry']
            validation_folder = os.path.join(of, "Validation")
            os.makedirs(validation_folder, exist_ok=True)

            # Generate images
            for i in range(count):
                # Check if generation was cancelled
                if self.generation_cancelled:
                    self.progress_window.destroy()
                    messagebox.showinfo('Cancelled', 'Image generation was cancelled')
                    return

                # Update progress
                percentage = ((i + 1) / count) * 100
                self.progress_bar['value'] = percentage
                self.progress_label.config(text=f"Generating images...\n{i + 1} of {count}")
                self.progress_window.update()

                # Determine if this image goes in validation set
                is_validation = i < int(count * validation_split)
                current_output_folder = validation_folder if is_validation else of

                generate_output_image(
                    settings['kernel_folder_entry'],
                    current_output_folder,
                    not_folder=settings['not_kernel_folder_entry'],
                    kernel_patches_folder=settings['kernel_patches_folder_entry'],
                    output_size=output_size,
                    splats_per_image=splats_per_image,
                    yolo=yolo,
                    not_weight=not_weight,
                    p_mirror_x=horizontal_flip_p,
                    p_mirror_y=vertical_flip_p,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    uniform_scale=uniform_scale,
                    black_threshold=black_threshold,
                    color_probability=color_probability,
                    low_variance=low_variance,
                    brightness_range=(brightness_range_min, brightness_range_max),
                    contrast_range=(contrast_range_min, contrast_range_max),
                    hue_range=(hue_range_min, hue_range_max),
                    invert_probability=invert_probability,
                    bbox_scale_min=bbox_scale_min,
                    bbox_scale_max=bbox_scale_max,
                    blur_probability=blur_probability,
                    blur_max=blur_max,
                    p_rotate=p_rotate,
                    min_kernel_patches=min_kernel_patches,
                    max_kernel_patches=max_kernel_patches
                )

            # Close progress window
            self.progress_window.destroy()

            # Show completion message and open output folder
            if not self.generation_cancelled:
                os.startfile(of)
                messagebox.showinfo('DONE', f'Created {count} new images')

        except ValueError as e:
            if hasattr(self, 'progress_window'):
                self.progress_window.destroy()
            messagebox.showerror("Error", "Please enter valid numeric values.")

        except Exception as e:
            if hasattr(self, 'progress_window'):
                self.progress_window.destroy()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def on_cancel(self):
        self.root.destroy()

    def create_progress_bar(self):
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Generating Images")
        self.progress_window.geometry("300x150")

        # Center the progress window
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()

        # Progress label
        self.progress_label = ttk.Label(self.progress_window, text="Generating images...\n0 of 0")
        self.progress_label.pack(pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.progress_window, length=200, mode='determinate')
        self.progress_bar.pack(pady=10)

        # Cancel button
        self.cancel_button = ttk.Button(self.progress_window, text="Cancel", command=self.cancel_generation)
        self.cancel_button.pack(pady=10)

        self.generation_cancelled = False

    def cancel_generation(self):
        self.generation_cancelled = True

    def update_progress(self, current, total):
        if self.progress_window.winfo_exists():
            percentage = (current / total) * 100
            self.progress_bar['value'] = percentage
            self.progress_label.config(text=f"Generating images...\n{current} of {total}")
            self.progress_window.update()

    def close_progress(self):
        if self.progress_window.winfo_exists():
            self.progress_window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SplatImagesGUI(root)
    root.mainloop()
