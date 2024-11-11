import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button, Entry, ttk, Listbox, MULTIPLE
import cv2
import easyocr
from PIL import Image, ImageTk
from tkinter import filedialog
import platform
class SetReaderSettings:
    def __init__(self, root, imgpath, reader, reader_settings=None, callback=None):
        self.root = root
        self.root.state('zoomed')  # For Windows

        self.reader = reader
        self.imgpath = imgpath
        self.callback = callback
        self.original_image = Image.open(self.imgpath)
        self.original_image_ratio = self.original_image.width / self.original_image.height
        self.resize_id = None

        # Define default settings with all parameters
        default_settings = {
            # General Parameters
            "decoder": "beamsearch",  # greedy, wordbeamsearch
            "beamWidth": 5,
            "batch_size": 1,
            "workers": 0,
            "allowlist": "",
            "blocklist": "",
            "detail": 1,
            "paragraph": False,
            "min_size": 10,
            "rotation_info": "",  # Will be converted to list when used

            # Contrast Parameters
            "contrast_ths": 0.1,
            "adjust_contrast": 0.5,

            # Text Detection Parameters
            "text_threshold": 0.7,
            "low_text": 0.4,
            "link_threshold": 0.4,
            "canvas_size": 2560,
            "mag_ratio": 1.0,

            # Bounding Box Parameters
            "slope_ths": 0.1,
            "ycenter_ths": 0.5,
            "height_ths": 0.5,
            "width_ths": 0.5,
            "add_margin": 0.1,
            "x_ths": 1.0,
            "y_ths": 0.5
        }

        # If reader_settings is provided, update default settings with provided values
        if reader_settings:
            default_settings.update(reader_settings)

        self.reader_settings = default_settings

        # Calculate minimum height needed for controls
        # Entry fields (approximately 50px each)
        entry_fields = 8  # decoder, beamWidth, allowlist, blocklist, rotation_info
        entry_fields_height = 50 * entry_fields

        # Sliders (approximately 60px each)
        slider_count = 17  # Count of all slider parameters
        sliders_height = 60 * slider_count

        # Label frames padding (approximately 40px each)
        frame_count = 4  # General, Contrast, Text Detection, Bounding Box
        frames_padding = 40 * frame_count

        # Save button (approximately 40px)
        save_button_height = 40

        # Padding and margins (approximately 20px top and bottom)
        padding_height = 40

        # Calculate total minimum height needed
        min_controls_height = (
                entry_fields_height +
                sliders_height +
                frames_padding +
                save_button_height +
                padding_height
        )

        # Set initial window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        initial_width = min(1200, screen_width - 100)
        initial_height = min(max(min_controls_height, 800), screen_height - 100)
        root.geometry(f"{initial_width}x{initial_height}")

        self.setup_ui()
        # Bind resize event after initial setup
        self.root.bind('<Configure>', self.on_window_resize)

    def create_labeled_entry(self, parent, label_text, setting_key, default_value=""):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(frame, text=label_text, anchor='w').pack(side=tk.LEFT)

        if setting_key == "decoder":
            # Decoder dropdown
            decoder_options = ["beamsearch", "greedy", "wordbeamsearch"]
            entry = ttk.Combobox(frame, values=decoder_options, state="readonly")
            entry.set(self.reader_settings.get(setting_key, default_value))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entry.bind('<<ComboboxSelected>>', lambda e: self.update_image())
        elif setting_key == "rotation_info":
            # Rotation info multi-select listbox
            frame_right = tk.Frame(frame)
            frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.X)

            # Create listbox with multiple selection enabled
            entry = Listbox(frame_right, selectmode=MULTIPLE, height=3)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

            # Add rotation options
            rotation_options = ["90", "180", "270"]
            for option in rotation_options:
                entry.insert(tk.END, option)

            # Select default values if any
            if self.reader_settings.get(setting_key):
                current_values = [str(x.strip()) for x in self.reader_settings.get(setting_key).split(',') if x.strip()]
                for i, option in enumerate(rotation_options):
                    if option in current_values:
                        entry.selection_set(i)

            # Add selection change handler
            entry.bind('<<ListboxSelect>>', lambda e: self.update_image())

            # Add helper buttons
            button_frame = tk.Frame(frame_right)
            button_frame.pack(side=tk.RIGHT, fill=tk.Y)

            tk.Button(button_frame, text="All",
                      command=lambda: (entry.selection_set(0, tk.END), self.update_image())
                      ).pack(fill=tk.X)
            tk.Button(button_frame, text="None",
                      command=lambda: (entry.selection_clear(0, tk.END), self.update_image())
                      ).pack(fill=tk.X)
        else:
            # Regular entry with optional Apply button
            entry_frame = tk.Frame(frame)
            entry_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X)

            entry = Entry(entry_frame)
            entry.insert(0, self.reader_settings.get(setting_key, default_value))

            # Add Apply button for specific fields
            if setting_key in ["beamWidth", "allowlist", "blocklist"]:
                entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
                apply_button = tk.Button(
                    entry_frame,
                    text="Apply",
                    command=lambda k=setting_key, e=entry: self.apply_entry_value(k, e)
                )
                apply_button.pack(side=tk.RIGHT, padx=(5, 0))
            else:
                entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)

        setattr(self, f"{setting_key}_entry", entry)
        return entry

    def apply_entry_value(self, setting_key, entry):
        # Update the setting value and refresh the image
        value = entry.get()
        if setting_key == "beamWidth":
            try:
                value = int(value)
            except ValueError:
                return
        self.reader_settings[setting_key] = value
        self.update_image()
    def update_image(self):
        # Get selected rotation values from listbox
        rotation_info = None
        if hasattr(self, 'rotation_info_entry'):
            selected_indices = self.rotation_info_entry.curselection()
            if selected_indices:
                rotation_values = [self.rotation_info_entry.get(i) for i in selected_indices]
                rotation_info = [int(x) for x in rotation_values]

        # Prepare parameters for readtext
        read_params = {
            # Only include parameters that are valid for readtext()
            "decoder": self.reader_settings['decoder'],
            "batch_size": int(self.reader_settings['batch_size']),
            "workers": int(self.reader_settings['workers']),
            "detail": int(self.reader_settings['detail']),
            "paragraph": bool(self.reader_settings['paragraph']),
            "contrast_ths": float(self.reader_settings['contrast_ths']),
            "adjust_contrast": float(self.reader_settings['adjust_contrast']),
            "text_threshold": float(self.reader_settings['text_threshold']),
            "low_text": float(self.reader_settings['low_text']),
            "link_threshold": float(self.reader_settings['link_threshold']),
            "canvas_size": int(self.reader_settings['canvas_size']),
            "mag_ratio": float(self.reader_settings['mag_ratio']),
            "slope_ths": float(self.reader_settings['slope_ths']),
            "ycenter_ths": float(self.reader_settings['ycenter_ths']),
            "height_ths": float(self.reader_settings['height_ths']),
            "width_ths": float(self.reader_settings['width_ths']),
            "add_margin": float(self.reader_settings['add_margin']),
            "x_ths": float(self.reader_settings['x_ths']),
            "y_ths": float(self.reader_settings['y_ths'])
        }

        # Add rotation_info if present
        if rotation_info is not None:
            read_params['rotation_info'] = rotation_info

        # Add allowlist if not empty
        if self.reader_settings['allowlist']:
            read_params['allowlist'] = self.reader_settings['allowlist']

        # Add blocklist if not empty
        if self.reader_settings['blocklist']:
            read_params['blocklist'] = self.reader_settings['blocklist']

        # Remove parameters that aren't supported by the current decoder
        if read_params['decoder'] == 'greedy':
            # Greedy decoder doesn't support beamWidth
            read_params.pop('beamWidth', None)

        results = self.reader.readtext(
            self.imgpath,
            **read_params
        )

        # Rest of the image display code remains the same
        img = cv2.imread(self.imgpath)

        # Get the available space in the window
        window_width = self.image_label.winfo_width()
        window_height = self.image_label.winfo_height()

        # Ensure minimum dimensions
        window_width = max(window_width, 100)
        window_height = max(window_height, 100)

        # Calculate scaling factors
        width_ratio = window_width / img.shape[1]
        height_ratio = window_height / img.shape[0]

        # Use the smaller ratio to maintain aspect ratio
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)

        # Resize the image
        scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        for (box, text, score) in results:
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in (box[0][0], box[0][1], box[2][0], box[2][1])]
            cv2.rectangle(scaled_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(scaled_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(scaled_img, f"{score:.2f}", (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = ImageTk.PhotoImage(scaled_img)
        self.image_label.config(image=scaled_img)
        self.image_label.image = scaled_img
    def save_settings(self):
        if self.callback:
            # Get selected rotation values
            rotation_info = ""
            if hasattr(self, 'rotation_info_entry'):
                selected_indices = self.rotation_info_entry.curselection()
                if selected_indices:
                    rotation_values = [self.rotation_info_entry.get(i) for i in selected_indices]
                    rotation_info = ','.join(rotation_values)

            self.reader_settings = {
                # General Parameters
                "decoder": self.decoder_entry.get(),
                "beamWidth": int(self.beamWidth_entry.get()),
                "batch_size": int(self.batch_size.get()),
                "workers": int(self.workers.get()),
                "allowlist": self.allowlist_entry.get(),
                "blocklist": self.blocklist_entry.get(),
                "min_size": int(self.min_size.get()),
                "rotation_info": rotation_info,

                # Contrast Parameters
                "contrast_ths": self.contrast_ths.get(),
                "adjust_contrast": self.adjust_contrast.get(),

                # Text Detection Parameters
                "text_threshold": self.text_threshold.get(),
                "low_text": self.low_text.get(),
                "link_threshold": self.link_threshold.get(),
                "canvas_size": int(self.canvas_size.get()),
                "mag_ratio": self.mag_ratio.get(),

                # Bounding Box Parameters
                "slope_ths": self.slope_ths.get(),
                "ycenter_ths": self.ycenter_ths.get(),
                "height_ths": self.height_ths.get(),
                "width_ths": self.width_ths.get(),
                "add_margin": self.add_margin.get(),
                "x_ths": self.x_ths.get(),
                "y_ths": self.y_ths.get()
            }
            self.callback(self.reader_settings)

    def create_labeled_slider(self, parent, setting_key, label, from_, to, resolution, default_value):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)

        slider = Scale(frame, from_=from_, to=to, orient=HORIZONTAL, resolution=resolution,
                       length=300, label=label)
        slider.set(self.reader_settings.get(setting_key, default_value))
        slider.pack(fill=tk.X)
        slider.bind("<ButtonRelease-1>", lambda event, s=setting_key: self.on_slider_release(event, s))
        setattr(self, setting_key, slider)
        return slider

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Controls frame (left side) - fixed width
        controls_frame = tk.Frame(self.root, width=400)
        controls_frame.grid(row=0, column=0, sticky='ns', padx=5, pady=5)
        controls_frame.grid_propagate(False)  # Prevent frame from shrinking

        # Create a container frame for both canvas and save button
        container_frame = tk.Frame(controls_frame)
        container_frame.pack(fill="both", expand=True)

        # Create a canvas and scrollbar for the controls
        canvas = tk.Canvas(container_frame, width=380)
        scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=canvas.yview)

        # Configure the canvas to work with the scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create the scrollable frame inside the canvas
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Create window with frame inside canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=365)

        def _on_mousewheel(event):
            # Get current scroll position
            current_pos = canvas.yview()

            # Calculate scroll amount based on event delta
            if platform.system() == "Windows":
                delta = -int(event.delta / 120)  # Windows
            elif platform.system() == "Darwin":
                delta = -event.delta  # macOS
            else:
                if event.num == 4:  # Linux scroll up
                    delta = -1
                elif event.num == 5:  # Linux scroll down
                    delta = 1
                else:
                    return

            # Scroll the canvas
            canvas.yview_scroll(delta, "units")

            # If we've moved, prevent the event from propagating
            if current_pos != canvas.yview():
                return "break"

        def _bind_mousewheel(event=None):
            # Bind mouse wheel to canvas
            if platform.system() == "Linux":
                canvas.bind_all("<Button-4>", _on_mousewheel)
                canvas.bind_all("<Button-5>", _on_mousewheel)
            else:
                canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event=None):
            # Unbind mouse wheel from canvas
            if platform.system() == "Linux":
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            else:
                canvas.unbind_all("<MouseWheel>")

        # Bind mouse enter/leave events for the scrollable area
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # Pack the canvas and scrollbar properly
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)


        button_frame = tk.Frame(controls_frame)
        button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        # Add Save button
        save_button = Button(button_frame, text="Save", command=self.save_settings)
        save_button.pack(side="left", expand=True, fill="x", padx=(0, 2))

        # Add Save & Close button
        save_close_button = Button(button_frame, text="Save & Close", command=self.save_and_close)
        save_close_button.pack(side="left", expand=True, fill="x", padx=(2, 0))

        # Create labeled frames for parameter groups
        general_frame = tk.LabelFrame(scrollable_frame, text="General Parameters")
        general_frame.pack(fill=tk.X, padx=5, pady=5)

        contrast_frame = tk.LabelFrame(scrollable_frame, text="Contrast Parameters")
        contrast_frame.pack(fill=tk.X, padx=5, pady=5)

        detection_frame = tk.LabelFrame(scrollable_frame, text="Text Detection Parameters")
        detection_frame.pack(fill=tk.X, padx=5, pady=5)

        bbox_frame = tk.LabelFrame(scrollable_frame, text="Bounding Box Parameters")
        bbox_frame.pack(fill=tk.X, padx=5, pady=5)

        # General Parameters
        self.create_labeled_entry(general_frame, "Decoder", "decoder", "greedy")
        self.create_labeled_entry(general_frame, "Beam Width", "beamWidth", "5")
        self.create_labeled_slider(general_frame, "batch_size", "Batch Size", 1, 100, 1, 1)
        self.create_labeled_slider(general_frame, "workers", "Workers", 0, 16, 1, 0)
        self.create_labeled_entry(general_frame, "Allowlist", "allowlist")
        self.create_labeled_entry(general_frame, "Blocklist", "blocklist")
        self.create_labeled_slider(general_frame, "min_size", "Min Size", 0, 100, 1, 10)
        self.create_labeled_entry(general_frame, "Rotation Info", "rotation_info")

        # Contrast Parameters
        self.create_labeled_slider(contrast_frame, "contrast_ths", "Contrast Threshold", 0, 1, 0.01, 0.1)
        self.create_labeled_slider(contrast_frame, "adjust_contrast", "Adjust Contrast", 0, 1, 0.01, 0.5)

        # Text Detection Parameters
        self.create_labeled_slider(detection_frame, "text_threshold", "Text Threshold", 0, 1, 0.01, 0.7)
        self.create_labeled_slider(detection_frame, "low_text", "Low Text", 0, 1, 0.01, 0.4)
        self.create_labeled_slider(detection_frame, "link_threshold", "Link Threshold", 0, 1, 0.01, 0.4)
        self.create_labeled_slider(detection_frame, "canvas_size", "Canvas Size", 1000, 5000, 10, 2560)
        self.create_labeled_slider(detection_frame, "mag_ratio", "Magnification Ratio", 0, 5, 0.1, 1)

        # Bounding Box Parameters
        self.create_labeled_slider(bbox_frame, "slope_ths", "Slope Threshold", 0, 1, 0.01, 0.1)
        self.create_labeled_slider(bbox_frame, "ycenter_ths", "YCenter Threshold", 0, 1, 0.01, 0.5)
        self.create_labeled_slider(bbox_frame, "height_ths", "Height Threshold", 0, 1, 0.01, 0.5)
        self.create_labeled_slider(bbox_frame, "width_ths", "Width Threshold", 0, 10, 0.1, 0.5)
        self.create_labeled_slider(bbox_frame, "add_margin", "Add Margin", -1, 1, 0.01, 0.1)
        self.create_labeled_slider(bbox_frame, "x_ths", "X Threshold", 0, 2, 0.1, 1.0)
        self.create_labeled_slider(bbox_frame, "y_ths", "Y Threshold", 0, 1, 0.01, 0.5)

        # Image frame (right side)
        image_frame = tk.Frame(self.root)
        image_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(image_frame)
        self.image_label.grid(row=0, column=0, sticky='nsew')

        # Initial update
        self.root.update_idletasks()
        self.update_image()

    def save_and_close(self):
        """Save settings and close the window"""
        self.save_settings()
        self.root.destroy()
    def create_entry_fields(self, parent):
        self.create_labeled_entry(parent, "Decoder", "decoder")
        self.create_labeled_entry(parent, "Allowlist", "allowlist")

    def create_sliders(self, parent):
        slider_settings = [
            ("mag_ratio", "Magnification", 0, 5, 0.1),
            ("text_threshold", "Text Threshold", 0, 1, 0.01),
            ("low_text", "Low Text", 0, 1, 0.01),
            ("link_threshold", "Link Threshold", 0, 1, 0.01),
            ("min_size", "Min Size", 0, 100, 1),
            ("ycenter_ths", "YCenter THS", 0, 1, 0.01),
            ("height_ths", "Height THS", 0, 1, 0.01),
            ("width_ths", "Width THS", 0, 10, 0.1),
            ("add_margin", "Add Margin", -1, 1, 0.01),
            ("batch_size", "Batch Size", 1, 100, 1)
        ]

        for setting, label, from_, to, resolution in slider_settings:
            slider = Scale(parent, from_=from_, to=to, orient=HORIZONTAL, resolution=resolution,
                           length=300, label=label)  # Reduced length to fit fixed width
            slider.set(self.reader_settings[setting])
            slider.pack(fill=tk.X)
            slider.bind("<ButtonRelease-1>", lambda event, s=setting: self.on_slider_release(event, s))
            setattr(self, setting, slider)

    def create_save_button(self, parent):
        Button(parent, text="Save", command=self.save_settings).pack(fill=tk.X)

    def on_slider_release(self, event, setting):
        self.reader_settings[setting] = getattr(self, setting).get()
        self.update_image()

    def on_window_resize(self, event):
        # Only process resize events from the root window
        if event.widget == self.root and self.resize_id is None:
            # Schedule new resize event
            self.resize_id = self.root.after(50, self.delayed_resize)

    def delayed_resize(self):
        self.resize_id = None
        self.update_image()




class OCRSettingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Settings")

        # Initialize EasyOCR reader
        print("Initializing EasyOCR... (this may take a moment)")
        self.reader = easyocr.Reader(['en'])

        # Create main button frame
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=20)

        # Create open file button
        self.open_button = tk.Button(
            self.button_frame,
            text="Open Image",
            command=self.open_file
        )
        self.open_button.pack()

        self.settings_window = None

    def open_file(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            # Close previous settings window if it exists
            if self.settings_window:
                self.settings_window.destroy()

            # Create new window for settings
            self.settings_window = tk.Toplevel(self.root)
            self.settings_window.title("OCR Settings - " + file_path)

            # Initialize settings window with the selected image
            SetReaderSettings(
                self.settings_window,
                file_path,
                self.reader,
                callback=self.on_settings_save
            )

    def on_settings_save(self, settings):
        print("Settings saved:", settings)


def main():
    root = tk.Tk()
    root.geometry("300x100")  # Small window for the initial file selection
    app = OCRSettingsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()