import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button, Entry
import cv2
import easyocr
from PIL import Image, ImageTk


class SetReaderSettings:
    def __init__(self, root, imgpath, reader, reader_settings=None, callback=None):
        self.root = root
        self.reader = reader
        self.imgpath = imgpath
        self.callback = callback
        self.original_image = Image.open(self.imgpath)
        self.original_image_ratio = self.original_image.width / self.original_image.height
        self.resize_id = None

        # Define default settings
        default_settings = {
            "low_text": 0.3,
            "min_size": 10,
            "ycenter_ths": 0.5,
            "height_ths": 0.5,
            "width_ths": 6.0,
            "add_margin": -0.1,
            "link_threshold": 0.2,
            "text_threshold": 0.3,
            "mag_ratio": 3.0,
            "allowlist": '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()',
            "decoder": 'beamsearch',
            "batch_size": 1
        }

        # If reader_settings is provided, update default settings with provided values
        if reader_settings:
            default_settings.update(reader_settings)

        self.reader_settings = default_settings

        # Calculate minimum height needed for controls
        # Each entry field (approximately 50px)
        entry_fields_height = 50 * 2  # 2 entry fields

        # Each slider (approximately 60px)
        slider_count = 10  # Number of sliders
        sliders_height = 60 * slider_count

        # Save button (approximately 40px)
        save_button_height = 40

        # Padding and margins (approximately 20px top and bottom)
        padding_height = 40

        # Calculate total minimum height needed
        min_controls_height = entry_fields_height + sliders_height + save_button_height + padding_height

        # Set initial window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        initial_width = min(1200, screen_width - 100)  # Max width 1200 or screen width - 100
        initial_height = min(max(min_controls_height, 800),
                             screen_height - 100)  # Use the larger of min_controls_height or 800, but not larger than screen height - 100
        root.geometry(f"{initial_width}x{initial_height}")

        self.setup_ui()
        # Bind resize event after initial setup
        self.root.bind('<Configure>', self.on_window_resize)

    def setup_ui(self):
        # Main container with weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Controls frame (left side) - fixed width
        controls_frame = tk.Frame(self.root, width=400)
        controls_frame.grid(row=0, column=0, sticky='ns', padx=5, pady=5)

        # Create a canvas and scrollbar for the controls
        canvas = tk.Canvas(controls_frame, width=380)
        scrollbar = tk.Scrollbar(controls_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Add controls to the scrollable frame instead of controls_frame
        self.create_entry_fields(scrollable_frame)
        self.create_sliders(scrollable_frame)
        self.create_save_button(scrollable_frame)

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
    def create_entry_fields(self, parent):
        self.create_labeled_entry(parent, "Decoder", "decoder")
        self.create_labeled_entry(parent, "Allowlist", "allowlist")

    def create_labeled_entry(self, parent, label_text, setting_key):
        tk.Label(parent, text=label_text).pack(fill=tk.X)
        entry = Entry(parent)
        entry.insert(0, self.reader_settings[setting_key])
        entry.pack(fill=tk.X)
        setattr(self, f"{setting_key}_entry", entry)

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

    def save_settings(self):
        if self.callback:
            self.reader_settings = {
                "low_text": self.low_text.get(),
                "min_size": int(self.min_size.get()),
                "ycenter_ths": self.ycenter_ths.get(),
                "height_ths": self.height_ths.get(),
                "width_ths": self.width_ths.get(),
                "add_margin": self.add_margin.get(),
                "link_threshold": self.link_threshold.get(),
                "text_threshold": self.text_threshold.get(),
                "mag_ratio": self.mag_ratio.get(),
                "allowlist": self.allowlist_entry.get(),
                "decoder": self.decoder_entry.get(),
                "batch_size": int(self.batch_size.get())
            }
            self.callback(self.reader_settings)

    def on_window_resize(self, event):
        # Only process resize events from the root window
        if event.widget == self.root and self.resize_id is None:
            # Schedule new resize event
            self.resize_id = self.root.after(50, self.delayed_resize)

    def delayed_resize(self):
        self.resize_id = None
        self.update_image()

    def update_image(self):
        results = self.reader.readtext(
            self.imgpath,
            detail=1,
            **self.reader_settings
        )

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

        final_results = []

        for item in results:
            box, text, score = item
            if score < 0.3:
                x1, y1, x2, y2 = [int(coord) for coord in (box[0][0], box[0][1], box[2][0], box[2][1])]
                cropped_img = img[y1:y2, x1:x2]
                rotated_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)

                rotated_results = self.reader.readtext(
                    rotated_img,
                    detail=1,
                    **self.reader_settings
                )

                for rotated_item in rotated_results:
                    rotated_box, rotated_text, rotated_score = rotated_item
                    rotated_x1, rotated_y1, rotated_x2, rotated_y2 = [int(coord) for coord in (
                        rotated_box[0][0], rotated_box[0][1], rotated_box[2][0], rotated_box[2][1])]

                    # Adjust the coordinates back to the original image
                    new_x1 = x1 + rotated_y1
                    new_y1 = y1 + (x2 - x1) - rotated_x2
                    new_x2 = x1 + rotated_y2
                    new_y2 = y1 + (x2 - x1) - rotated_x1

                    final_results.append(
                        [[[new_x1, new_y1], [new_x2, new_y1], [new_x2, new_y2], [new_x1, new_y2]], rotated_text,
                         rotated_score])
            else:
                final_results.append(item)

        for (box, text, score) in final_results:
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in (box[0][0], box[0][1], box[2][0], box[2][1])]
            cv2.rectangle(scaled_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(scaled_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(scaled_img, f"{score:.2f}", (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = ImageTk.PhotoImage(scaled_img)
        self.image_label.config(image=scaled_img)
        self.image_label.image = scaled_img


import tkinter as tk
from tkinter import filedialog
import easyocr
from PIL import Image


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