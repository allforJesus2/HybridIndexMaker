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

        self.reader_settings = reader_settings or {
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
            "decoder": 'beamsearch'
        }

        self.setup_ui()

    def setup_ui(self):
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.create_entry_fields(controls_frame)
        self.create_sliders(controls_frame)
        self.create_save_button(controls_frame)

        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

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
            ("scale_factor", "Scale Factor", 0.1, 5, 0.1)
        ]

        for setting, label, from_, to, resolution in slider_settings:
            slider = Scale(parent, from_=from_, to=to, orient=HORIZONTAL, resolution=resolution,
                           length=400, label=label)
            slider.set(self.reader_settings.get(setting, 1.0))  # Default to 1.0 for scale_factor
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
            self.reader_settings.update({
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
            })
            self.callback(self.reader_settings)

    def update_image(self):
        results = self.reader.readtext(
            self.imgpath,
            detail=1,
            **{k: v for k, v in self.reader_settings.items() if k != 'scale_factor'}
        )

        img = cv2.imread(self.imgpath)
        scale_factor = self.reader_settings.get('scale_factor', 1.0)
        scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

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
                    **{k: v for k, v in self.reader_settings.items() if k != 'scale_factor'}
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