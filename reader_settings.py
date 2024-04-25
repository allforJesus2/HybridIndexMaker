import tkinter as tk
from tkinter import Scale, HORIZONTAL, Button, Entry
import cv2
import easyocr
from PIL import Image, ImageTk

class SetReaderSettings:
    def __init__(self, root, imgpath, reader, reader_settings=None, callback=None):
        self.root = root
        self.reader = reader#easyocr.Reader(['en'])
        self.imgpath = imgpath
        self.callback = callback
        self.original_image = Image.open(self.imgpath)  # Open the image using PIL
        self.original_image_ratio = self.original_image.width / self.original_image.height  # Calculate the aspect ratio

        if reader_settings:
            self.reader_settings = reader_settings
        else:
            self.reader_settings = {
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



        # Create a frame for the controls
        controls_frame = tk.Frame(root)
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        decoder_label = tk.Label(controls_frame, text="Decoder")
        decoder_label.pack(fill=tk.X)

        self.decoder_entry = Entry(controls_frame)
        self.decoder_entry.insert(0, self.reader_settings["decoder"])
        self.decoder_entry.pack(fill=tk.X)

        allowlist_label = tk.Label(controls_frame, text="Allowlist")
        allowlist_label.pack(fill=tk.X)

        self.allowlist_entry = Entry(controls_frame)
        self.allowlist_entry.insert(0, self.reader_settings["allowlist"])
        self.allowlist_entry.pack(fill=tk.X)

        self.mag_ratio = Scale(controls_frame, from_=0, to=5, orient=HORIZONTAL, resolution=0.1, length=400,
                               label='Magnification', command=self.update_image)
        self.mag_ratio.set(self.reader_settings["mag_ratio"])
        self.mag_ratio.pack(fill=tk.X)

        self.text_threshold = Scale(controls_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.01, length=400,
                                    label='Text Threshold', command=self.update_image)
        self.text_threshold.set(self.reader_settings["text_threshold"])
        self.text_threshold.pack(fill=tk.X)

        self.low_text = Scale(controls_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.01, length=400,
                              label='Low Text', command=self.update_image)
        self.low_text.set(self.reader_settings["low_text"])
        self.low_text.pack(fill=tk.X)

        self.link_threshold = Scale(controls_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.01, length=400,
                                    label='Link Threshold', command=self.update_image)

        self.link_threshold.set(self.reader_settings["link_threshold"])
        self.link_threshold.pack(fill=tk.X)

        self.min_size = Scale(controls_frame, from_=0, to=100, orient=HORIZONTAL, resolution=1, length=400,
                              label='Min Size', command=self.update_image)
        self.min_size.set(self.reader_settings["min_size"])
        self.min_size.pack(fill=tk.X)

        self.ycenter_ths = Scale(controls_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.01, length=400,
                                 label='YCenter THS', command=self.update_image)
        self.ycenter_ths.set(self.reader_settings["ycenter_ths"])
        self.ycenter_ths.pack(fill=tk.X)

        self.height_ths = Scale(controls_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.01, length=400,
                                label='Height THS', command=self.update_image)
        self.height_ths.set(self.reader_settings["height_ths"])
        self.height_ths.pack(fill=tk.X)

        self.width_ths = Scale(controls_frame, from_=0, to=10, orient=HORIZONTAL, resolution=0.1, length=400,
                               label='Width THS', command=self.update_image)
        self.width_ths.set(self.reader_settings["width_ths"])
        self.width_ths.pack(fill=tk.X)

        self.add_margin = Scale(controls_frame, from_=-1, to=1, orient=HORIZONTAL, resolution=0.01, length=400,
                                label='Add Margin', command=self.update_image)
        self.add_margin.set(self.reader_settings["add_margin"])
        self.add_margin.pack(fill=tk.X)

        # Create a frame for the image
        image_frame = tk.Frame(root)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        #image_frame.pack_propagate(0)  # Prevent the frame from resizing based on contents

        self.image_label = tk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.save_button = Button(controls_frame, text="Save", command=self.save_settings)
        self.save_button.pack(fill=tk.X)

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
            }
            self.callback(self.reader_settings)

    def update_image(self, event=None):
        results = self.reader.readtext(self.imgpath, detail=1,
                                        text_threshold=self.text_threshold.get(),
                                        low_text=self.low_text.get(),
                                        link_threshold=self.link_threshold.get(),
                                        min_size=int(self.min_size.get()),
                                        ycenter_ths=self.ycenter_ths.get(),
                                        height_ths=self.height_ths.get(),
                                        width_ths=self.width_ths.get(),
                                        add_margin=self.add_margin.get(),
                                       mag_ratio=self.mag_ratio.get(),
                                       allowlist=self.allowlist_entry.get(),
                                        decoder=self.decoder_entry.get())

        img = cv2.imread(self.imgpath)
        scale_factor = 3  # Set the desired scale factor here
        scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        for result in results:
            box = result[0]
            text = result[1]
            x1, y1, x2, y2 = [int(coord) * scale_factor for coord in (box[0][0], box[0][1], box[2][0], box[2][1])]
            cv2.rectangle(scaled_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(scaled_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
        scaled_img = Image.fromarray(scaled_img)

        scaled_img = ImageTk.PhotoImage(scaled_img)
        self.image_label.config(image=scaled_img)
        self.image_label.image = scaled_img