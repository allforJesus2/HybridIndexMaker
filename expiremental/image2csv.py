import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tkinter as tk
from tkinter import ttk, filedialog
import easyocr
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw


class ZoomableCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.bind('<MouseWheel>', self._on_mousewheel)
        self.scale = 1.0

    def _on_mousewheel(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= scale_factor
        self.scale = max(0.1, min(5.0, self.scale))
        self.rescale()

    def rescale(self):
        if hasattr(self, 'original_image'):
            new_size = (int(self.original_size[0] * self.scale),
                        int(self.original_size[1] * self.scale))
            resized = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(resized)
            self.delete("all")
            self.create_image(0, 0, anchor="nw", image=self.image)
            self.configure(scrollregion=self.bbox("all"))


class SpreadsheetOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spreadsheet OCR to CSV")
        self.reader = None
        self.current_image = None
        self.annotations = []

        # Main frame with weights
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Create left and right panes
        self.paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Left pane for image
        self.left_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame, weight=1)

        # Right pane for controls and preview
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=1)

        # Zoomable canvas with scrollbars
        self.canvas_frame = ttk.Frame(self.left_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.canvas = ZoomableCanvas(self.canvas_frame, bg='white')
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        # Parameters frame
        self.params_frame = ttk.LabelFrame(self.right_frame, text="OCR Parameters", padding="5")
        self.params_frame.grid(row=0, column=0, sticky="ew", pady=5)

        # OCR Parameters with value labels
        self.text_threshold = tk.DoubleVar(value=0.7)
        self.low_text = tk.DoubleVar(value=0.4)
        self.link_threshold = tk.DoubleVar(value=0.4)

        self.create_slider("Text Threshold", self.text_threshold, 0, 1, 0)
        self.create_slider("Low Text", self.low_text, 0, 1, 1)
        self.create_slider("Link Threshold", self.link_threshold, 0, 1, 2)

        # Control buttons
        self.button_frame = ttk.Frame(self.right_frame)
        self.button_frame.grid(row=1, column=0, pady=10, sticky="ew")

        ttk.Button(self.button_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(self.button_frame, text="Process", command=self.process_image).grid(row=0, column=1, padx=5)
        ttk.Button(self.button_frame, text="Save CSV", command=self.save_csv).grid(row=0, column=2, padx=5)

        self.show_annotations = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.button_frame, text="Show Annotations",
                        variable=self.show_annotations,
                        command=self.toggle_annotations).grid(row=0, column=3, padx=5)

        # Grid preview
        self.preview_frame = ttk.LabelFrame(self.right_frame, text="Data Preview")
        self.preview_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        self.right_frame.grid_rowconfigure(2, weight=1)

        self.tree = ttk.Treeview(self.preview_frame, show="headings")
        self.tree_scroll_y = ttk.Scrollbar(self.preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree_scroll_x = ttk.Scrollbar(self.preview_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree_scroll_y.grid(row=0, column=1, sticky="ns")
        self.tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.right_frame, textvariable=self.status_var).grid(row=3, column=0, pady=5)

        self.processed_data = None

    def create_slider(self, label, variable, min_val, max_val, row):
        frame = ttk.Frame(self.params_frame)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.grid_columnconfigure(1, weight=1)

        ttk.Label(frame, text=label).grid(row=0, column=0, padx=5)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable,
                           orient=tk.HORIZONTAL)
        slider.grid(row=0, column=1, padx=5)

        value_label = ttk.Label(frame, text=f"{variable.get():.2f}")
        value_label.grid(row=0, column=2, padx=5)

        def update_label(*args):
            value_label.config(text=f"{variable.get():.2f}")

        variable.trace_add("write", update_label)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image = Image.open(file_path)
            self.canvas.original_image = self.display_image
            self.canvas.original_size = self.display_image.size
            self.canvas.scale = 1.0
            self.canvas.rescale()
            self.status_var.set("Image loaded successfully")
            self.annotations = []

    def process_image(self):
        if self.current_image is None:
            self.status_var.set("Please load an image first")
            return

        try:
            if self.reader is None:
                self.status_var.set("Initializing EasyOCR...")
                self.root.update()
                self.reader = easyocr.Reader(['en'])

            self.status_var.set("Processing image...")
            self.root.update()

            result = self.reader.readtext(
                self.current_image,
                text_threshold=self.text_threshold.get(),
                low_text=self.low_text.get(),
                link_threshold=self.link_threshold.get()
            )

            # Store annotations
            self.annotations = result

            # Convert OCR results to structured data
            data = []
            current_row = []
            current_y = result[0][0][0][1] if result else 0
            y_threshold = 10

            for detection in result:
                box, text, conf = detection
                y_coord = box[0][1]

                if abs(y_coord - current_y) > y_threshold and current_row:
                    data.append(current_row)
                    current_row = []
                    current_y = y_coord

                current_row.append(text)

            if current_row:
                data.append(current_row)

            self.processed_data = pd.DataFrame(data)
            self.update_preview()
            self.status_var.set("Processing complete!")

            if self.show_annotations.get():
                self.draw_annotations()

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def update_preview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.processed_data is not None:
            # Configure columns with fixed width and minimum size
            self.tree["columns"] = [str(i) for i in range(len(self.processed_data.columns))]
            for col in self.tree["columns"]:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=40, minwidth=30, stretch=True)

            # Add data
            for idx, row in self.processed_data.iterrows():
                self.tree.insert("", "end", values=row.tolist())

    def draw_annotations(self):
        if not hasattr(self.canvas, 'original_image'):
            return

        img_draw = self.canvas.original_image.copy()
        draw = ImageDraw.Draw(img_draw)

        scale = img_draw.size[0] / self.current_image.shape[1]

        for box, text, conf in self.annotations:
            points = [(int(x * scale), int(y * scale)) for x, y in box]
            draw.polygon(points, outline='red')

        self.canvas.original_image = img_draw
        self.canvas.rescale()

    def toggle_annotations(self):
        if self.annotations:
            if self.show_annotations.get():
                self.draw_annotations()
            else:
                self.canvas.original_image = self.display_image
                self.canvas.rescale()

    def save_csv(self):
        if self.processed_data is None:
            self.status_var.set("Please process the image first")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if save_path:
            self.processed_data.to_csv(save_path, index=False)
            self.status_var.set("CSV saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpreadsheetOCRApp(root)
    root.mainloop()