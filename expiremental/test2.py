import os
import statistics

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for duplicate OpenMP library issue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
from queue import Queue
import easyocr
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw
import json


class ZoomableCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.scale = 1.0
        self.image_offset = (0, 0)
        self.bind('<MouseWheel>', self.zoom)
        self.bind('<ButtonPress-1>', self.start_pan)
        self.bind('<B1-Motion>', self.pan)
        self.bind('<Configure>', self.reset_view)

    def zoom(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.scale_image(scale_factor, event.x, event.y)

    def scale_image(self, scale_factor, x=0, y=0):
        new_scale = self.scale * scale_factor
        if 0.1 <= new_scale <= 5.0:
            # Calculate image position relative to canvas
            bbox = self.bbox("all")
            if not bbox: return

            # Adjust offset to zoom around mouse position
            self.image_offset = (
                x - (x - self.image_offset[0]) * scale_factor,
                y - (y - self.image_offset[1]) * scale_factor
            )

            self.scale = new_scale
            self.rescale()

    def start_pan(self, event):
        self.scan_mark(event.x, event.y)

    def pan(self, event):
        self.scan_dragto(event.x, event.y, gain=1)
        bbox = self.bbox("all")
        if bbox:
            self.image_offset = (bbox[0], bbox[1])

    def reset_view(self, event=None):
        if hasattr(self, 'original_image'):
            self.rescale()

    def rescale(self):
        if hasattr(self, 'original_image'):
            new_size = (int(self.original_size[0] * self.scale),
                        int(self.original_size[1] * self.scale))
            resized = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
            self.image = ImageTk.PhotoImage(resized)
            self.delete("all")
            self.create_image(self.image_offset, anchor="nw", image=self.image)
            self.configure(scrollregion=self.bbox("all"))


class SpreadsheetOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spreadsheet OCR to CSV")
        self.setup_ui()
        self.setup_ocr()
        self.setup_threading()
        self.load_config()

    def setup_ui(self):
        self.root.geometry("1200x800")
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.setup_image_pane()
        self.setup_control_pane()
        self.setup_status_bar()

    def setup_image_pane(self):
        self.left_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame, weight=2)

        self.canvas = ZoomableCanvas(self.left_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        scroll_x = ttk.Scrollbar(self.left_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scroll_y = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_control_pane(self):
        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=1)

        self.setup_parameters()
        self.setup_buttons()
        self.setup_preview()

    def setup_parameters(self):
        self.params_frame = ttk.LabelFrame(self.right_frame, text="OCR Parameters", padding="10")
        self.params_frame.pack(fill=tk.X, pady=5)

        self.params = {
            'text_threshold': (0.7, 0, 1),
            'low_text': (0.4, 0, 1),
            'link_threshold': (0.4, 0, 1),
            'contrast_boost': (1.0, 0.1, 3.0)
        }

        self.vars = {}
        for i, (name, (val, min_, max_)) in enumerate(self.params.items()):
            var = tk.DoubleVar(value=val)
            self.vars[name] = var
            self.create_slider(name.replace('_', ' ').title(), var, min_, max_, i)

    def create_slider(self, label, var, min_, max_, row):
        frame = ttk.Frame(self.params_frame)
        frame.grid(row=row, column=0, sticky='ew', pady=2)

        ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
        slider = ttk.Scale(frame, variable=var, from_=min_, to=max_,
                           orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        value = ttk.Label(frame, textvariable=var, width=5)
        value.pack(side=tk.LEFT)
        var.trace_add('write', lambda *_, v=var, l=value: l.config(text=f'{v.get():.2f}'))

    def setup_buttons(self):
        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        buttons = [
            ('Load Image', self.load_image),
            ('Process', self.process_image),
            ('Save CSV', self.save_csv),
            ('Config', self.config_menu)
        ]

        for i, (text, cmd) in enumerate(buttons):
            ttk.Button(btn_frame, text=text, command=cmd).grid(row=0, column=i, padx=2, sticky='ew')

        self.show_annotations = tk.BooleanVar()
        ttk.Checkbutton(btn_frame, text='Annotations', variable=self.show_annotations,
                        command=self.toggle_annotations).grid(row=0, column=4, padx=2)

    def config_menu(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Save Config", command=self.save_config)
        menu.add_command(label="Load Config", command=self.load_config_dialog)
        menu.tk_popup(*self.root.winfo_pointerxy())

    def setup_preview(self):
        self.preview_frame = ttk.LabelFrame(self.right_frame, text="Data Preview")
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.preview_frame, show='headings')
        vsb = ttk.Scrollbar(self.preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(self.preview_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

    def setup_status_bar(self):
        self.status_var = tk.StringVar()
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_ocr(self):
        self.reader = None
        self.current_image = None
        self.annotations = []
        self.processed_data = None

    def setup_threading(self):
        self.task_queue = Queue()
        self.root.after(100, self.process_queue)

    def process_queue(self):
        try:
            func, args, kwargs = self.task_queue.get_nowait()
            func(*args, **kwargs)
        except:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")
        ])
        if path:
            try:
                self.current_image = cv2.imread(path)
                img = Image.open(path)
                self.display_image = img.copy()
                self.canvas.original_image = img
                self.canvas.original_size = img.size
                self.canvas.scale = 1.0
                self.canvas.rescale()
                self.status("Image loaded")
                self.annotations = []
            except Exception as e:
                self.error(f"Error loading image: {str(e)}")

    def status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def error(self, message):
        messagebox.showerror("Error", message)

    def process_image(self):
        if self.current_image is None:
            self.error("Please load an image first")
            return

        self.status("Initializing OCR...")
        Thread(target=self._process_image, daemon=True).start()

    def _process_image(self):
        try:
            if self.reader is None:
                self.reader = easyocr.Reader(['en'])

            # Preprocess image
            img = self.preprocess_image(self.current_image)

            # Run OCR
            self.status("Processing...")
            result = self.reader.readtext(
                img,
                text_threshold=self.vars['text_threshold'].get(),
                low_text=self.vars['low_text'].get(),
                link_threshold=self.vars['link_threshold'].get()
            )

            # Process results
            self.annotations = result
            self.processed_data = self.structure_data(result)

            self.task_queue.put((self.update_preview, (), {}))
            self.task_queue.put((self.status, ("Processing complete",), {}))

            if self.show_annotations.get():
                self.task_queue.put((self.draw_annotations, (), {}))

        except Exception as e:
            self.task_queue.put((self.error, (f"Processing error: {str(e)}",), {}))

    def preprocess_image(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Contrast adjustment
        contrast = self.vars['contrast_boost'].get()
        if contrast != 1.0:
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)

        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        return denoised

    def structure_data(self, result):
        # First pass: Find table boundaries and cell positions
        x_positions = set()
        y_positions = set()

        for box, _, _ in result:
            # Get all unique x and y positions
            for point in box:
                x_positions.add(round(point[0], 1))
                y_positions.add(round(point[1], 1))

        # Create grid lines based on cell boundaries
        x_lines = sorted(x_positions)
        y_lines = sorted(y_positions)

        # Create a grid matrix to store cell contents
        grid = {}

        # Second pass: Place text in grid cells
        for box, text, conf in result:
            # Find cell boundaries
            x_min = min(p[0] for p in box)
            x_max = max(p[0] for p in box)
            y_min = min(p[1] for p in box)
            y_max = max(p[1] for p in box)

            # Find grid coordinates
            col_start = next(i for i, x in enumerate(x_lines) if abs(x - x_min) < 5)
            col_end = next(i for i, x in enumerate(x_lines) if abs(x - x_max) < 5)
            row_start = next(i for i, y in enumerate(y_lines) if abs(y - y_min) < 5)
            row_end = next(i for i, y in enumerate(y_lines) if abs(y - y_max) < 5)

            # Handle merged cells by filling all grid positions
            for row in range(row_start, row_end + 1):
                for col in range(col_start, col_end + 1):
                    grid[(row, col)] = {
                        'text': text,
                        'merged_rows': row_end - row_start + 1,
                        'merged_cols': col_end - col_start + 1,
                        'is_main_cell': (row == row_start and col == col_start)
                    }

        # Convert grid to DataFrame format
        max_row = max(row for row, _ in grid.keys()) if grid else 0
        max_col = max(col for _, col in grid.keys()) if grid else 0

        # Build rows with proper merged cell handling
        rows = []
        for i in range(max_row + 1):
            row_data = []
            j = 0
            while j <= max_col:
                if (i, j) in grid:
                    cell = grid[(i, j)]
                    if cell['is_main_cell']:
                        row_data.append(cell['text'])
                        # Skip positions covered by merged cells
                        if cell['merged_cols'] > 1:
                            j += cell['merged_cols'] - 1
                    elif not any(grid.get((i - k, j), {}).get('is_main_cell', False)
                                 for k in range(1, max_row + 1)):
                        # Add empty cell if not part of a vertical merge
                        row_data.append('')
                else:
                    row_data.append('')
                j += 1
            rows.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Clean up column names
        df.columns = [f"Col {i + 1}" for i in range(len(df.columns))]

        return df

    def update_preview(self):
        self.tree.delete(*self.tree.get_children())

        if self.processed_data is not None:
            self.tree["columns"] = list(self.processed_data.columns)
            for col in self.tree["columns"]:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, stretch=True)

            for _, row in self.processed_data.iterrows():
                self.tree.insert("", tk.END, values=row.tolist())

    def draw_annotations(self):
        if not self.annotations or not hasattr(self.canvas, 'original_image'):
            return

        draw_img = self.canvas.original_image.copy()
        draw = ImageDraw.Draw(draw_img)

        for box, text, _ in self.annotations:
            scaled_box = [(x * draw_img.width / self.current_image.shape[1],
                           y * draw_img.height / self.current_image.shape[0])
                          for x, y in box]
            draw.polygon(scaled_box, outline='red')

        self.canvas.original_image = draw_img
        self.canvas.rescale()

    def toggle_annotations(self):
        if self.show_annotations.get():
            self.draw_annotations()
        else:
            self.canvas.original_image = self.display_image
            self.canvas.rescale()

    def save_csv(self):
        if self.processed_data is None:
            self.error("No data to save")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if path:
            try:
                self.processed_data.to_csv(path, index=False)
                self.status("CSV saved successfully")
            except Exception as e:
                self.error(f"Save failed: {str(e)}")

    def save_config(self):
        config = {k: v.get() for k, v in self.vars.items()}
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f)
            self.status("Config saved")
        except Exception as e:
            self.error(f"Error saving config: {str(e)}")

    def load_config(self):
        try:
            with open('config.json') as f:
                config = json.load(f)
            for k, v in config.items():
                self.vars[k].set(v)
            self.status("Config loaded")
        except:
            pass

    def load_config_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                with open(path) as f:
                    config = json.load(f)
                for k, v in config.items():
                    self.vars[k].set(v)
                self.status("Config loaded")
            except Exception as e:
                self.error(f"Error loading config: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpreadsheetOCRApp(root)
    root.mainloop()