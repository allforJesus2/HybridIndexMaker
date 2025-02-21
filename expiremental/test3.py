import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for duplicate OpenMP library issue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import easyocr
import cv2
import numpy as np
from typing import List, Tuple, Dict
import csv
from dataclasses import dataclass


@dataclass
class OCRResult:
    box: List[List[int]]
    text: str
    score: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return min(xs), min(ys), max(xs), max(ys)


class GridDetector:
    def __init__(self, image_width: int, image_height: int, min_gap: int = 10):
        self.width = image_width
        self.height = image_height
        self.min_gap = min_gap

    def find_gaps(self, ocr_results: List[OCRResult], axis: str = 'vertical') -> List[int]:
        is_vertical = axis == 'vertical'
        dim = self.width if is_vertical else self.height
        occupied = np.zeros(dim, dtype=bool)

        for result in ocr_results:
            min_x, min_y, max_x, max_y = result.bounds
            if is_vertical:
                occupied[int(min_x):int(max_x)] = True
            else:
                occupied[int(min_y):int(max_y)] = True

        gaps = []
        start = None

        for i in range(dim):
            if not occupied[i]:
                if start is None:
                    start = i
            elif start is not None:
                if i - start >= self.min_gap:
                    gaps.append((start + i) // 2)
                start = None

        return gaps

    def assign_to_cells(self, ocr_results: List[OCRResult], v_lines: List[int], h_lines: List[int]) -> Dict[
        Tuple[int, int], str]:
        cells = {}
        v_lines = [0] + v_lines + [self.width]
        h_lines = [0] + h_lines + [self.height]

        for result in ocr_results:
            min_x, min_y, max_x, max_y = result.bounds
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            col = next(i for i, x in enumerate(v_lines[1:]) if center_x < x)
            row = next(i for i, y in enumerate(h_lines[1:]) if center_y < y)

            cells[(row, col)] = result.text

        return cells


class OCRSpreadsheetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Spreadsheet to CSV Converter")

        # Variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.ocr_results = None
        self.show_annotations = tk.BooleanVar(value=True)

        # OCR Parameters
        self.text_threshold = tk.DoubleVar(value=0.7)
        self.low_text = tk.DoubleVar(value=0.4)
        self.link_threshold = tk.DoubleVar(value=0.4)

        self._create_widgets()

    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Process", command=self.process_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(button_frame, text="Show Annotations", variable=self.show_annotations,
                        command=self.toggle_annotations).pack(side=tk.LEFT, padx=5)

        # Parameters frame
        param_frame = ttk.LabelFrame(main_frame, text="OCR Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(param_frame, text="Text Threshold:").grid(row=0, column=0, padx=5)
        ttk.Scale(param_frame, from_=0, to=1, variable=self.text_threshold, orient=tk.HORIZONTAL
                  ).grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="Low Text:").grid(row=1, column=0, padx=5)
        ttk.Scale(param_frame, from_=0, to=1, variable=self.low_text, orient=tk.HORIZONTAL
                  ).grid(row=1, column=1, padx=5)

        ttk.Label(param_frame, text="Link Threshold:").grid(row=2, column=0, padx=5)
        ttk.Scale(param_frame, from_=0, to=1, variable=self.link_threshold, orient=tk.HORIZONTAL
                  ).grid(row=2, column=1, padx=5)

        # Image preview
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=2, column=0, pady=5)

        # Table preview
        self.table = ttk.Treeview(main_frame)
        self.table.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.update_image_preview()

    def update_image_preview(self):
        if self.original_image is None:
            return

        preview = self.processed_image if self.processed_image is not None else self.original_image
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        # Resize for display
        height, width = preview.shape[:2]
        max_size = 500
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            preview = cv2.resize(preview, (int(width * scale), int(height * scale)))

        image = Image.fromarray(preview)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def process_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first")
            return

        try:
            # Initialize EasyOCR
            reader = easyocr.Reader(['en'])

            # Get OCR results
            results = reader.readtext(
                self.original_image,
                text_threshold=self.text_threshold.get(),
                low_text=self.low_text.get(),
                link_threshold=self.link_threshold.get()
            )

            self.ocr_results = [OCRResult(box=r[0], text=r[1], score=r[2]) for r in results]

            # Process grid
            height, width = self.original_image.shape[:2]
            detector = GridDetector(width, height)
            v_lines = detector.find_gaps(self.ocr_results, 'vertical')
            h_lines = detector.find_gaps(self.ocr_results, 'horizontal')

            # Get cell assignments
            self.cells = detector.assign_to_cells(self.ocr_results, v_lines, h_lines)

            # Update previews
            self.update_table_preview()
            self.update_processed_image()

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

    def update_processed_image(self):
        if not self.show_annotations.get():
            self.processed_image = self.original_image.copy()
            return

        self.processed_image = self.original_image.copy()
        for result in self.ocr_results:
            points = np.array(result.box, np.int32)
            cv2.polylines(self.processed_image, [points], True, (0, 255, 0), 2)

        self.update_image_preview()

    def update_table_preview(self):
        # Clear existing items
        for item in self.table.get_children():
            self.table.delete(item)

        # Get dimensions
        max_row = max(r for r, _ in self.cells.keys()) + 1
        max_col = max(c for _, c in self.cells.keys()) + 1

        # Configure columns
        self.table['columns'] = [f'Col{i}' for i in range(max_col)]
        for col in self.table['columns']:
            self.table.heading(col, text=col)

        # Add data
        for row in range(max_row):
            values = [self.cells.get((row, col), '') for col in range(max_col)]
            self.table.insert('', 'end', values=values)

    def toggle_annotations(self):
        if self.processed_image is not None:
            self.update_processed_image()

    def save_csv(self):
        if not hasattr(self, 'cells'):
            messagebox.showerror("Error", "No data to save")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if filepath:
            try:
                max_row = max(r for r, _ in self.cells.keys()) + 1
                max_col = max(c for _, c in self.cells.keys()) + 1

                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in range(max_row):
                        writer.writerow([self.cells.get((row, col), '') for col in range(max_col)])

                messagebox.showinfo("Success", "CSV file saved successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save CSV: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRSpreadsheetApp(root)
    root.mainloop()