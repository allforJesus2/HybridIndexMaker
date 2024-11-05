import tkinter as tk
from tkinter import filedialog, ttk
import os
import pickle
from PIL import Image, ImageTk, ImageDraw


class OCRViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("OCR Results Viewer")
        self.geometry("800x600")

        # Variables
        self.root_dir = tk.StringVar()
        self.search_term = tk.StringVar()
        self.crop_width = tk.StringVar(value="200")
        self.crop_height = tk.StringVar(value="200")
        self.matches = []

        self._create_widgets()

    def _create_widgets(self):
        # Directory selection
        dir_frame = ttk.Frame(self)
        dir_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(dir_frame, text="Root Directory:").pack(side='left')
        ttk.Entry(dir_frame, textvariable=self.root_dir, width=50).pack(side='left', padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._browse_directory).pack(side='left')

        # Search frame
        search_frame = ttk.Frame(self)
        search_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(search_frame, text="Search Word:").pack(side='left')
        ttk.Entry(search_frame, textvariable=self.search_term, width=30).pack(side='left', padx=5)
        ttk.Button(search_frame, text="Search", command=self._search).pack(side='left')

        # Crop dimensions
        crop_frame = ttk.Frame(self)
        crop_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(crop_frame, text="Crop Width:").pack(side='left')
        ttk.Entry(crop_frame, textvariable=self.crop_width, width=10).pack(side='left', padx=5)
        ttk.Label(crop_frame, text="Crop Height:").pack(side='left')
        ttk.Entry(crop_frame, textvariable=self.crop_height, width=10).pack(side='left', padx=5)

        # Results
        results_frame = ttk.Frame(self)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.results_tree = ttk.Treeview(results_frame, columns=('File', 'Word', 'Confidence'), show='headings')
        self.results_tree.heading('File', text='File')
        self.results_tree.heading('Word', text='Word')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.pack(fill='both', expand=True)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(button_frame, text="View Selected", command=self._view_selected).pack(side='left', padx=5)
        ttk.Button(button_frame, text="View All", command=self._view_all).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save with Boxes", command=self._save_with_boxes).pack(side='left')

    def _browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.root_dir.set(directory)

    def _search(self):
        self.matches.clear()
        self.results_tree.delete(*self.results_tree.get_children())

        search_term = self.search_term.get().lower()
        root_dir = self.root_dir.get()

        for folder in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, folder)):
                ocr_path = os.path.join(root_dir, folder, 'ocr.pkl')
                img_path = None

                # Find corresponding image file
                for file in os.listdir(os.path.join(root_dir, folder)):
                    if file.endswith('.png'):
                        img_path = os.path.join(root_dir, folder, file)
                        break

                if os.path.exists(ocr_path) and img_path:
                    with open(ocr_path, 'rb') as f:
                        ocr_results = pickle.load(f)

                    for result in ocr_results:
                        box, word, confidence = result
                        if search_term in word.lower():
                            x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
                            self.matches.append((img_path, box, word, confidence))
                            self.results_tree.insert('', 'end', values=(img_path, word, f"{confidence:.2f}"))

    def _view_selected(self):
        selection = self.results_tree.selection()
        if not selection:
            return

        selected_items = [self.matches[int(self.results_tree.index(item))] for item in selection]
        self._show_crops(selected_items)

    def _view_all(self):
        self._show_crops(self.matches)

    def _show_crops(self, matches):
        if not matches:
            return

        crop_width = int(self.crop_width.get())
        crop_height = int(self.crop_height.get())

        for img_path, box, word, confidence in matches:
            x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]

            # Calculate center and crop coordinates
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            crop_x1 = max(0, center_x - crop_width // 2)
            crop_y1 = max(0, center_y - crop_height // 2)
            crop_x2 = crop_x1 + crop_width
            crop_y2 = crop_y1 + crop_height

            # Open and crop image
            img = Image.open(img_path)
            crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Draw red box on the cropped region
            draw = ImageDraw.Draw(crop)
            local_x1 = x1 - crop_x1
            local_y1 = y1 - crop_y1
            local_x2 = x2 - crop_x1
            local_y2 = y2 - crop_y1
            draw.rectangle([(local_x1, local_y1), (local_x2, local_y2)], outline='red', width=2)

            # Create new window for crop
            window = tk.Toplevel(self)
            window.title(f"Match: {word} ({os.path.basename(img_path)})")

            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(crop)
            label = ttk.Label(window, image=photo)
            label.image = photo  # Keep a reference
            label.pack()

    def _save_with_boxes(self):
        selection = self.results_tree.selection()
        if not selection:
            return

        selected_items = [self.matches[int(self.results_tree.index(item))] for item in selection]
        if not selected_items:
            return

        # Group matches by image file
        image_matches = {}
        for img_path, box, word, confidence in selected_items:
            if img_path not in image_matches:
                image_matches[img_path] = []
            image_matches[img_path].append((box, word, confidence))

        # Process each image
        for img_path, matches in image_matches.items():
            # Open original image
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)

            # Draw red boxes for all matches in this image
            for box, word, _ in matches:
                x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)

            # Save the image
            output_path = os.path.join(os.path.dirname(img_path), 'ocr_view.png')
            img.save(output_path)

            # Open the saved image
            os.startfile(output_path)


if __name__ == "__main__":
    app = OCRViewer()
    app.mainloop()