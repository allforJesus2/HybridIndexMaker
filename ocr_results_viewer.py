import tkinter as tk
from tkinter import filedialog, ttk
import os
import pickle
from PIL import Image, ImageTk, ImageDraw


class CropWindow:
    def __init__(self, parent, img_path, box, word, initial_crop_size=(200, 200)):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Match: {word} ({os.path.basename(img_path)})")

        # Store parameters
        self.img_path = img_path
        self.box = box
        self.word = word
        self.crop_size = list(initial_crop_size)  # [width, height]
        self.step_size = 50  # pixels to increase/decrease by

        # Create widgets
        self.create_widgets()

        # Initial crop
        self.update_crop()

    def create_widgets(self):
        # Control frame at the top
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill='x', padx=5, pady=5)

        # Size controls
        size_frame = ttk.LabelFrame(control_frame, text="Crop Size")
        size_frame.pack(side='left', padx=5)

        # Width controls
        width_frame = ttk.Frame(size_frame)
        width_frame.pack(side='left', padx=5)
        ttk.Label(width_frame, text="Width:").pack(side='left')
        ttk.Button(width_frame, text="-", width=2,
                   command=lambda: self.adjust_size(-self.step_size, 0)).pack(side='left', padx=2)
        self.width_label = ttk.Label(width_frame, text=str(self.crop_size[0]))
        self.width_label.pack(side='left', padx=2)
        ttk.Button(width_frame, text="+", width=2,
                   command=lambda: self.adjust_size(self.step_size, 0)).pack(side='left', padx=2)

        # Height controls
        height_frame = ttk.Frame(size_frame)
        height_frame.pack(side='left', padx=5)
        ttk.Label(height_frame, text="Height:").pack(side='left')
        ttk.Button(height_frame, text="-", width=2,
                   command=lambda: self.adjust_size(0, -self.step_size)).pack(side='left', padx=2)
        self.height_label = ttk.Label(height_frame, text=str(self.crop_size[1]))
        self.height_label.pack(side='left', padx=2)
        ttk.Button(height_frame, text="+", width=2,
                   command=lambda: self.adjust_size(0, self.step_size)).pack(side='left', padx=2)

        # Step size controls
        step_frame = ttk.LabelFrame(control_frame, text="Step Size")
        step_frame.pack(side='left', padx=5)
        ttk.Button(step_frame, text="200px",
                   command=lambda: self.set_step_size(200)).pack(side='left', padx=2)
        ttk.Button(step_frame, text="500px",
                   command=lambda: self.set_step_size(500)).pack(side='left', padx=2)
        ttk.Button(step_frame, text="1000px",
                   command=lambda: self.set_step_size(1000)).pack(side='left', padx=2)

        # View Original button
        view_frame = ttk.Frame(control_frame)
        view_frame.pack(side='left', padx=5)
        ttk.Button(view_frame, text="View Original",
                   command=self.open_original_image).pack(side='left', padx=2)

        # Image label
        self.image_label = ttk.Label(self.window)
        self.image_label.pack(padx=5, pady=5)

    def open_original_image(self):
        """Open the original image using the system's default image viewer"""
        if os.path.exists(self.img_path):
            os.startfile(self.img_path)

    def set_step_size(self, size):
        self.step_size = size

    def adjust_size(self, width_delta, height_delta):
        # Update crop size (minimum 50x50)
        self.crop_size[0] = max(50, self.crop_size[0] + width_delta)
        self.crop_size[1] = max(50, self.crop_size[1] + height_delta)

        # Update labels
        self.width_label.config(text=str(self.crop_size[0]))
        self.height_label.config(text=str(self.crop_size[1]))

        # Update crop
        self.update_crop()

    def update_crop(self):
        # Original box coordinates
        x1, y1, x2, y2 = self.box[0][0], self.box[0][1], self.box[2][0], self.box[2][1]

        # Calculate center of the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Calculate crop coordinates
        crop_x1 = max(0, center_x - self.crop_size[0] // 2)
        crop_y1 = max(0, center_y - self.crop_size[1] // 2)
        crop_x2 = crop_x1 + self.crop_size[0]
        crop_y2 = crop_y1 + self.crop_size[1]

        # Open and crop image
        img = Image.open(self.img_path)
        # Ensure crop doesn't exceed image bounds
        crop_x2 = min(crop_x2, img.width)
        crop_y2 = min(crop_y2, img.height)
        crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Draw red box on the cropped region
        draw = ImageDraw.Draw(crop)
        local_x1 = x1 - crop_x1
        local_y1 = y1 - crop_y1
        local_x2 = x2 - crop_x1
        local_y2 = y2 - crop_y1
        draw.rectangle([(local_x1, local_y1), (local_x2, local_y2)], outline='red', width=2)

        # Update displayed image
        photo = ImageTk.PhotoImage(crop)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference

class OCRViewer:
    def __init__(self, parent, folder=None):
        # Store the parent window reference
        self.parent = parent

        # Configure the parent window
        self.parent.title("OCR Results Viewer")
        self.parent.geometry("800x600")

        # Variables
        self.root_dir = tk.StringVar(value=folder if folder else "")
        self.search_term = tk.StringVar()
        self.crop_width = tk.StringVar(value="200")
        self.crop_height = tk.StringVar(value="200")
        self.matches = []

        # Keep track of opened windows
        self.opened_windows = []

        self._create_widgets()

        # If a folder was provided, automatically perform search
        if folder:
            self._search()

    def _create_widgets(self):
        # Directory selection
        dir_frame = ttk.Frame(self.parent)
        dir_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(dir_frame, text="Root Directory:").pack(side='left')
        ttk.Entry(dir_frame, textvariable=self.root_dir, width=50).pack(side='left', padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._browse_directory).pack(side='left')

        # Search frame
        search_frame = ttk.Frame(self.parent)
        search_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(search_frame, text="Search Word:").pack(side='left')
        search_entry = ttk.Entry(search_frame, textvariable=self.search_term, width=30)
        search_entry.pack(side='left', padx=5)
        # Bind Enter key to search
        search_entry.bind('<Return>', lambda event: self._search())
        ttk.Button(search_frame, text="Search", command=self._search).pack(side='left')

        # Crop dimensions
        crop_frame = ttk.Frame(self.parent)
        crop_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(crop_frame, text="Default Crop Width:").pack(side='left')
        ttk.Entry(crop_frame, textvariable=self.crop_width, width=10).pack(side='left', padx=5)
        ttk.Label(crop_frame, text="Default Crop Height:").pack(side='left')
        ttk.Entry(crop_frame, textvariable=self.crop_height, width=10).pack(side='left', padx=5)

        # Results
        results_frame = ttk.Frame(self.parent)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.results_tree = ttk.Treeview(results_frame, columns=('File', 'Word', 'Confidence'), show='headings')
        self.results_tree.heading('File', text='File')
        self.results_tree.heading('Word', text='Word')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.pack(fill='both', expand=True)
        self.results_tree.bind('<Double-1>', lambda event: self._view_selected())
        # Buttons
        button_frame = ttk.Frame(self.parent)
        button_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(button_frame, text="View Selected", command=self._view_selected).pack(side='left', padx=5)
        ttk.Button(button_frame, text="View All", command=self._view_all).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save with Boxes", command=self._save_with_boxes).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close All Windows", command=self._close_all_windows).pack(side='left', padx=5)

    def _close_all_windows(self):
        """Close all opened popup windows"""
        for window in self.opened_windows:
            if window.winfo_exists():  # Check if window still exists
                window.destroy()
        self.opened_windows.clear()

    def _browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.root_dir.set(directory)

    def _search(self):
        self.matches.clear()
        self.results_tree.delete(*self.results_tree.get_children())

        search_term = self.search_term.get().lower()
        root_dir = self.root_dir.get()

        if not root_dir:
            return

        for folder in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, folder)):
                ocr_path = os.path.join(root_dir, folder, 'ocr.pkl')
                img_path = None

                # Get the original page number from the folder name (e.g., 'page_16_results' -> 'page_16.png')
                original_page = folder.replace('_results', '.png')
                img_path = os.path.join(root_dir, original_page)

                # Check if the image exists
                if not os.path.exists(img_path):
                    continue

                if os.path.exists(ocr_path) and img_path:
                    with open(ocr_path, 'rb') as f:
                        ocr_results = pickle.load(f)

                    for result in ocr_results:
                        box, word, confidence = result
                        if search_term in word.lower():
                            self.matches.append((img_path, box, word, confidence))
                            self.results_tree.insert('', 'end',
                                                     values=(os.path.basename(img_path), word, f"{confidence:.2f}"))

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

        initial_crop_size = (int(self.crop_width.get()), int(self.crop_height.get()))

        for img_path, box, word, confidence in matches:
            # Create new window with crop controls
            crop_window = CropWindow(self.parent, img_path, box, word, initial_crop_size)
            self.opened_windows.append(crop_window.window)

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
    root = tk.Tk()
    app = OCRViewer(root)
    root.mainloop()