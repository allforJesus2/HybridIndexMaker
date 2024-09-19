import tkinter as tk
from tkinter import filedialog, colorchooser, ttk
from PIL import Image, ImageDraw
from os import startfile
class FindAnInstrumentApp:
    def __init__(self, root, img_path=''):
        self.root = root
        root.title("FAIA (find an instrument app)")
        # Widgets
        self.image_path_label = tk.Label(root, text="Image Path:")
        self.image_path_entry = tk.Entry(root, width=50)
        self.browse_button = tk.Button(root, text="Browse", command=self.browse_image)

        self.coords_label = tk.Label(root, text="Coordinates ([x1, y1, x2, y2], one per line):")
        self.coords_text = tk.Text(root, height=10, width=30)

        self.color_label = tk.Label(root, text="Color:")
        self.color_combo = tk.StringVar()
        self.color_combobox = tk.ttk.Combobox(root, textvariable=self.color_combo, values=["Red", "Green", "Blue", "Orange", "Yellow"], state='readonly')
        self.color_combobox.current(0)

        self.thickness_label = tk.Label(root, text="Line Thickness:")
        self.thickness_entry = tk.Entry(root, width=10)
        self.thickness_entry.insert(tk.END, '5')

        self.apply_button = tk.Button(root, text="Apply", command=self.draw_rectangles)  # Renamed method

        # Layout
        self.image_path_label.grid(row=0, column=0, padx=10, pady=10)
        self.image_path_entry.grid(row=0, column=1, padx=10, pady=10)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)
        self.image_path_entry.insert(tk.END, img_path)

        self.coords_label.grid(row=1, column=0, padx=10, pady=10)
        self.coords_text.grid(row=1, column=1, rowspan=3, padx=10, pady=10)

        self.color_label.grid(row=4, column=0, padx=10, pady=10)
        self.color_combobox.grid(row=4, column=1, padx=10, pady=10)

        self.thickness_label.grid(row=5, column=0, padx=10, pady=10)
        self.thickness_entry.grid(row=5, column=1, padx=10, pady=10)

        self.apply_button.grid(row=6, columnspan=3, pady=20)

    def browse_image(self):
        self.image_path_entry.delete(0, tk.END)
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        self.image_path_entry.insert(tk.END, file_path)

    def draw_rectangles(self):  # Renamed and updated method
        try:
            image_path = self.image_path_entry.get().strip()
            coords_lines = self.coords_text.get("1.0", tk.END).strip().split('\n')  # Get lines of coordinates
            color = self.color_combobox.get()
            thickness = int(self.thickness_entry.get())

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            for coords_str in coords_lines:
                if coords_str.startswith('tensor'):
                    remove = ['(',')','[',']',' ','tensor']
                    for rm in remove:
                        coords_str = coords_str.replace(rm, '')
                    coords = list(map(round, map(float, coords_str.split(','))))
                else:
                    coords = list(map(int, coords_str.strip('[]').split(',')))
                draw.rectangle(coords, outline=color, width=thickness)

            # Save and display modified image
            img.save("modified_image.png")
            startfile("modified_image.png")
        except Exception as e:
            print(f"An error occurred: {e}")


