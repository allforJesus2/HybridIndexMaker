# add box functionality

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class DetectoResultsEdit:
    def __init__(self, root, img, labels, boxes, scores, low_score, callback=None):


        if isinstance(img, np.ndarray):
            # Step 2: Convert the image from BGR to RGB
            cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Step 3: Convert the image to PIL format
            self.img = Image.fromarray(cv2_image_rgb)
        else:
            self.img = img

        self.labels = labels
        self.boxes = boxes
        self.scores = scores
        self.low_score = low_score
        self.window_width = 1000
        self.scale = self.window_width / self.img.width  # Assuming window width is 800
        self.root = root# tk.Tk()
        self.root.title("Image Display")
        self.canvas = None
        self.image_tk = None  # Initialize the image object variable

        self.callback = callback

        self.mouse_pressed = False  # Track if the mouse is pressed
        self.start_x = 0  # Initial click position
        self.start_y = 0
        self.current_box = None  # Current box being drawn

        self.selected_label = tk.StringVar()  # Variable to store the selected label
        self.labels_list = list(set(self.labels))  # Example labels
        self.initialize_canvas()
        self.draw_image()
        self.setup_event_binding()
        self.create_dropdown_menu()  # Create the dropdown menu
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Bind the close event to a custom method
        self.root.mainloop()

    def on_close(self):
        self.root.destroy()  # This is necessary to ensure the window is properly closed
        self.return_values = (self.labels, self.boxes, self.scores)  # Store the return values
        if self.callback:
            self.callback(self.return_values)

    def create_dropdown_menu(self):
        # Create the dropdown menu
        self.dropdown_menu = tk.OptionMenu(self.root, self.selected_label, *self.labels_list)
        self.dropdown_menu.pack(side=tk.BOTTOM)  # Position the dropdown menu

    def initialize_canvas(self):
        aspect_ratio = self.img.width / self.img.height
        # print('aspect ratio', aspect_ratio)
        self.window_height = int(self.window_width / aspect_ratio)
        self.canvas = tk.Canvas(self.root, width=self.window_width, height=self.window_height)
        self.canvas.pack()

    def draw_image(self):
        print((self.window_width, self.window_height))
        img_resized = self.img.resize((self.window_width, self.window_height))
        self.image_tk = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
        # print('image created')
        self.draw_boxes()

    def draw_boxes(self):
        for i in range(len(self.labels)):
            if self.scores[i] > self.low_score:
                x1, y1, x2, y2 = [int(x * self.scale) for x in self.boxes[i]]
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='red')
                text_x = x1
                text_y = y1
                self.canvas.create_text(text_x, text_y, text=f"{self.labels[i]} - {self.scores[i]:.2f}", anchor=tk.NW)

    def setup_event_binding(self):
        self.canvas.bind('<Button-1>', self.start_drawing)  # Left-click to start drawing
        self.canvas.bind('<B1-Motion>', self.draw_box)  # Drag to draw
        self.canvas.bind('<ButtonRelease-1>', self.end_drawing)  # Release to finalize
        self.canvas.bind('<Button-3>', self.remove_box)

    def start_drawing(self, event):
        self.mouse_pressed = True
        self.start_x = event.x
        self.start_y = event.y

    def draw_box(self, event):
        if self.mouse_pressed:
            if self.current_box:
                self.canvas.delete(self.current_box)  # Remove the previous box
            x1, y1 = self.start_x, self.start_y
            x2, y2 = event.x, event.y

            self.current_box = self.canvas.create_rectangle(x1, y1, x2, y2, outline='orange')

    def end_drawing(self, event):
        if self.mouse_pressed:
            self.mouse_pressed = False
            if self.current_box:
                self.canvas.delete(self.current_box)  # Remove the temporary box

                x1, y1, x2, y2 = [self.start_x, self.start_y, event.x, event.y]

                # swap if you do the reverso box
                if y2 < y1:
                    y1, y2 = y2, y1
                if x2 < x1:
                    x1, x2 = x2, x1

                # Add the new box to the list
                coords = [x1, y1, x2, y2]
                coords = [x / self.scale for x in coords]
                self.boxes = np.append(self.boxes, [coords], axis=0)
                # Optionally, add a label and score for the new box
                self.labels = np.append(self.labels, [self.selected_label.get()])
                self.scores = np.append(self.scores, [1.0])  # Example score
                self.draw_image()  # Redraw the image with the new box

    def remove_box(self, event):
        index = self.find_index(event.x, event.y, self.boxes)
        print('index', index)
        if index != -1:
            # print('len labels',len(self.boxes))
            # Create new lists excluding the clicked box
            self.labels = np.delete(self.labels, index)
            self.boxes = np.delete(self.boxes, index, axis=0)
            self.scores = np.delete(self.scores, index)
            # Redraw the remaining boxes and labels
            print('len labels', len(self.boxes))
            self.draw_image()
            self.draw_boxes()

    def find_index(self, x, y, boxes):
        print(x, y)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(x * self.scale) for x in boxes[i]]

            if x1 <= x and x <= x2 and y1 <= y and y <= y2:
                print(x1, y1, x2, y2)
                return i
        return -1

"""
img =Image.open(imgpath)
low_score = 0.5
labels, boxes, scores = ImageDisplay(img, labels, boxes, scores, low_score).return_values
"""