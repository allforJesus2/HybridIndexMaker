import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def merge_lines(lines, distance_threshold):
    if lines is None:
        return None

    def distance(v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    merged_lines = []
    to_remove = set()

    for i in range(len(lines)):
        if i in to_remove:
            continue
        x1, y1, x2, y2 = lines[i][0]
        v1 = (x1, y1)
        v2 = (x2, y2)

        for j in range(i + 1, len(lines)):
            if j in to_remove:
                continue
            x3, y3, x4, y4 = lines[j][0]
            v3 = (x3, y3)
            v4 = (x4, y4)

            if (distance(v1, v3) < distance_threshold and distance(v2, v4) < distance_threshold) or \
                    (distance(v1, v4) < distance_threshold and distance(v2, v3) < distance_threshold):
                new_v1 = ((v1[0] + v3[0]) / 2, (v1[1] + v3[1]) / 2)
                new_v2 = ((v2[0] + v4[0]) / 2, (v2[1] + v4[1]) / 2)
                merged_lines.append([[int(new_v1[0]), int(new_v1[1]), int(new_v2[0]), int(new_v2[1])]])
                to_remove.add(i)
                to_remove.add(j)
                break

    for i in range(len(lines)):
        if i not in to_remove:
            merged_lines.append(lines[i])

    return np.array(merged_lines)

class HoughLinesApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hough Lines GUI")

        # Create a main frame to hold the canvas and slider frame
        main_frame = tk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a frame for the canvas
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.scale_factor = 1.0  # Initialize with a default value

        # Bind the canvas resize event to update the image
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Create a floating window for sliders and buttons
        self.control_window = tk.Toplevel(window)
        self.control_window.title("Controls")
        self.control_window.resizable(False, False)

        # Create a frame for the sliders
        slider_frame = tk.Frame(self.control_window)
        slider_frame.pack(side=tk.TOP, pady=5)

        self.image_path = None
        self.original_image = None  # Store the original image
        self.processed_image = None
        self.photo = None
        self.direct_mode = False  # Flag for direct mode

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.scale_factor = 1.0

        # Sliders for HoughLinesP parameters with labels
        self.rho_slider = tk.Scale(slider_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_image,
                                   label="Rho", length=300)
        self.rho_slider.set(1)
        self.rho_slider.pack(side=tk.TOP, pady=5)

        self.theta_slider = tk.Scale(slider_frame, from_=1, to=90, orient=tk.HORIZONTAL, command=self.update_image,
                                     label="Theta", length=300)
        self.theta_slider.set(180)
        self.theta_slider.pack(side=tk.TOP, pady=5)

        self.threshold_slider = tk.Scale(slider_frame, from_=1, to=500, orient=tk.HORIZONTAL, command=self.update_image,
                                         label="Threshold", length=300)
        self.threshold_slider.set(100)
        self.threshold_slider.pack(side=tk.TOP, pady=5)

        self.min_line_length_slider = tk.Scale(slider_frame, from_=1, to=500, orient=tk.HORIZONTAL,
                                               command=self.update_image, label="Min Line Length", length=300)
        self.min_line_length_slider.set(100)
        self.min_line_length_slider.pack(side=tk.TOP, pady=5)

        self.max_line_gap_slider = tk.Scale(slider_frame, from_=1, to=200, orient=tk.HORIZONTAL,
                                            command=self.update_image, label="Max Line Gap", length=300)
        self.max_line_gap_slider.set(80)
        self.max_line_gap_slider.pack(side=tk.TOP, pady=5)

        # Additional slider for line scale
        self.line_scale_slider = tk.Scale(slider_frame, from_=100, to=500, orient=tk.HORIZONTAL,
                                          command=self.update_image, label="Line Scale", length=300)
        self.line_scale_slider.set(1)
        self.line_scale_slider.pack(side=tk.TOP, pady=5)

        self.dilation_slider = tk.Scale(slider_frame, from_=0, to=20, orient=tk.HORIZONTAL, command=self.update_image,
                                        label="Dilation Kernel Size", length=300)
        self.dilation_slider.set(0)
        self.dilation_slider.pack(side=tk.TOP, pady=5)

        self.erosion_slider = tk.Scale(slider_frame, from_=0, to=20, orient=tk.HORIZONTAL, command=self.update_image,
                                       label="Erosion Kernel Size", length=300)
        self.erosion_slider.set(0)
        self.erosion_slider.pack(side=tk.TOP, pady=5)

        # Slider for vertex merging distance
        self.vertex_merge_slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                            command=self.update_image, label="Vertex Merge Distance", length=300)
        self.vertex_merge_slider.set(0)
        self.vertex_merge_slider.pack(side=tk.TOP, pady=5)

        # Slider for vertex merging distance
        self.vertex_merge_iterations_slider = tk.Scale(slider_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                                       command=self.update_image, label="Vertex Merge Iterations",
                                                       length=300)
        self.vertex_merge_iterations_slider.set(1)
        self.vertex_merge_iterations_slider.pack(side=tk.TOP, pady=5)

        self.threshold_value_slider = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                               command=self.update_image, label="Binary Threshold Value for Direct Mode", length=300)
        self.threshold_value_slider.set(0)
        self.threshold_value_slider.pack(side=tk.TOP, pady=5)

        # New slider for image scaling
        self.image_scale_slider = tk.Scale(slider_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                           command=self.update_image, label="Image Scale (%)", length=300)
        self.image_scale_slider.set(100)
        self.image_scale_slider.pack(side=tk.TOP, pady=5)

        # Create a frame for the buttons
        button_frame = tk.Frame(self.control_window)
        button_frame.pack(side=tk.TOP, pady=5)

        # Button to open file dialog
        self.open_button = tk.Button(button_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=5)

        self.save_output_button = tk.Button(button_frame, text="Save Output Image", command=self.save_output_image)
        self.save_output_button.pack(side=tk.LEFT, padx=5)

        self.save_lines_button = tk.Button(button_frame, text="Save Lines Image", command=self.save_lines_image)
        self.save_lines_button.pack(side=tk.LEFT, padx=5)

        self.apply_houghlines_directly_button = tk.Button(button_frame, text="Toggle Direct Mode",
                                                          command=self.toggle_direct_mode)
        self.apply_houghlines_directly_button.pack(side=tk.LEFT, padx=5)

    def on_canvas_resize(self, event):
        if self.photo:
            self.canvas.delete("all")
            canvas_width = event.width
            canvas_height = event.height
            if self.original_image is not None:
                original_height, original_width = self.original_image.shape[:2]
                self.scale_factor = min(canvas_width / original_width, canvas_height / original_height)
            self.photo = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)).resize(
                    (canvas_width, canvas_height))
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def save_output_image(self):
        if self.processed_image is not None:
            cv2.imwrite('output_image.jpg', self.processed_image)
            print("Output image saved as output_image.jpg")
            os.startfile('output_image.jpg')

    def save_lines_image(self):
        if self.lines is not None:
            lines_image = np.zeros(self.original_image.shape, dtype=np.uint8)
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imwrite('lines_image.jpg', lines_image)
            os.startfile('lines_image.jpg')
            print("Lines image saved as lines_image.jpg")

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)  # Store the original image
            self.update_image()

    def grayscale_threshold_invert(self, image):
        # Grayscale the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the threshold value from the slider
        threshold_value = self.threshold_value_slider.get()

        # Threshold the image to make it binary
        if threshold_value == 0:
            # Use Otsu's method if threshold is 0
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Use the threshold value from the slider
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Invert the binary image
        inverted = cv2.bitwise_not(binary)

        # Ensure the image is of type uint8 and has values of 0 and 255
        inverted = inverted.astype(np.uint8)

        return inverted

    def display_binary_image(self, binary_image):
        cv2.imshow('Binary Image', binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def toggle_direct_mode(self):
        self.direct_mode = not self.direct_mode
        print('Direct mode is ', self.direct_mode)
        self.update_image()
        print('line count: ', len(self.lines))

    def update_image(self, event=None):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            rho = self.rho_slider.get()
            theta = (np.pi / 180) * self.theta_slider.get()
            threshold = self.threshold_slider.get()
            min_line_length = self.min_line_length_slider.get()
            max_line_gap = self.max_line_gap_slider.get()
            line_scale = self.line_scale_slider.get() / 100
            dilation_size = self.dilation_slider.get()
            erosion_size = self.erosion_slider.get()
            vertex_merge_distance = self.vertex_merge_slider.get()
            vertex_merge_iterations = self.vertex_merge_iterations_slider.get()
            image_scale = self.image_scale_slider.get() / 100

            # Resize the image based on the image scale slider
            if image_scale != 1.0:
                self.image = cv2.resize(self.image, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)

            if self.direct_mode:
                binary = self.grayscale_threshold_invert(self.image)

                # Apply dilation
                if dilation_size > 0:
                    kernel = np.ones((dilation_size, dilation_size), np.uint8)
                    binary = cv2.dilate(binary, kernel, iterations=1)

                # Apply erosion
                if erosion_size > 0:
                    kernel = np.ones((erosion_size, erosion_size), np.uint8)
                    binary = cv2.erode(binary, kernel, iterations=1)

                self.lines = cv2.HoughLinesP(binary, rho, theta, threshold, minLineLength=min_line_length,
                                             maxLineGap=max_line_gap)
            else:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                # Apply dilation
                if dilation_size > 0:
                    kernel = np.ones((dilation_size, dilation_size), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)

                # Apply erosion
                if erosion_size > 0:
                    kernel = np.ones((erosion_size, erosion_size), np.uint8)
                    edges = cv2.erode(edges, kernel, iterations=1)

                self.lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length,
                                             maxLineGap=max_line_gap)

            if self.lines is not None:
                if vertex_merge_distance > 0:
                    for i in range(vertex_merge_iterations):
                        self.lines = merge_lines(self.lines, vertex_merge_distance)
                for line in self.lines:
                    x1, y1, x2, y2 = line[0]

                    dx = (x2 - x1) * (line_scale - 1)
                    dy = (y2 - y1) * (line_scale - 1)
                    x1_scaled = int(x1 - dx / 2)
                    y1_scaled = int(y1 - dy / 2)
                    x2_scaled = int(x2 + dx / 2)
                    y2_scaled = int(y2 + dy / 2)
                    cv2.line(self.image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 1)

        # After processing the image and drawing lines
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # Calculate the scaling factor to fit the image to the canvas
        image_height, image_width = self.image.shape[:2]
        self.scale_factor = min(canvas_width / image_width, canvas_height / image_height)

        # Resize the image using the scaling factor
        new_width = int(image_width * self.scale_factor)
        new_height = int(image_height * self.scale_factor)

        self.processed_image = cv2.resize(self.image, (new_width, new_height))

        # Create a blank image with canvas dimensions
        canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate offset to center the image
        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        # Place the resized image onto the blank canvas image
        canvas_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = self.processed_image

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def on_canvas_click(self, event):
        if self.lines is not None and self.original_image is not None:
            # Get the original image dimensions
            original_height, original_width = self.original_image.shape[:2]

            # Calculate the scaled image dimensions
            scaled_width = int(original_width * self.scale_factor)
            scaled_height = int(original_height * self.scale_factor)

            # Calculate the offset to center the scaled image on the canvas
            offset_x = max(0, (self.canvas.winfo_width() - scaled_width) // 2)
            offset_y = max(0, (self.canvas.winfo_height() - scaled_height) // 2)

            # Adjust click coordinates based on the scaling and offset
            click_x = (event.x - offset_x) / self.scale_factor
            click_y = (event.y - offset_y) / self.scale_factor

            # Ensure the click coordinates are within the bounds of the original image
            if 0 <= click_x < original_width and 0 <= click_y < original_height:
                for line in self.lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate the distance from the click to the line
                    distance = self.point_line_distance(click_x, click_y, x1, y1, x2, y2)
                    if distance < 5:  # Increased tolerance for easier clicking
                        self.change_line_color(line, (0, 0, 255))  # Changed to red (BGR format)
                        print(f"Clicked line: {line}")
                        # self.update_image()  # Refresh the image to show the color change
                        break

    def change_line_color(self, line, color):
        x1, y1, x2, y2 = line[0]
        cv2.line(self.image, (x1, y1), (x2, y2), color, 2)

        # Dynamically get the canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Resize the image to fit the canvas
        self.processed_image = cv2.resize(self.image, (canvas_width, canvas_height))

        # Convert the processed image to a format suitable for Tkinter
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)))

        # Clear the canvas and redraw the image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def merge_vertices(self, lines, distance_threshold):
        if lines is None:
            return None

        vertices = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            vertices.append((x1, y1))
            vertices.append((x2, y2))

        merged_vertices = []
        for vertex in vertices:
            if not merged_vertices:
                merged_vertices.append(vertex)
            else:
                merged = False
                for i, merged_vertex in enumerate(merged_vertices):
                    if np.linalg.norm(np.array(vertex) - np.array(merged_vertex)) < distance_threshold:
                        merged_vertices[i] = ((vertex[0] + merged_vertex[0]) / 2, (vertex[1] + merged_vertex[1]) / 2)
                        merged = True
                        break
                if not merged:
                    merged_vertices.append(vertex)

        merged_lines = []
        for i in range(0, len(merged_vertices), 2):
            if i + 1 < len(merged_vertices):
                merged_lines.append([[int(merged_vertices[i][0]), int(merged_vertices[i][1]),
                                      int(merged_vertices[i + 1][0]), int(merged_vertices[i + 1][1])]])

        return np.array(merged_lines)

if __name__ == "__main__":
    root = tk.Tk()
    app = HoughLinesApp(root)
    root.mainloop()
'''
the idea is to break the pid into lines
In order to collect the lines we perform ocr then use a regular expression to box each of the line numbers.
Next we use houghlines to generate a collection of lines.
Lines that have endpoints within a certain distance are grouped together
If a hough line intersects a box (expanded by a certain amount) that line is registered to that line number (line_no:[line,...])
first order and 2nd order lines
All lines that have been registered are whited out
We then iterate though all the registered lines placeing them back on the image one at a time
That line is drawn in with the line color and we use the flood fill algorithm to color the image and assign instruments the line number

Now we have exceptions
sometimes line numbers are dpecited
'''