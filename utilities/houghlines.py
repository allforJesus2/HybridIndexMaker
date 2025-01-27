import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from dataclasses import dataclass
from typing import Tuple, Optional
try:
    from utilities.extend_line_vertices import *
except:
    from extend_line_vertices import *


@dataclass
class CannyParams:
    low_threshold: int
    high_threshold: int
    aperture_size: int


@dataclass
class HoughParams:
    rho: int
    theta: float
    threshold: int
    min_line_length: int
    max_line_gap: int

@dataclass
class ExtensionParams:
    merge_threshold: int
    look_ahead: int
    max_neighbors: int

class HoughLinesApp:
    def __init__(self, window, params_callback=None, image=None, canny_params=None, hough_params=None,
                 extension_params=None):
        self.window = window
        self.window.title("Hough Lines GUI")
        self.params_callback = params_callback

        # Initialize parameters with defaults or provided values
        self.canny_params = canny_params or {'low_threshold': 50, 'high_threshold': 150, 'aperture_size': 3}
        self.hough_params = hough_params or {
            'rho': 1,
            'theta': np.pi / 180 * 180,
            'threshold': 100,
            'min_line_length': 100,
            'max_line_gap': 80
        }
        self.extension_params = extension_params or {
            'merge_threshold': 10,
            'look_ahead': 20,
            'max_neighbors':2,
        }

        # Create main container frame
        main_container = tk.Frame(window)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create left frame for controls with a canvas and scrollbar
        control_frame = tk.Frame(main_container)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Add a canvas and scrollbar to the control frame
        control_canvas = tk.Canvas(control_frame)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scrollbar
        v_scrollbar = ttk.Scrollbar(control_frame, orient=tk.VERTICAL, command=control_canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        control_canvas.configure(yscrollcommand=v_scrollbar.set)
        control_canvas.bind(
            '<Configure>',
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )

        # Create an inner frame to hold the controls
        inner_control_frame = tk.Frame(control_canvas)
        control_canvas.create_window((0, 0), window=inner_control_frame, anchor="nw")

        # Create right frame for canvas and scrollbars (unchanged)
        canvas_frame = tk.Frame(main_container)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbars (unchanged)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        # Create canvas (unchanged)
        self.canvas = tk.Canvas(canvas_frame,
                                xscrollcommand=h_scrollbar.set,
                                yscrollcommand=v_scrollbar.set)

        # Configure scrollbar commands (unchanged)
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        # Pack everything in the correct order (unchanged)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mouse wheel events for scrolling (unchanged)
        self.canvas.bind('<MouseWheel>', self._on_mousewheel_y)  # Windows
        self.canvas.bind('<Shift-MouseWheel>', self._on_mousewheel_x)  # Windows with Shift
        self.canvas.bind('<Button-4>', self._on_mousewheel_y)  # Linux scroll up
        self.canvas.bind('<Button-5>', self._on_mousewheel_y)  # Linux scroll down
        self.canvas.bind('<Shift-Button-4>', self._on_mousewheel_x)  # Linux with Shift
        self.canvas.bind('<Shift-Button-5>', self._on_mousewheel_x)  # Linux with Shift

        # Initialize variables (unchanged)
        self.image_path = None
        self.original_image = image if image is not None else None
        self.processed_image = None
        self.photo = None
        self.scale_factor = 1.0
        self.lines = None

        # Create image conversion options (now inside inner_control_frame)
        convert_frame = tk.LabelFrame(inner_control_frame, text="Image Conversion")
        convert_frame.pack(pady=5, padx=5, fill="x")

        self.convert_var = tk.StringVar(value="gray")
        tk.Radiobutton(convert_frame, text="Grayscale", variable=self.convert_var,
                       value="gray", command=self.update_image).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(convert_frame, text="Binary", variable=self.convert_var,
                       value="binary", command=self.update_image).pack(side=tk.LEFT, padx=10)

        # Add binary threshold slider (now inside inner_control_frame)
        self.binary_threshold = tk.Scale(inner_control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         command=self.update_image, label="Binary Threshold", length=200)
        self.binary_threshold.set(127)
        self.binary_threshold.pack(pady=5)

        # Create Canny edge detection sliders (now inside inner_control_frame)
        self.canny_low_slider = tk.Scale(inner_control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                         command=self.update_image, label="Canny Low Threshold", length=200)
        self.canny_low_slider.set(50)
        self.canny_low_slider.pack(pady=5)

        self.canny_high_slider = tk.Scale(inner_control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                          command=self.update_image, label="Canny High Threshold", length=200)
        self.canny_high_slider.set(150)
        self.canny_high_slider.pack(pady=5)

        # Create aperture size radio buttons (now inside inner_control_frame)
        aperture_frame = tk.LabelFrame(inner_control_frame, text="Aperture Size")
        aperture_frame.pack(pady=5, padx=5, fill="x")

        self.aperture_var = tk.IntVar(value=3)
        for size in [3, 5, 7]:
            tk.Radiobutton(aperture_frame, text=str(size), variable=self.aperture_var,
                           value=size, command=self.update_image).pack(side=tk.LEFT, padx=10)

        # Create Hough transform sliders (now inside inner_control_frame)
        self.rho_slider = tk.Scale(inner_control_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                   command=self.update_image, label="Rho", length=200)
        self.rho_slider.set(1)
        self.rho_slider.pack(pady=5)

        self.theta_slider = tk.Scale(inner_control_frame, from_=1, to=90, orient=tk.HORIZONTAL,
                                     command=self.update_image, label="Theta", length=200)
        self.theta_slider.set(180)
        self.theta_slider.pack(pady=5)

        self.threshold_slider = tk.Scale(inner_control_frame, from_=1, to=500, orient=tk.HORIZONTAL,
                                         command=self.update_image, label="Threshold", length=200)
        self.threshold_slider.set(100)
        self.threshold_slider.pack(pady=5)

        self.min_line_length_slider = tk.Scale(inner_control_frame, from_=1, to=500, orient=tk.HORIZONTAL,
                                               command=self.update_image, label="Min Line Length", length=200)
        self.min_line_length_slider.set(100)
        self.min_line_length_slider.pack(pady=5)

        self.max_line_gap_slider = tk.Scale(inner_control_frame, from_=1, to=200, orient=tk.HORIZONTAL,
                                            command=self.update_image, label="Max Line Gap", length=200)
        self.max_line_gap_slider.set(80)
        self.max_line_gap_slider.pack(pady=5)

        self.merge_threshold_slider = tk.Scale(inner_control_frame, from_=0, to=30, orient=tk.HORIZONTAL,
                                               command=self.update_image, label="Merge Threshold", length=200)
        self.merge_threshold_slider.set(self.extension_params['merge_threshold'])
        self.merge_threshold_slider.pack(pady=5)

        self.look_ahead_slider = tk.Scale(inner_control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                          command=self.update_image, label="Look Ahead", length=200)
        self.look_ahead_slider.set(self.extension_params['look_ahead'])
        self.look_ahead_slider.pack(pady=5)

        self.max_neighbors_slider = tk.Scale(inner_control_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                          command=self.update_image, label="Max Neighbors", length=200)
        self.max_neighbors_slider.set(self.extension_params['max_neighbors'])
        self.max_neighbors_slider.pack(pady=5)

        # Create buttons frame (now inside inner_control_frame)
        button_frame = tk.Frame(inner_control_frame)
        button_frame.pack(pady=10)

        # Add buttons (unchanged)
        tk.Button(button_frame, text="Open Image", command=self.open_image).pack(pady=2)
        tk.Button(button_frame, text="Save Output Image", command=self.save_output_image).pack(pady=2)
        tk.Button(button_frame, text="Save Lines Image", command=self.save_lines_image).pack(pady=2)
        tk.Button(button_frame, text="Get Parameters", command=self.get_parameters).pack(pady=2)

        # Set initial values for sliders based on parameters (unchanged)
        self.canny_low_slider.set(self.canny_params['low_threshold'])
        self.canny_high_slider.set(self.canny_params['high_threshold'])
        self.aperture_var.set(self.canny_params['aperture_size'])

        self.rho_slider.set(self.hough_params['rho'])
        self.theta_slider.set(self.hough_params['theta'] * 180 / np.pi)  # Convert radians to degrees
        self.threshold_slider.set(self.hough_params['threshold'])
        self.min_line_length_slider.set(self.hough_params['min_line_length'])
        self.max_line_gap_slider.set(self.hough_params['max_line_gap'])

        # Update image if initial image was provided (unchanged)
        if self.original_image is not None:
            self.update_image()

    def _on_mousewheel_y(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    def _on_mousewheel_x(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.xview_scroll(-1, "units")
        else:
            self.canvas.xview_scroll(1, "units")


    def get_parameters(self) -> Tuple[CannyParams, HoughParams, ExtensionParams]:
        """Get current parameters for Canny and HoughLinesP"""
        canny_params = CannyParams(
            low_threshold=self.canny_low_slider.get(),
            high_threshold=self.canny_high_slider.get(),
            aperture_size=self.aperture_var.get()
        )

        hough_params = HoughParams(
            rho=self.rho_slider.get(),
            theta=(np.pi / 180) * self.theta_slider.get(),
            threshold=self.threshold_slider.get(),
            min_line_length=self.min_line_length_slider.get(),
            max_line_gap=self.max_line_gap_slider.get()
        )

        extension_params = ExtensionParams(
            merge_threshold=self.merge_threshold_slider.get(),
            look_ahead=self.look_ahead_slider.get(),
            max_neighbors=self.max_neighbors_slider.get()
        )

        if self.params_callback:
            self.params_callback(canny_params, hough_params, extension_params)

        return canny_params, hough_params, extension_params

    def update_image(self, event=None):
        if self.original_image is not None:
            # Store current scroll position
            current_x = self.canvas.xview()[0]
            current_y = self.canvas.yview()[0]

            self.image = self.original_image.copy()

            # Get parameters through the new method
            canny_params, hough_params, extension_params = self.get_parameters()

            # Convert image based on selected method
            if self.convert_var.get() == "gray":
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:  # binary
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, gray = cv2.threshold(gray, self.binary_threshold.get(), 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(
                gray,
                canny_params.low_threshold,
                canny_params.high_threshold,
                apertureSize=canny_params.aperture_size
            )

            self.lines = cv2.HoughLinesP(
                edges,
                hough_params.rho,
                hough_params.theta,
                hough_params.threshold,
                minLineLength=hough_params.min_line_length,
                maxLineGap=hough_params.max_line_gap
            )

            if self.merge_threshold_slider.get() != 0 and self.look_ahead_slider.get() != 0:
                self.lines = extend_line_vertices_optimized(self.lines,int(self.merge_threshold_slider.get()),
                                                            int(self.look_ahead_slider.get()),
                                                            max_neighbors=int(self.max_neighbors_slider.get()))
                print('lines extended')

            if self.lines is not None:
                for line in self.lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    # Draw blue X markers at endpoints
                    marker_size = 4
                    # First endpoint
                    cv2.line(self.image,
                             (x1 - marker_size, y1 - marker_size),
                             (x1 + marker_size, y1 + marker_size),
                             (255, 0, 0), 1)  # Blue color
                    cv2.line(self.image,
                             (x1 - marker_size, y1 + marker_size),
                             (x1 + marker_size, y1 - marker_size),
                             (255, 0, 0), 1)

                    # Second endpoint
                    cv2.line(self.image,
                             (x2 - marker_size, y2 - marker_size),
                             (x2 + marker_size, y2 + marker_size),
                             (255, 0, 0), 1)
                    cv2.line(self.image,
                             (x2 - marker_size, y2 + marker_size),
                             (x2 + marker_size, y2 - marker_size),
                             (255, 0, 0), 1)

            # Keep original image size (scale = 1.0)
            self.processed_image = self.image

            # Create the PhotoImage at original size
            self.photo = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            )

            # Update canvas
            self.canvas.delete("all")

            # Create the image on the canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Update the scroll region to match the original image size
            image_height, image_width = self.image.shape[:2]
            self.canvas.config(scrollregion=(0, 0, image_width, image_height))

            # Restore previous scroll position
            self.canvas.xview_moveto(current_x)
            self.canvas.yview_moveto(current_y)


    # Other methods remain the same...
    def on_canvas_resize(self, event):
        if self.photo:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            if self.original_image is not None:
                image_height, image_width = self.original_image.shape[:2]
                self.canvas.config(scrollregion=(0, 0, image_width, image_height))

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.update_image()

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


# Example usage with callback
def parameter_callback(canny_params: CannyParams, hough_params: HoughParams, extension_params: ExtensionParams):
    print("\nCurrent Parameters:")
    print(f"Canny Parameters: {canny_params}")
    print(f"Hough Parameters: {hough_params}")
    print(f"Extension Parameters: {extension_params}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HoughLinesApp(root, params_callback=parameter_callback)
    root.mainloop()