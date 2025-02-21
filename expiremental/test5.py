import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

class ContourDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contour Detection using OpenCV")

        # Initialize the image and contour variables
        self.original_image = None
        self.contoured_image = None

        # Load button
        load_button = Button(root, text="Load Image", command=self.load_image)
        load_button.pack(pady=10)

        # Detect contours button
        detect_button = Button(root, text="Detect Contours", command=self.detect_contours)
        detect_button.pack(pady=10)

    def load_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.contoured_image = self.original_image.copy()
            #cv2.imshow('Original Image', self.original_image)

    def detect_contours(self):
        # Ensure an image is loaded
        if self.original_image is None:
            return

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on a blank image (optional)
        contour_image = self.original_image.copy()
        cv2.drawContours(contour_image, contours, -1, color=(0, 255, 0), thickness=1)

        # Display the result
        self.contoured_image = contour_image

        # Convert the image to a Tkinter-compatible format for display
        img = Image.fromarray(cv2.cvtColor(self.contoured_image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label_contours.config(image=imgtk)
        label_contours.image = imgtk

if __name__ == "__main__":
    root = Tk()
    app = ContourDetectionApp(root)
    label_contours = Label(root)
    label_contours.pack(pady=10)

    root.mainloop()
