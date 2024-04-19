'''
lets start with an image
load the image to a canvas
lets predfine the window as 1200x900
scale the image to fit the canvas
let the user draw boxes to capture regions
perform object detection and instrument recognition on captured regions
identify if button is pressed append to xlwings document
have a title bar that displays the mode
have keys that switch modes
enter can write group


so this is the ideal workflow
screen opens with blank canvas and a open file button
load an pdf/image/folder
first page is rendered on canvas
begin mode shows capture PID (P)
capture instrument group (I)
user draws a box on a region containing some instruments
consol prints instruments and inst numbers captured
user presses L to switch to line capture. draws box and ocr is performed to capture line
I is pressed to get service in
O is pressed to get service out *optionally E is pressed to get equipment
consol prints ready to write
pressing enter appends rows to sheet

after all are captured user presses N for next which opens up the next image


at the core we load an image and blit it to the screen
we have a top bar with buttons: capture pid, instrument group, line, service out, service in,
 equipment, write to xlwings

prompt sequence
write me a tkinter app that can load an image from a folder and display it scaled to 1200x900 maintaining aspect ratio.
the top bar is where the user can open the folder containing the images.
there will be a next button to load the next image

make it so the user draw boxes on the image that crop and save the image.
make it so we have a write to xlsx button to write the contents of the ocr to the xlwings document

TODO
OPEN WINDOW NEXT TO
CORRECTION FUNCTION
RIGHT CLICK TO REMOVE BOX
SERVICES WITH NO LINE AND EQUIP CONFLICT
LEFT CLICKING NO WHERE DUPLICATES INSTRUMENTS
MAKE DPI POP UP IN CENTER
ZOOM WINDOW SNIP
CLEARER TEXT
BIGGER TEXT FOR MODE
ONLY USE CV2
INSTRUMENT BLACKLIST
OCR WHITELIST
configuration: inst scaling, reader settings, blacklist
refactor release box
set minscore inst

FIX LINE EQUIP CONFLICT
THICK LINE FOR CURRENT BOX
ZOOM
RIGHT CLICK DELETE BOX
REMEMBER OCR BOX LEFT CLICK
ONLY INST
'''
# pip install tkinter PIL xlwings detecto easyocr opencv-python numpy matplotlib torch PyMuPDF
from functions import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import xlwings as xw
from detecto.core import Model
import easyocr
import cv2
import numpy as np
import importlib.util

class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")



        #reader settings
        # Initialize OCR settings
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 0.4
        self.min_size = 10
        self.ycenter_ths = 0.5
        self.height_ths = 0.5
        self.width_ths = 0.5
        self.add_margin = 0.1

        self.line = None
        self.service_in = None
        self.service_out = None
        self.equipment = None
        self.inst_data = None
        self.pid = None
        self.persistent_boxes = []
        self.persistent_texts = []
        self.correct_fn = None
        self.correct_fn_path = None

        self.image_list = []
        self.current_image_index = 0
        self.mouse_pressed = False
        self.start_x = 0
        self.start_y = 0
        self.current_box = None
        self.wb = None
        self.sheet = None
        self.capture = 'pid'
        self.capture_actions = {
            'pid': self.capture_pid,
            'instruments': self.capture_instruments,
            'line': self.capture_line,
            'equipment': self.capture_equipment,
            'service_in': self.capture_service_in,
            'service_out': self.capture_service_out
        }

        print('loading reader')
        self.reader = easyocr.Reader(['en'])
        print('reader loaded')

        self.labels = ['inst', 'dcs', 'ball', 'globe', 'diaphragm', 'knife', 'vball', 'plug', 'butterfly', 'gate']

        try:
            self.model_inst_path = r"models\saved_model_vid-v3.18_GEVO.pth"
            # load instrument recognition model
            print('loading model')
            self.model_inst = Model.load(self.model_inst_path, self.labels)
            print('model loaded')
        except Exception as e:
            print('Error',e)
            print('load model failed. you will likely have to load the model from command')

        self.minscore_inst = 0.7
        self.inst_data = []


        # Create a menu bar
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Folder", command=self.open_folder)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # Create a commands menu
        self.commands_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.commands_menu.add_command(label="Next", command=self.next_image)
        self.commands_menu.add_command(label="Previous", command=self.previous_image)
        self.commands_menu.add_command(label="Capture PID", command=lambda: self.set_capture('pid'))
        self.commands_menu.add_command(label="Capture Instrument Group", command=lambda: self.set_capture('instruments'))
        self.commands_menu.add_command(label="Capture Line", command=lambda: self.set_capture('line'))
        self.commands_menu.add_command(label="Capture Equipment", command=lambda: self.set_capture('equipment'))
        self.commands_menu.add_command(label="Capture Service In", command=lambda: self.set_capture('service_in'))
        self.commands_menu.add_command(label="Capture Service Out", command=lambda: self.set_capture('service_out'))
        self.commands_menu.add_command(label="Append Data to Index", command=self.append_data_to_excel)
        self.commands_menu.add_command(label="Clear instrument group", command=self.clear_instrument_group)
        self.commands_menu.add_command(label="Go to Page", command=self.open_go_to_page)
        self.commands_menu.add_command(label="Create images from PDF", command=self.open_pdf2png)
        self.commands_menu.add_command(label="Clear boxes", command=self.clear_boxes)
        self.commands_menu.add_command(label="Load Tag Correction Function", command=self.load_correct_fn)
        self.commands_menu.add_command(label="Load Object detection model", command=self.load_model)
        self.commands_menu.add_command(label="Merge pdfs", command=self.merge_pdfs)

        self.menu_bar.add_cascade(label="Commands", menu=self.commands_menu)

        self.root.config(menu=self.menu_bar)
        self.capture='pid'
        self.pid_coords = None

        # Create a canvas to display the image
        self.canvas = tk.Canvas(self.root, width=1200, height=900)
        self.canvas.pack()
        # Create a separate window for displaying captured data
        self.data_window = tk.Toplevel(self.root)
        self.data_window.title("Captured Data")
        self.data_window.geometry("400x400")

        # Create a Text widget to display the captured data
        self.data_text = tk.Text(self.data_window, wrap=tk.WORD)
        self.data_text.pack(fill=tk.BOTH, expand=True)

        # Bind the window close event to update the data display
        self.data_window.protocol("WM_DELETE_WINDOW", self.update_data_display)
        # Ensure data_window is always on top of the root window
        self.root.bind("<FocusIn>", lambda event: self.data_window.lift())

        #key bindings
        # Bind key shortcuts to the respective commands
        self.root.bind('n', lambda event: self.next_image())
        self.root.bind('b', lambda event: self.previous_image())
        self.root.bind('p', lambda event: self.set_capture('pid'))
        self.root.bind('a', lambda event: self.set_capture('instruments'))
        self.root.bind('f', lambda event: self.set_capture('line'))
        self.root.bind('e', lambda event: self.set_capture('equipment'))
        self.root.bind('z', lambda event: self.set_capture('service_in'))
        self.root.bind('x', lambda event: self.set_capture('service_out'))
        self.root.bind('w', lambda event: self.append_data_to_excel())
        self.root.bind('c', lambda event: self.clear_instrument_group())
        # key bindings
        # Bind key shortcuts to the respective commands
        self.root.bind('N', lambda event: self.next_image())
        self.root.bind('B', lambda event: self.previous_image())
        self.root.bind('P', lambda event: self.set_capture('pid'))
        self.root.bind('A', lambda event: self.set_capture('instruments'))
        self.root.bind('F', lambda event: self.set_capture('line'))
        self.root.bind('E', lambda event: self.set_capture('equipment'))
        self.root.bind('Z', lambda event: self.set_capture('service_in'))
        self.root.bind('X', lambda event: self.set_capture('service_out'))
        self.root.bind('W', lambda event: self.append_data_to_excel())
        self.root.bind('C', lambda event: self.clear_instrument_group())

        # shift key binding
        self.shift_held = False
        self.root.bind('<KeyPress-Shift_L>', self.shift_pressed)
        self.root.bind('<KeyRelease-Shift_L>', self.shift_released)

        # Bind mouse events for cropping
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_box)
        self.canvas.bind('<ButtonRelease-1>', self.end_drawing)

    def merge_pdfs(self):
        folder_that_has_pdfs = filedialog.askdirectory(title='Folder that has PDFs')
        merge_pdf(folder_that_has_pdfs)

    #def create_ocr_settings_gui(self):
    def shift_pressed(self, event):
        self.shift_held = True

    def shift_released(self, event):
        self.shift_held = False
    def update_capture_text(self, event):
        # Update the text's position to follow the cursor
        x, y = event.x, event.y
        self.canvas.coords(self.capture_text, x-16, y-8)

    def load_correct_fn(self):
        # Create a module specification
        self.correct_fn_path = filedialog.askopenfilename(filetypes=[('Python Files', '*.py')])
        if self.correct_fn_path:
            try:
                spec = importlib.util.spec_from_file_location('my_module', self.correct_fn_path)

                # Create a module from the specification
                module = importlib.util.module_from_spec(spec)

                # Load the module
                spec.loader.exec_module(module)

                # Access the 'correct' function from the module
                # correct(tag, tag_no)
                self.correct_fn = module.correct
                print('loaded correct fn')
            except Exception as e:
                print('faild to load correct fn', e)
                self.correct_fn = None

    def load_model(self):
        # Create a module specification
        self.model_inst_path = filedialog.askopenfilename(filetypes=[('PTH Files', '*.pth')])
        #labels = ['inst', 'dcs', 'ball', 'globe', 'diaphragm', 'knife', 'vball', 'plug', 'butterfly', 'gate']
        # self.model_inst_path = r"models\saved_model_vid-v3.18_GEVO.pth"
        # load instrument recognition model
        print('loading model')
        self.model_inst = Model.load(self.model_inst_path, self.labels)

    def open_go_to_page(self):

        page_index = tk.simpledialog.askinteger("Go to Page", "Enter the page index:")

        if page_index is not None:
            self.go_to_page(page_index - 1)  # Adjust the index since it's 0-based
            self.clear_boxes()

    def open_pdf2png(self):
        # Ask for DPI value
        dpi = tk.simpledialog.askinteger("DPI", "Enter the desired DPI value:")

        # Open file dialog to select PDF file
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        pdf_file = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF Files", "*.pdf")])

        if pdf_file and dpi:
            # Call the pdf2png function with the selected PDF file and DPI value
            pdf2png(pdf_file, dpi)

    def clear_instrument_group(self):

        if self.inst_data:
            self.canvas.delete(self.persistent_boxes[-1])  # Remove the previous box
        self.inst_data = []
        self.update_data_display()

    def set_capture(self, capture_type):
        self.capture = capture_type
        self.canvas.itemconfig(self.capture_text, text=self.capture)

    def open_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.image_list = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            self.current_image_index = 0
            self.load_image()

    def append_data_to_excel(self):

        try:
            # Example data to append
            file = 'index.xlsx'
            workbook_path = os.path.join(self.folder_path, file)

            # Check if the file exists, if not, create a new workbook
            if not os.path.exists(workbook_path):
                self.wb = xw.Book()  # Create a new workbook
                self.wb.save(workbook_path)  # Save the workbook to the specified path
                #self.wb.close()  # Close the workbook

            # Open the workbook (it will be opened in the background)
            wb = xw.Book(workbook_path)
            # Check if the 'Instrument Index' sheet exists, if not, create it
            if 'Instrument Index' not in wb.sheet_names:
                wb.sheets.add(name='Instrument Index')

                # Define the header row
                header = ['PID', 'TAG', 'TAG_NO', 'LABEL', 'LINE/EQUIP', 'SERVICE']

                # Write the header row to the first sheet of the workbook
                sheet = wb.sheets['Instrument Index']
                for i, column_header in enumerate(header, start=1):
                    sheet.range((1, i)).value = column_header


            ws = wb.sheets['Instrument Index']
            for data in self.inst_data:
                last_row = ws.range('A1').expand('down').last_cell.row
                ws.range(last_row+1, 1).value = self.pid
                ws.range(last_row + 1, 2).value = data['tag']
                ws.range(last_row + 1, 3).value = data['tag_no']
                ws.range(last_row + 1, 4).value = data['label']
                if self.line:
                    ws.range(last_row + 1, 5).value = self.line

                if self.service_in and self.service_out:
                    ws.range(last_row + 1, 6).value = self.service_in + ' TO ' + self.service_out
                elif self.service_in:
                    ws.range(last_row + 1, 6).value = 'FROM ' + self.service_in
                elif self.service_out:
                    ws.range(last_row + 1, 6).value = 'TO ' + self.service_out

                if self.equipment:
                    words = self.equipment.split(' ')
                    ws.range(last_row + 1, 5).value = words[0]
                    ws.range(last_row + 1, 6).value = ' '.join(words[1:])


            self.inst_data = []
            self.update_data_display()
        except Exception as e:
            tk.messagebox.showerror(e)
            print(f"error {e}")

    def load_image(self):
        if self.image_list:
            image_path = self.image_list[self.current_image_index]
            self.original_image = Image.open(image_path)
            width, height = self.original_image.size

            # Get the canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Resize the image to fit the canvas while maintaining aspect ratio
            image_width, image_height = self.original_image.size
            aspect_ratio = image_width / image_height

            if aspect_ratio > canvas_width / canvas_height:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)



            self.scaled_image = self.original_image.resize((new_width, new_height))
            self.photo = ImageTk.PhotoImage(self.scaled_image)

            # Clear the previous image from the canvas
            self.canvas.delete("image")
            self.canvas.pack(fill=tk.BOTH, expand=True)

            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

            # Calculate the scaling factor for correct capturing
            scaled_width, scaled_height = self.scaled_image.size
            self.scale_x = width / scaled_width
            self.scale_y = height / scaled_height

            self.create_capture_text()

    def create_capture_text(self):
        # Remove the previous capture text if it exists
        if hasattr(self, 'capture_text') and self.capture_text:
            self.canvas.delete(self.capture_text)

        # Create the capture text
        self.capture_text = self.canvas.create_text(0, 0, text=self.capture, font=("Arial", 8), fill="orange")

        # Bind the mouse motion event to update the capture text position
        self.canvas.bind("<Motion>", self.update_capture_text)

    def go_to_page(self, page_index):
        if self.image_list:
            if page_index < 0:
                self.current_image_index = 0  # Set to the first image
            elif page_index >= len(self.image_list):
                self.current_image_index = len(self.image_list) - 1  # Set to the last image
            else:
                self.current_image_index = page_index

            self.load_image()

            if self.pid_coords:
                # Crop the image using the scaled coordinates
                print(self.pid_coords)
                cropped_image = self.original_image.crop(self.pid_coords)

                cropped_image = pil_to_cv2(cropped_image)

                self.capture_pid(cropped_image)

                self.update_data_display()

    def next_image(self):
        self.go_to_page(self.current_image_index + 1)

    def previous_image(self):
        self.go_to_page(self.current_image_index - 1)


    def start_drawing(self, event):
        self.mouse_pressed = True
        self.start_x = event.x
        self.start_y = event.y

    def draw_box(self, event):
        if self.mouse_pressed:
            if self.current_box and self.current_box not in self.persistent_boxes:
                self.canvas.delete(self.current_box)  # Remove the previous box
                #self.canvas.delete(self.current_text)  # Remove the previous text
            x1, y1 = self.start_x, self.start_y
            x2, y2 = event.x, event.y

            self.current_box = self.canvas.create_rectangle(x1, y1, x2, y2, outline='orange')

            # Calculate the center coordinates of the box
            #center_x = (x1 + x2) // 2
            #center_y = (y1 + y2) // 2

            # Draw the text above the box
            #self.current_text = self.canvas.create_text(x1, y1 - 10, text=self.capture,
            #                                            font=('Helvetica', 8, 'bold'),
            #                                            justify='left', fill='orange')

    def end_drawing(self, event):
        if self.mouse_pressed:
            self.mouse_pressed = False
            if self.current_box:
                # Calculate the coordinates of the cropping box
                x1, y1, x2, y2 = self.canvas.coords(self.current_box)
                # Ensure the coordinates are in the correct order
                x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

                # Apply the scaling factor to the cropping coordinates
                self.cropped_x1 = int(x1 * self.scale_x)
                self.cropped_y1 = int(y1 * self.scale_y)
                self.cropped_x2 = int(x2 * self.scale_x)
                self.cropped_y2 = int(y2 * self.scale_y)

                # Crop the image using the scaled coordinates
                cropped_image = self.original_image.crop((self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2))
                self.cropped_image = pil_to_cv2(cropped_image)
                #here we do perform a different command based on the self.capture
                # Perform the action based on self.capture
                if self.capture in self.capture_actions:
                    self.capture_actions[self.capture](self.cropped_image)
                    # self.capture == 'pid':
                    #    self.pid_coords = (self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2)
                    self.update_data_display()
                else:
                    print(f"Invalid capture action: {self.capture}")

                # If self.capture is 'instrument', keep the box and text on the canvas
                if self.capture == 'instruments':
                    self.persistent_boxes.append(self.current_box)
                    #self.persistent_texts.append(self.current_text)
                else:
                    # Remove the cropping box and the text
                    self.canvas.delete(self.current_box)
                    #self.canvas.delete(self.current_text)
                    self.current_box = None
                    #self.current_text = None

    def clear_boxes(self):


        for box, text in zip(self.persistent_boxes, self.persistent_texts):
            self.canvas.delete(box)  # Remove the previous box
            self.canvas.delete(text)

        # not sure if this is necessary as zip clears stuff
        self.persistent_boxes = []
        self.persistent_texts = []




    def capture_pid(self, cropped_image):
        print('Perform actions for capturing PID')
        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, min_size=10, low_text=0.5, link_threshold=0.2,
                                 text_threshold=0.3, width_ths=6.0, decoder='beamsearch')
        if result[0][1]:
            self.pid = result[0][1]
            self.pid_coords = (self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2)
        else:
            print('no result')

    def capture_instruments(self, cropped_image):
        # Perform actions for capturing instruments
        # cropped_image = pil_to_cv2(cropped_image)
        labels, boxes, scores = model_predict_on_mozaic(cropped_image, self.model_inst)
        inst_prediction_data = zip(labels, boxes, scores)
        inst_data = return_inst_data(inst_prediction_data, cropped_image, 0, self.reader, self.minscore_inst,
                                     self.correct_fn)
        self.inst_data.extend(inst_data)
        print(self.inst_data)


    def capture_line(self, cropped_image):
        # Check if height is greater than width
        height, width = cropped_image.shape[:2]
        if height > width:
            # Rotate the image 90 degrees clockwise
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, min_size=10, low_text=0.5, link_threshold=0.2,
                                 text_threshold=0.3, width_ths=6.0, decoder='beamsearch')
        if result[0][1]:
            self.line = result[0][1]
        else:
            print('no result')
        self.equipment = None

    def capture_equipment(self, cropped_image):
        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, min_size=10, low_text=0.5, link_threshold=0.2,
                                 text_threshold=0.3, width_ths=6.0, decoder='beamsearch')
        if result:
            self.equipment = ' '.join([box[1] for box in result])
            print(self.equipment)
        else:
            self.equipment = ''
            print('no result')
        self.line = None

    def capture_service_in(self, cropped_image):
        # Perform actions for capturing service in
        result = self.reader.readtext(cropped_image, min_size=10, low_text=0.5, link_threshold=0.2,
                                 text_threshold=0.3, width_ths=6.0, decoder='beamsearch')

        if not result:
            self.service_in = ''
            return

        just_text = ' '.join([box[1] for box in result])

        if not self.shift_held:
            self.service_in = just_text
        else:
            self.service_in = merge_common_substring_with_single_chars(self.service_in, just_text)

    def capture_service_out(self, cropped_image):
        # Perform actions for capturing service out
        result = self.reader.readtext(cropped_image, min_size=10, low_text=0.5, link_threshold=0.2,
                                 text_threshold=0.3, width_ths=6.0, decoder='beamsearch')

        if not result:
            self.service_out = ''
            return

        just_text = ' '.join([box[1] for box in result])

        if not self.shift_held:
            self.service_out = just_text
        else:
            self.service_out = merge_common_substring_with_single_chars(self.service_out, just_text)



    def update_data_display(self):
        self.data_text.delete('1.0', tk.END)  # Clear the text box
        self.data_text.insert(tk.END, f"PID: {self.pid}\n")
        self.data_text.insert(tk.END, f"Page: {self.current_image_index + 1} of {len(self.image_list)}\n")
        self.data_text.insert(tk.END, f"Line: {self.line}\n")
        self.data_text.insert(tk.END, f"Service In: {self.service_in}\n")
        self.data_text.insert(tk.END, f"Service Out: {self.service_out}\n")
        self.data_text.insert(tk.END, f"Equipment: {self.equipment}\n")

        self.data_text.insert(tk.END, f"Instrument Data:\n")
        for data in self.inst_data:
            self.data_text.insert(tk.END, f"{data}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
