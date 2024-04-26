
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
from reader_settings import SetReaderSettings
import json

class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        root.wm_state('zoomed')
        #reader settings
        # Initialize General OCR setting

        # Instrument reader settings
        self.instrument_reader_settings = {
            "low_text": 0.3,
            "min_size": 10,
            "ycenter_ths": 0.5,
            "height_ths": 0.5,
            "width_ths": 6.0,
            "add_margin": -0.1,
            "link_threshold": 0.2,
            "text_threshold": 0.3,
            "mag_ratio": 3.0,
            "allowlist": '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()',
            "decoder": 'beamsearch'
        }

        # general reader settings
        self.reader_settings = {
            "low_text": 0.3,
            "min_size": 10,
            "ycenter_ths": 0.5,
            "height_ths": 0.5,
            "width_ths": 6.0,
            "add_margin": 0.1,
            "link_threshold": 0.2,
            "text_threshold": 0.3,
            "mag_ratio": 1.0,
            "allowlist": '',
            "decoder": 'beamsearch'
        }

        self.line = None
        self.service_in = None
        self.service_out = None
        self.equipment = None
        self.inst_data = None
        self.pid = None
        self.comment = None
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
        self.current_text = None
        self.wb = None
        self.sheet = None
        self.capture = 'pid'
        self.capture_actions = {
            'pid': self.capture_pid,
            'instruments': self.capture_instruments,
            'line': self.capture_line,
            'equipment': self.capture_equipment,
            'service_in': self.capture_service_in,
            'service_out': self.capture_service_out,
            'comment': self.capture_comment
        }

        print('loading reader')
        self.reader = easyocr.Reader(['en'])
        print('reader loaded')

        self.labels = ['inst', 'dcs', 'ball', 'globe', 'diaphragm', 'knife', 'vball', 'plug', 'butterfly', 'gate']

        self.model_inst_path = r"saved_model_vid-v3.18_GEVO.pth"
        try:
            # load instrument recognition model
            print('loading model')
            self.model_inst = Model.load(self.model_inst_path, self.labels)
            print('model loaded')
        except Exception as e:
            print('Error',e)
            print('load model failed. you will likely have to load the model from command')

        self.minscore_inst = 0.7
        self.inst_data = []
        self.active_inst_box_count = 0
        self.show_line = False

        # Create a menu bar
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Folder", command=self.open_folder)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # Create a commands menu
        self.command_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.command_menu.add_command(label="Create images from PDF", command=self.open_pdf2png)
        self.command_menu.add_command(label="Load Object detection model", command=self.load_model)
        self.command_menu.add_command(label="Merge pdfs", command=self.merge_pdfs)
        #self.command_menu.add_command(label="Set comment", command=self.set_comment)
        self.command_menu.add_command(label="Swap services", command=self.swap_services)
        #self.command_menu.add_command(label="Clear boxes", command=self.clear_boxes)
        self.command_menu.add_command(label="Clear instrument group", command=self.clear_instrument_group)
        self.command_menu.add_command(label="Load a Tag Correction Function", command=self.load_correct_fn)
        self.command_menu.add_command(label="Append Data to Index", command=self.append_data_to_excel)
        self.command_menu.add_command(label="Toggle show line text", command=self.toggle_show_line_text)

        self.menu_bar.add_cascade(label="Commands", menu=self.command_menu)

        # Create a capture menu
        self.capture_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.capture_menu.add_command(label="Capture PID", command=lambda: self.set_capture('pid'))
        self.capture_menu.add_command(label="Capture Instrument Group", command=lambda: self.set_capture('instruments'))
        self.capture_menu.add_command(label="Capture Line", command=lambda: self.set_capture('line'))
        self.capture_menu.add_command(label="Capture Equipment", command=lambda: self.set_capture('equipment'))
        self.capture_menu.add_command(label="Capture Service In", command=lambda: self.set_capture('service_in'))
        self.capture_menu.add_command(label="Capture Service Out", command=lambda: self.set_capture('service_out'))
        self.capture_menu.add_command(label="Capture comment", command=lambda: self.set_capture('comment'))
        self.menu_bar.add_cascade(label="Capture", menu=self.capture_menu)

        # Create page menu
        self.page_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.page_menu.add_command(label="Next", command=self.next_image)
        self.page_menu.add_command(label="Previous", command=self.previous_image)
        self.page_menu.add_command(label="Go to Page", command=self.open_go_to_page)
        self.menu_bar.add_cascade(label="Page", menu=self.page_menu)

        # settings menu
        self.settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.settings_menu.add_command(label="Open instrument reader settings", command=self.open_instrument_reader_settings)
        self.settings_menu.add_command(label="Open general reader settings", command=self.open_general_reader_settings)
        self.settings_menu.add_command(label="Save Settings", command=self.save_attributes)
        self.menu_bar.add_cascade(label="Settings", menu=self.settings_menu)

        # Create a Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="Keybindings", command=self.show_keybindings)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        self.root.config(menu=self.menu_bar)
        self.capture='pid'
        self.pid_coords = None

        # Create a canvas to display the image
        self.canvas = tk.Canvas(self.root, width=1200, height=900)
        self.canvas.pack()
        # Create a separate window for displaying captured data
        self.data_window = tk.Toplevel(self.root)
        self.data_window.title("Captured Data")

        # Get the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate the x-coordinate and height of the data_window
        x = screen_width - 260  # assuming a width of 400 for the data_window
        height = screen_height-100

        # Set the geometry of the data_window
        self.data_window.geometry(f'250x{height}+{x}+32')


        # Create a Text widget to display the captured data
        self.data_text = tk.Text(self.data_window, wrap=tk.WORD, font=("Courier", 14))
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
        self.root.bind('v', lambda event: self.vote())
        self.root.bind('s', lambda event: self.swap_services())
        self.root.bind('g', lambda event: self.set_capture('comment'))

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
        self.root.bind('V', lambda event: self.vote())
        self.root.bind('S', lambda event: self.swap_services())
        self.root.bind('G', lambda event: self.set_capture('comment'))

        # shift key binding
        self.shift_held = False
        self.root.bind('<KeyPress-Shift_L>', self.shift_pressed)
        self.root.bind('<KeyRelease-Shift_L>', self.shift_released)

        # Bind mouse events for cropping
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_box)
        self.canvas.bind('<ButtonRelease-1>', self.end_drawing)
    def toggle_show_line_text(self):
        self.show_line = not self.show_line

    def show_keybindings(self):
        keybindings = """
        n/N: Next image
        b/B: Previous image
        p/P: Capture PID
        a/A: Capture Instrument Group
        f/F: Capture Line
        e/E: Capture Equipment
        z/Z: Capture Service In
        x/X: Capture Service Out
        w/W: Write Data to Excel
        c/C: Clear Instrument Group
        v/V: Vote to normalize tag numbers
        s/S: Swap Services
        g/G: Set Comment
        """
        tk.messagebox.showinfo("Keybindings", keybindings)
    def save_attributes(self):
        """Save class attributes to a JSON file"""
        attributes_to_save = [
            'pid_coords',
            'current_image_index',
            'instrument_reader_settings',
            'model_inst_path'
            # Add any other attribute names you want to save here
        ]

        attributes = {attr_name: getattr(self, attr_name) for attr_name in attributes_to_save}

        attributes_file = os.path.join(self.folder_path, 'attributes.json')
        with open(attributes_file, 'w') as file:
            json.dump(attributes, file)

    def load_attributes(self):
        """Load class attributes from a JSON file"""
        attributes_file = os.path.join(self.folder_path, 'attributes.json')
        if os.path.exists(attributes_file):
            try:
                with open(attributes_file, 'r') as file:
                    attributes = json.load(file)
                    # Automatically load attributes if they exist in the JSON file
                    for key, value in attributes.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
            except json.JSONDecodeError:
                print("Error: Invalid JSON file format.")
        else:
            print("Attribute file not found. Using default values.")

    def set_instrument_reader_settings(self, rs):
        self.instrument_reader_settings = rs
        print('instrument reader settings: ', rs)

    def set_reader_settings(self, rs):
        self.reader_settings = rs
        print('reader settings: ', rs)

    def open_instrument_reader_settings(self):
        # Create a new window for ObjectDetectionApp
        img_path = 'instrument_capture.png'
        if os.path.exists(img_path):
            reader_settings_root = tk.Toplevel()
            SetReaderSettings(reader_settings_root, img_path, self.reader,
                              reader_settings=self.instrument_reader_settings,
                              callback=self.set_instrument_reader_settings)
        else:
            print('first capture an instrument')

    def open_general_reader_settings(self):
        # Create a new window for ObjectDetectionApp
        img_path = 'ocr_capture.png'
        if os.path.exists(img_path):
            reader_settings_root = tk.Toplevel()
            SetReaderSettings(reader_settings_root, img_path, self.reader,
                              reader_settings=self.reader_settings,
                              callback=self.set_reader_settings)
        else:
            print('first capture an instrument')

    def swap_services(self):
        self.service_in, self.service_out = self.service_out, self.service_in
        self.update_data_display()

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
            for box in self.persistent_boxes[-self.active_inst_box_count:]:
                self.canvas.delete(box)  # Remove the previous box
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
            self.load_attributes()
            self.go_to_page(self.current_image_index)


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
                header = ['PID', 'TAG', 'TAG_NO', 'LABEL', 'LINE/EQUIP', 'SERVICE', 'COMMENT']

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

                if self.service_in and self.service_out:
                    ws.range(last_row + 1, 6).value = self.service_in + ' TO ' + self.service_out
                elif self.service_in:
                    ws.range(last_row + 1, 6).value = 'FROM ' + self.service_in
                elif self.service_out:
                    ws.range(last_row + 1, 6).value = 'TO ' + self.service_out

                if self.line:
                    ws.range(last_row + 1, 5).value = self.line
                elif self.equipment:
                    words = self.equipment.split(' ')
                    ws.range(last_row + 1, 5).value = words[0]
                    ws.range(last_row + 1, 6).value = ' '.join(words[1:])

                if self.comment:
                    ws.range(last_row + 1, 7).value = self.comment

            #self.persistent_boxes.append(self.current_box)

            # Slice the list to get the last 4 items
            active_boxes = self.persistent_boxes[-self.active_inst_box_count:]

            # Iterate over the sliced list and change the outline color of each item
            for box_id in active_boxes:
                self.canvas.itemconfig(box_id, outline='#87CEEB')

            self.active_inst_box_count = 0
            #self.comment = ''

            #change last box color

            self.inst_data = []
            self.update_data_display()

        except Exception as e:
            tk.messagebox.showerror(e)
            print(f"error {e}")

    def set_comment(self):
        self.comment = tk.simpledialog.askstring("Input", "Please enter a Comment:")

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
                #this is weird but we need to set the self.cropped coords so that we dont let the last used coord overwrite pid
                self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2 = self.pid_coords
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
                if self.current_text:
                    self.canvas.delete(self.current_text)  # Remove the previous text
                # we do it here so we can view the last line capture
            x1, y1 = self.start_x, self.start_y
            x2, y2 = event.x, event.y

            self.current_box = self.canvas.create_rectangle(x1, y1, x2, y2, outline='orange')


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
                filename = 'ocr_capture.png'
                cv2.imwrite(filename, self.cropped_image)

                # Perform the action based on self.capture
                if self.capture in self.capture_actions:
                    self.capture_actions[self.capture](self.cropped_image)
                    self.update_data_display()
                else:
                    print(f"Invalid capture action: {self.capture}")

            # We do this so that we dont click and re-extend the instrument group
            if self.current_box not in self.persistent_boxes:
                self.canvas.delete(self.current_box)
            self.current_box = None

    def clear_boxes(self):


        for box in self.persistent_boxes:
            self.canvas.delete(box)  # Remove the previous box
            #self.canvas.delete(text)

        # not sure if this is necessary as zip clears stuff
        self.persistent_boxes = []
        #self.persistent_texts = []

    def capture_pid(self, cropped_image):
        print('Perform actions for capturing PID')
        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, **self.reader_settings)
        if result[0][1]:
            self.pid = result[0][1]
            self.pid_coords = (self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2)
        else:
            print('no result')

    def capture_comment(self, cropped_image):
        print('Perform actions for capturing a comment')
        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, **self.reader_settings)
        if result:
            self.comment =  ' '.join([box[1] for box in result])
        else:
            self.comment =''

        #self.current_text = self.canvas.create_text(self.start_x, self.start_y, text=self.comment, fill="blue", font=("Courier", 12))


    def capture_instruments(self, cropped_image):
        # Perform actions for capturing instruments
        # cropped_image = pil_to_cv2(cropped_image)
        labels, boxes, scores = model_predict_on_mozaic(cropped_image, self.model_inst)
        if labels:
            #self.last_inst_box = self.current_box
            self.persistent_boxes.append(self.current_box)
            self.active_inst_box_count += 1
            # self.persistent_texts.append(self.current_text)
            inst_prediction_data = zip(labels, boxes, scores)
            inst_data = return_inst_data(inst_prediction_data, cropped_image, 0, self.reader, self.minscore_inst,
                                         self.correct_fn, self.instrument_reader_settings)

            self.inst_data.extend(inst_data)
            print(self.inst_data)

    def vote(self):
        # Create a dictionary to store tag_no counts
        tag_counts = {}

        # Count the occurrences of each tag_no
        for data in self.inst_data:
            tag_no = data['tag_no']
            tag_counts[tag_no] = tag_counts.get(tag_no, 0) + 1

        # Find the most frequent tag_no
        most_frequent_tag_no = max(tag_counts, key=tag_counts.get)

        # Assign the most frequent tag_no to each entry
        for data in self.inst_data:
            data['tag_no'] = most_frequent_tag_no

        self.update_data_display()

    def capture_line(self, cropped_image):
        # Check if height is greater than width
        height, width = cropped_image.shape[:2]
        if height > width:
            # Rotate the image 90 degrees clockwise
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, **self.reader_settings)
        if result:
            #self.line = result[0][1]
            self.line = ' '.join([box[1] for box in result])
        else:
            print('no result')
        self.equipment = None
        if self.show_line:
            self.current_text = self.canvas.create_text(self.start_x, self.start_y, text=self.line, fill="blue", font=("Courier", 12))


    def capture_equipment(self, cropped_image):
        # Perform actions for capturing line
        result = self.reader.readtext(cropped_image, **self.reader_settings)
        if result:
            self.equipment = ' '.join([box[1] for box in result])
            print(self.equipment)
        else:
            self.equipment = ''
            print('no result')
        self.line = None
        #self.current_text = self.canvas.create_text(self.start_x, self.start_y, text=self.equipment, fill="blue", font=("Courier", 12))


    def capture_service_in(self, cropped_image):
        # Perform actions for capturing service in
        result = self.reader.readtext(cropped_image, **self.reader_settings)

        if not result:
            self.service_in = ''
            return

        just_text = ' '.join([box[1] for box in result])

        if not self.shift_held:
            self.service_in = just_text
        else:
            self.service_in = merge_common_substrings(self.service_in, just_text)

        self.equipment = None
        #self.current_text = self.canvas.create_text(self.start_x, self.start_y, text=self.service_in, fill="blue", font=("Courier", 12))


    def capture_service_out(self, cropped_image):
        # Perform actions for capturing service out
        result = self.reader.readtext(cropped_image, **self.reader_settings)

        if not result:
            self.service_out = ''
            return

        just_text = ' '.join([box[1] for box in result])

        if not self.shift_held:
            self.service_out = just_text
        else:
            self.service_out = merge_common_substrings(self.service_out, just_text)

        self.equipment = None
        #self.current_text = self.canvas.create_text(self.start_x, self.start_y, text=self.service_out, fill="blue", font=("Courier", 12))



    def update_data_display(self):

        self.data_text.delete('1.0', tk.END)  # Clear the text box

        self.data_text.insert(tk.END, f"PID: {self.pid}\n")
        self.data_text.insert(tk.END, f"Page: {self.current_image_index + 1} of {len(self.image_list)}\n")
        self.data_text.insert(tk.END, f"Line: {self.line}\n\n")
        self.data_text.insert(tk.END, f"Service In: {self.service_in}\n\n")
        self.data_text.insert(tk.END, f"Service Out: {self.service_out}\n\n")
        self.data_text.insert(tk.END, f"Equipment:{self.equipment}\n\n")
        self.data_text.insert(tk.END, f"Comment: {self.comment}\n\n")
        self.data_text.insert(tk.END, f"Instrument Data:\n\n")
        for data in self.inst_data:
            self.data_text.insert(tk.END, f"{data['label']}\t{data['tag']}\t{data['tag_no']}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
