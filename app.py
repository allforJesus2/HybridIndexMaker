
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
import openpyxl
from easyocr_mosaic import *
from convolutioner import ConvolutionReplacer
import xml.etree.ElementTree as ET
import re

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

        labels = '3WAY, ARROW, BALL, BUTTERFLY, CHECK, CORIOLIS, DCS, DIAPHRAM, GATE, GLOBE, INST, KNIFE, MAGNETIC, ORIFICE, PLUG, SEAL, ULTRASONIC, VBALL'
        self.labels = labels.split(', ')
        self.model_inst_path = "models/arrows65.pth"
        self.model_equipment_path = "models/equipment_services_v2.pth"
        try:
            # load instrument recognition model
            print('loading model')
            self.model_inst = Model.load(self.model_inst_path, self.labels)
            print('model loaded')
        except Exception as e:
            print('Error',e)
            print('load model failed. you will likely have to load the model from command')

        labels = 'tank, pump, service_in, service_out'
        self.labels_equipment = labels.split(', ')
        self.model_equipment_path = "models/equipment_services_v2.pth"
        try:
            # load instrument recognition model
            print('loading equipment/service model')
            self.model_equip = Model.load(self.model_equipment_path, self.labels_equipment)
            print('model loaded')
        except Exception as e:
            print('Error',e)
            print('load model failed. you will likely have to load the model from command')


        self.minscore_inst = 0.74
        self.inst_data = []
        self.active_inst_box_count = 0
        self.show_line = False
        self.write_mode = 'xlwings'

        self.equipment_defined = None

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
        self.command_menu.add_command(label="Live Write Mode", command=lambda: self.set_write_mode('xlwings'))
        self.command_menu.add_command(label="Silent/quick Write Mode", command=lambda: self.set_write_mode('openpyxl'))
        self.command_menu.add_command(label="Save workbook", command=self.save_workbook)
        self.command_menu.add_command(label="Auto Generate Index", command=self.auto_generate_index)
        self.command_menu.add_command(label="Generate type xlsx", command=self.create_tag_type_xlsx)
        self.command_menu.add_command(label="Generate type xlsx ai", command=self.create_tag_type_xlsx_ai)
        self.command_menu.add_command(label="Generate type xlsx ai v2", command=self.create_tag_type_xlsx_ai_v2)
        self.command_menu.add_command(label="Generate Filename PID xlsx", command=self.make_pid_page_xlsx)


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
        self.settings_menu.add_command(label="Set min score for instruments", command=self.set_minscore)

        self.menu_bar.add_cascade(label="Settings", menu=self.settings_menu)

        # Create a Help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="Keybindings", command=self.show_keybindings)
        self.help_menu.add_command(label="Object detection lables", command=self.show_labels)

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


    def set_minscore(self):
        self.minscore_inst = tk.simpledialog.askfloat(prompt="Enter Object Minscore", title="Enter Object Minscore", initialvalue=self.minscore_inst)

    def set_association_radius(self):
        self.association_radius = tk.simpledialog.askfloat(prompt="Enter Object association radius", title="Enter association ( Radius", initialvalue=self.minscore_inst)

    def load_labels(self):

        folder_path = filedialog.askdirectory(title="Select folder containing xmls for labels")
        all_object_names = []

        # Iterate through XML files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)

                # Load XML data
                with open(file_path, 'r') as xml_file:
                    xml_data = xml_file.read()

                # Parse XML data
                rootd = ET.fromstring(xml_data)

                # Extract object names
                object_names = [obj.find('name').text for obj in rootd.findall('.//object')]
                all_object_names.extend(object_names)

        all_object_names_set = set(all_object_names)
        all_object_names = list(all_object_names_set)
        all_object_names.sort()
        print(all_object_names)
        return all_object_names
    def create_tag_type_xlsx_ai(self):
        model_path = filedialog.askopenfilename(title="PTH Model file",filetypes=[('PTH File', '*.pth')])
        labels = self.load_labels()
        model = Model.load(model_path, labels)


        image_folder = filedialog.askdirectory(title='Folder that has PNGs')
        expansion_pixels = tk.simpledialog.askinteger("Expand box", "Enter box expansion pixels:", initialvalue=180)
        confidence = tk.simpledialog.askinteger("Confidence", "Enter convolution confidence threshold:", initialvalue=80)/100

        tag_type_xlsx = openpyxl.Workbook()
        ws = tag_type_xlsx.create_sheet('tagtype')
        # Define the header row
        header = ['TAG', 'TAG_NO', 'LABEL', 'TYPE', 'PAGE', 'PID']
        # Write the header row to the first sheet of the workbook
        ws.append(header)


        #for img in image_folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                file_path = os.path.join(image_folder, filename)
                img = cv2.imread(file_path)
                print(filename)
                if self.pid_coords:
                    self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2 = self.pid_coords
                    cropped_image = img[self.cropped_y1:self.cropped_y2, self.cropped_x1:self.cropped_x2]
                    self.capture_pid(cropped_image)

                labels, boxes, scores = model_predict_on_mozaic(img, model)
                results = zip(labels, boxes, scores)
                #img_with_boxes = self.plot_pic(img, labels, boxes, scores)
                #output_image_path = "result_image.png"
                #cv2.imwrite(output_image_path, img_with_boxes)

                for label, box, score in results:
                    if score > confidence:
                        x1d, y1d, x2d, y2d = box
                        top_left, bottom_right = (int(x1d), int(y1d)), (int(x2d),int(y2d))
                        # Expand the box by 100px (adjustable parameter)

                        x1, y1 = int(max(0, top_left[0] - expansion_pixels)), int(max(0, top_left[1] - expansion_pixels))
                        x2, y2 = int(min(img.shape[1], bottom_right[0] + expansion_pixels)), int(
                            min(img.shape[0], bottom_right[1] + expansion_pixels))

                        # Crop the image
                        cropped_image = img[y1:y2, x1:x2]

                        self.capture_instruments(cropped_image)


                        for data in self.inst_data:
                            data['label'] = label
                            row = [data['tag'], data['tag_no'], data['label'], data['type'], filename, self.pid if self.pid else ""]
                            ws.append(row)

                        self.inst_data = []

        counter = 1
        while True:
            try:
                save_location = os.path.join(image_folder, f"tag_type{counter}.xlsx")
                tag_type_xlsx.save(save_location)
                print(f"File saved as: {save_location}")
                break
            except Exception as e:
                print(f"Error saving file: {e}")
                counter += 1

    def create_tag_type_xlsx_ai_v2(self):
        model_path = filedialog.askopenfilename(title="PTH Model file",filetypes=[('PTH File', '*.pth')])
        labels = self.load_labels()
        model = Model.load(model_path, labels)


        image_folder = filedialog.askdirectory(title='Folder that has PNGs')
        radius = tk.simpledialog.askinteger("Expand Radius", "Enter radius expansion pixels for valves:", initialvalue=180)
        minscore = tk.simpledialog.askinteger("Confidence", "Enter convolution confidence threshold:", initialvalue=80)/100
        key_tag = 'INST'

        tag_type_xlsx = openpyxl.Workbook()
        ws = tag_type_xlsx.create_sheet('tagtype')
        # Define the header row
        header = ['TAG', 'TAG_NO', 'TYPE', 'PAGE', 'PID']
        # Write the header row to the first sheet of the workbook
        ws.append(header)

        # Collect all.png and.jpg files, ignoring case
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg'))]

        # Sort the collected files
        image_files.sort()

        #for img in image_folder
        for filename in image_files:
            file_path = os.path.join(image_folder, filename)
            img = cv2.imread(file_path)
            print(filename)
            if self.pid_coords:
                self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2 = self.pid_coords
                cropped_image = img[self.cropped_y1:self.cropped_y2, self.cropped_x1:self.cropped_x2]
                self.capture_pid(cropped_image)

            labels, boxes, scores = model_predict_on_mozaic(img, model)


            results = (labels, boxes, scores)


            # data = {'tag': tag, 'tag_no': tag_no, 'label': label}
            data = return_inst_data2(results, img, self.reader, minscore, self.instrument_reader_settings,
                                     radius=radius)
            print(data)
            for inst in data:
                row = [inst['tag'], inst['tag_no'], inst['label'], filename, self.pid if self.pid else ""]
                ws.append(row)



        counter = 1
        while True:
            try:
                save_location = os.path.join(image_folder, f"tag_type{counter}.xlsx")
                tag_type_xlsx.save(save_location)
                print(f"File saved as: {save_location}")
                break
            except Exception as e:
                print(f"Error saving file: {e}")
                counter += 1

    def make_pid_page_xlsx(self):
        #for image in folder
        #get pid
        #get page
        #write to xlsx
        pid_page_xlsx = openpyxl.Workbook()
        ws = pid_page_xlsx.create_sheet('pid_page')
        image_folder = filedialog.askdirectory(title='Folder that has PNGs')
        if self.pid_coords:

            for filename in os.listdir(image_folder):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    file_path = os.path.join(image_folder, filename)
                    img = cv2.imread(file_path)

                    self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2 = self.pid_coords
                    cropped_image = img[self.cropped_y1:self.cropped_y2, self.cropped_x1:self.cropped_x2]
                    self.capture_pid(cropped_image)

                    row = [filename, self.pid if self.pid else ""]
                    ws.append(row)

        counter = 1
        while True:
            try:
                save_location = os.path.join(image_folder, f"pid_page_{counter}.xlsx")
                pid_page_xlsx.save(save_location)
                print(f"File saved as: {save_location}")
                break
            except Exception as e:
                print(f"Error saving file: {e}")
                counter += 1

    def create_tag_type_xlsx(self):
        kernel_folder = filedialog.askdirectory(title='Folder that has Kernels')
        image_folder = filedialog.askdirectory(title='Folder that has PNGs')
        scale = tk.simpledialog.askinteger("Scale percent", "Enter a scale percent 1-100:", initialvalue=100)/100
        expansion_pixels = tk.simpledialog.askinteger("Expand box", "Enter box expansion pixels:", initialvalue=180)
        confidence = tk.simpledialog.askinteger("Confidence", "Enter convolution confidence threshold:", initialvalue=80)/100
        rotation = tk.simpledialog.askinteger("Rotation", "Enter rotation directions 1-4:", initialvalue=1)

        CR = ConvolutionReplacer(kernel_folder, scale, rotation)


        tag_type_xlsx = openpyxl.Workbook()
        ws = tag_type_xlsx.create_sheet('tagtype')
        # Define the header row
        header = ['TAG', 'TAG_NO', 'TYPE', 'PAGE', 'PID']
        # Write the header row to the first sheet of the workbook
        ws.append(header)


        #for img in image_folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                file_path = os.path.join(image_folder, filename)
                img = cv2.imread(file_path)

                if self.pid_coords:
                    self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2 = self.pid_coords
                    cropped_image = img[self.cropped_y1:self.cropped_y2, self.cropped_x1:self.cropped_x2]
                    self.capture_pid(cropped_image)

                result_boxes_image, final_detections_rescaled = CR.detect(img, threshold=confidence)
                #final_detections_rescaled = [(top_left, bottom_right, label, rotation, score)...]
                cv2.imwrite('temp.png',result_boxes_image)
                #os.startfile('temp.png')

                for detection in final_detections_rescaled:
                    label = detection[2]

                    top_left, bottom_right = detection[0], detection[1]
                    # Expand the box by 100px (adjustable parameter)

                    x1, y1 = int(max(0, top_left[0] - expansion_pixels)), int(max(0, top_left[1] - expansion_pixels))
                    x2, y2 = int(min(img.shape[1], bottom_right[0] + expansion_pixels)), int(
                        min(img.shape[0], bottom_right[1] + expansion_pixels))

                    # Crop the image
                    cropped_image = img[y1:y2, x1:x2]

                    self.capture_instruments(cropped_image)


                    for data in self.inst_data:
                        data['label'] = label
                        row = [data['tag'], data['tag_no'], data['label'], filename, self.pid if self.pid else ""]
                        ws.append(row)

                    self.inst_data = []

        counter = 1
        while True:
            try:
                save_location = os.path.join(image_folder, f"tag_type{counter}.xlsx")
                tag_type_xlsx.save(save_location)
                print(f"File saved as: {save_location}")
                break
            except Exception as e:
                print(f"Error saving file: {e}")
                counter += 1

    def auto_generate_index(self):
        self.page_setup() #load both models if necessary, create results folder and load self.img
        output_file = openpyxl.Workbook()
        self.output_sheet = output_file.active

        self.process_pid_image()

        self.output_file_path = os.path.join(self.folder_path, 'page_' + str(self.current_image_index) + '.xlsx')
        output_file.save(self.output_file_path)

    def process_pid_image(self):


        self.ocr_results = HardOCR(self.img, self.reader, self.reader_settings)
        ocr_img = plot_ocr_results(self.img, self.ocr_results)
        cv2.imwrite(os.path.join(self.results_folder, 'ocr_img.png'), ocr_img)

        labels, boxes, scores = model_predict_on_mozaic(self.img, self.model_inst)
        inst_img = plot_pic(self.img, labels, boxes, scores)
        cv2.imwrite(os.path.join(self.results_folder, 'inst_img.png'), inst_img)
        #(prediction_data, img, clip, reader, minscore, correct_fn, rs):

        inst_prediction_data = zip(labels, boxes, scores)

        inst_data = return_inst_data2(inst_prediction_data, self.img, self.reader,
                                     self.minscore_inst, self.instrument_reader_settings)

        print(inst_data)
        self.get_equipment_and_services()
        cv2.imwrite(os.path.join(self.results_folder, 'equipment_results_img.png'), self.equipment_results_img)

        self.re_lines =  r'^[0-9]{5}-.*'
        self.re_equipment = r'^[A-Z]{2,3}-[0-9]{4}.?'
        self.erosion = 10
        self.shrink_factor = 4
        self.lower_thresh = 128
        self.line_box_expand = 20
        self.equipment_box_expand = 100
        self.inst_tag_blacklist = 'mw, sp'
        self.include_dcs = 0
        line_img, equipment_img, service_in_img, service_out_img, lines, line_colors, equipments, services_in, services_out =\
            process_images_and_data(
            self.img,
            self.img_no_equipment,
            self.img,
            self.ocr_results,
            self.re_lines,
            self.re_equipment,
            self.erosion,
            self.shrink_factor,
            self.lower_thresh,
            self.line_box_expand,
            self.services,
            self.valid_equipment_boxes,
            self.equipment_box_expand,
            self.include_inside,
            self.shrink_factor
        )

        #cv2.imwrite(os.path.join(self.results_folder, 'line_img.png'), line_img)
        #cv2.imwrite(os.path.join(self.results_folder, 'service_in.png'), service_in_img)
        #cv2.imwrite(os.path.join(self.results_folder, 'service_out.png'), service_out_img)
        #cv2.imwrite(os.path.join(self.results_folder, 'equipment_img.png'), equipment_img)

        print(f'services in {services_in}')
        alldata = []

        for inst in inst_data:
            print(inst)
            tag, tag_no, box, itype = inst['tag'], inst['tag_no'], inst['box'], inst['type']

            blacklist = self.inst_tag_blacklist.upper().split(', ')
            if tag.upper() in blacklist:
                continue

            data = {'page': self.current_image_index, 'pid_id': self.pid, 'box': box, 'tag': tag, 'tag_no': tag_no, 'label': itype}
            print(data)
            other_data = get_row_info(box, line_img, equipment_img, service_in_img, service_out_img, self.shrink_factor, lines,
                                      line_colors, equipments, services_in, services_out)
            data.update(other_data)

            #data.update({'comment': comment, 'alarm': alarm, 'sis': ''})
            alldata.append(data)

        img_sis = process_image(self.img, self.erosion, self.shrink_factor, self.lower_thresh)
        figure_out_if_instrument_has_sis(alldata, img_sis, self.shrink_factor)

        if alldata:
            headers = list(alldata[0].keys())
            self.output_sheet.append(headers)

            for data in alldata:
                write_row(data, self.output_sheet, include_dcs=self.include_dcs)

            expand_columns_to_fit(self.output_sheet)



    def page_setup(self):

        if self.model_equip == None:
            labels = ['tank', 'pump', 'service_in', 'service_out']
            # load instrument recognition model
            self.model_equip = Model.load(self.model_equipment_path, labels)

        if self.model_inst == None:
            # load instrument recognition model
            self.model_inst = Model.load(self.model_inst_path, self.labels)


        #self.page_number = self.sorted_pages[0]
        # create a folder to save the result images to
        self.results_folder = os.path.join(self.folder_path, f'results_page_{str(self.current_image_index)}')
        # print(self.results_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)


        # convert page number to a file
        self.image_filename = os.path.join(self.folder_path, 'page_' + str(self.current_image_index) + '.png')
        image_path = os.path.join(self.folder_path, self.image_filename)
        self.img = cv2.imread(image_path)

    def get_equipment_and_services(self):
        #shouldn't have to do this but the code is inefficient
        self.page_setup()
        #image_path = os.path.join(self.folder_path, self.image_filename)
        #self.img = cv2.imread(image_path)
        self.minscore_service = 0.7
        self.minscore_equipment = 0.7
        self.service_comment_expand = 20
        self.include_inside = True
        self.comment_remove_pattern = None

        # labels, boxes, scores = self.model_equipment.predict(img)
        # plot_prediction_grid(model, [img], figsize=(45,45), score_filter=0)
        if not self.equipment_defined:
            self.equip_labels, self.equip_boxes, self.equip_scores = self.model_equip.predict(self.img)
        self.equipment_defined = False # this is just a flag that say user has not predfined the labels, boxes, and scores

        # these need to have a self because they may be predefined
        self.valid_equipment_boxes = [] # score above minscore
        self.services = []

        self.equipment_results_img = plot_pic(self.img, self.equip_labels, self.equip_boxes, self.equip_scores,
                                              minscore=.5)

        self.img_no_equipment = copy.copy(self.img)
        for label, box, score in zip(self.equip_labels, self.equip_boxes, self.equip_scores):

            if (label == 'tank' or label == 'pump'):
                if score > self.minscore_equipment:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    # white out box
                    print(f"x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
                    pt1 = (x1, y1)
                    pt2 = (x2, y2)
                    self.valid_equipment_boxes.append(box)
                    # check if there is a equipment name close by

                    # draw boxes for viewing results
                    cv2.rectangle(self.img_no_equipment, pt1, pt2, (255, 255, 255), cv2.FILLED)

            if label == 'service_in' or label == 'service_out':
                if score > self.minscore_service:
                    # here we want to create a list of services
                    # service = (text, box, line)
                    # then for service in services
                    # get_comment(ocr_results, box, box_expand)
                    text = get_comment(self.ocr_results, box, self.service_comment_expand, include_inside=self.include_inside,
                                       remove_pattern=self.comment_remove_pattern)
                    print(f'{label}: {text}')
                    # line = ''
                    self.services.append([text, label, box])

    def set_write_mode(self, mode):
        self.write_mode = mode

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

    def show_labels(self):

        tk.messagebox.showinfo("object recognition labels", self.labels)
    def save_attributes(self):
        """Save class attributes to a JSON file"""
        attributes_to_save = [
            'pid_coords',
            'current_image_index',
            'instrument_reader_settings',
            'reader_settings',
            'show_line',
            'model_inst_path',
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
        self.save_attributes()
        print('instrument reader settings: ', rs)

    def set_reader_settings(self, rs):
        self.reader_settings = rs
        self.save_attributes()
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
        width = tk.simpledialog.askinteger("DPI", "Enter the image width:", initialvalue=5000)

        # Open file dialog to select PDF file
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        pdf_file = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF Files", "*.pdf")])

        if pdf_file and width:
            # Call the pdf2png function with the selected PDF file and DPI value
            pdf2png(pdf_file, width)

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

        # Define a custom sorting key function
        def natural_sort_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.image_list = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            # Sort the collected files using the natural sort key function
            self.image_list.sort(key=natural_sort_key)
            self.current_image_index = 0
            self.load_attributes()
            self.go_to_page(self.current_image_index)

    def append_data_to_excel(self):
        try:
            # Example data to append
            file = 'index.xlsx'
            self.workbook_path = os.path.join(self.folder_path, file)

            if self.write_mode == 'xlwings':
                self.append_data_with_xlwings()
            elif self.write_mode == 'openpyxl':
                self.append_data_with_openpyxl()
            else:
                raise ValueError(f"Invalid write mode: {self.write_mode}")

        except Exception as e:
            tk.messagebox.showerror(e)
            print(f"error {e}")

    def append_data_with_xlwings(self):
        # Check if the file exists, if not, create a new workbook
        if not os.path.exists(self.workbook_path):
            self.wb = xw.Book()  # Create a new workbook
            self.wb.save(self.workbook_path)  # Save the workbook to the specified path

        # Open the workbook (it will be opened in the background)
        wb = xw.Book(self.workbook_path)
        # Check if the 'Instrument Index' sheet exists, if not, create it
        if 'Instrument Index' not in wb.sheet_names:
            wb.sheets.add(name='Instrument Index')

            # Define the header row
            header = ['PID', 'TAG', 'TAG_NO', 'LABEL', 'TYPE', 'LINE/EQUIP', 'SERVICE', 'COMMENT']

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
            ws.range(last_row + 1, 5).value = data['type']


            if self.service_in and self.service_out:
                if self.service_in == self.service_out:
                    ws.range(last_row + 1, 7).value = self.service_in + ' RECIRCULATION'
                else:
                    ws.range(last_row + 1, 7).value = self.service_in + ' TO ' + self.service_out
            elif self.service_in:
                ws.range(last_row + 1, 7).value = self.service_in + " OUTLET"
            elif self.service_out:
                ws.range(last_row + 1, 7).value = 'TO ' + self.service_out

            if self.line:
                ws.range(last_row + 1, 6).value = self.line
            elif self.equipment:
                words = self.equipment.split(' ')
                ws.range(last_row + 1, 6).value = words[0]
                ws.range(last_row + 1, 7).value = ' '.join(words[1:])

            if self.comment:
                ws.range(last_row + 1, 8).value = self.comment

        #self.persistent_boxes.append(self.current_box)

        self.turn_boxes_blue()

    def append_data_with_openpyxl(self):
        print('starting fn')
        # Check if the file exists, if not, create a new workbook
        if not os.path.exists(self.workbook_path):
            self.wb = openpyxl.Workbook()
            #self.wb.save(workbook_path)
        else:
            if not self.wb:
                self.wb = openpyxl.load_workbook(self.workbook_path)
                print('wb loaded')

        # Check if the 'Instrument Index' sheet exists, if not, create it
        if 'Instrument Index' not in self.wb.sheetnames:
            self.ws = self.wb.create_sheet('Instrument Index')
            # Define the header row
            header = ['PID', 'TAG', 'TAG_NO', 'LABEL', 'LINE/EQUIP', 'SERVICE', 'COMMENT']
            # Write the header row to the first sheet of the workbook
            self.ws.append(header)
        else:
            self.ws = self.wb['Instrument Index']

        print('starting data append')
        # Append data to the worksheet
        for data in self.inst_data:
            row = [self.pid, data['tag'], data['tag_no'], data['label']]

            if self.service_in and self.service_out:
                row.append(self.service_in + ' TO ' + self.service_out)
            elif self.service_in:
                row.append('FROM ' + self.service_in)
            elif self.service_out:
                row.append('TO ' + self.service_out)
            else:
                row.append('')

            if self.line:
                row.append(self.line)
            elif self.equipment:
                words = self.equipment.split(' ')
                row.append(words[0])
                if len(words) > 1:
                    row.append(' '.join(words[1:]))
                else:
                    row.append('')
            else:
                row.extend(['', ''])

            if self.comment:
                row.append(self.comment)
            else:
                row.append('')

            self.ws.append(row)


        self.turn_boxes_blue()

    def save_workbook(self):
        # Save the workbook
        if self.write_mode == 'openpyxl':
            self.wb.save(self.workbook_path)
            print('wb saved')
        else:
            print('not in openpyxl write mode. cant save')

    def turn_boxes_blue(self):
        # Slice the list to get the last 4 items
        active_boxes = self.persistent_boxes[-self.active_inst_box_count:]

        # Iterate over the sliced list and change the outline color of each item
        for box_id in active_boxes:
            self.canvas.itemconfig(box_id, outline='#87CEEB')

        self.active_inst_box_count = 0
        self.inst_data = []
        self.update_data_display()

    '''
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
    '''
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
        if result:
            self.pid = ' '.join([box[1] for box in result])
            self.pid_coords = (self.cropped_x1, self.cropped_y1, self.cropped_x2, self.cropped_y2)
        else:
            self.pid = ''
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
            inst_data = return_inst_data2(inst_prediction_data, cropped_image, self.reader, self.minscore_inst,
                                          self.instrument_reader_settings)

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
            self.data_text.insert(tk.END, f"{data['tag']}\t{data['tag_no']}\t{data['type']}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
