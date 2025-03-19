import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tkinter import filedialog, messagebox
from detecto import core
import copy
from model_predict_mosaic import *
from .generate_random_training_data import *
import yaml
from ultralytics import YOLO

import tkinter as tk
from tkinter import ttk
import sys
import io

import os
import shutil
import xml.etree.ElementTree as ET
import random
import tempfile
try:
    from detecto_gui.splat_images_gui import SplatImagesGUI
except:
    from splat_images_gui import SplatImagesGUI


# Standalone Functions

# region Utilities

def make_yolo_annotation_from_xml(xml_file, class_list, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_list:
            continue

        class_id = class_list.index(class_name)

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_annotations.append(yolo_annotation)

    # Write annotations to file
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    with open(output_file, 'w') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')

    return output_file


def create_yolov8_dataset(source_dir, output_dir, class_list, train_ratio=0.8):
    print('Make sure labels are loaded')

    # Create directory structure
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

    # Get all XML files
    xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]

    # Shuffle and split the data
    random.shuffle(xml_files)
    split_index = int(len(xml_files) * train_ratio)
    train_files = xml_files[:split_index]
    val_files = xml_files[split_index:]

    # Process files
    for split, files in [('train', train_files), ('val', val_files)]:
        for xml_file in files:
            xml_path = os.path.join(source_dir, xml_file)
            img_file = xml_file.rsplit('.', 1)[0] + '.jpg'
            img_path = os.path.join(source_dir, img_file)

            # Check if image exists (jpg or png)
            if not os.path.exists(img_path):
                img_file = xml_file.rsplit('.', 1)[0] + '.png'
                img_path = os.path.join(source_dir, img_file)
                if not os.path.exists(img_path):
                    print(f"Skipping {xml_file} - no corresponding image found")
                    continue

            # Copy image
            dest_img_path = os.path.join(output_dir, split, 'images', img_file)
            shutil.copy(img_path, dest_img_path)

            # Create YOLO format label
            labels_dir = os.path.join(output_dir, split, 'labels')
            label_file = make_yolo_annotation_from_xml(xml_path, class_list, labels_dir)

            if label_file:
                print(f"Processed: {img_file} -> {os.path.basename(label_file)}")
            else:
                print(f"Warning: No annotations created for {img_file}")

    # Create data.yaml file
    data_yaml = {
        'train': './train',
        'val': './val',
        'nc': len(class_list),
        'names': class_list
    }

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("YOLOv8 dataset created successfully")
    print(f"data.yaml file created at {os.path.join(output_dir, 'data.yaml')}")

    print("YOLOv8 dataset created successfully")


# endregion


# Create a stdout redirector class
class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        # Only update when we get a newline
        if '\n' in self.buffer:
            self.text_widget.insert(tk.END, self.buffer)
            self.text_widget.see(tk.END)  # Auto-scroll to the end
            self.buffer = ""
        
    def flush(self):
        if self.buffer:
            self.text_widget.insert(tk.END, self.buffer)
            self.text_widget.see(tk.END)
            self.buffer = ""


class ObjectDetectionApp:

    # region Core

    def __init__(self, root, model_path = None, labels=[]):
        self.root = root
        self.root.title("Object Detection App")
        self.labels = labels # Initialize labels as an empty list
        self.model = None
        self.model_path = model_path
        self.model_yolo = None
        self.use_yolo = False


        # Create a notebook (tabbed interface)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        model_tab = ttk.Frame(notebook)
        data_tab = ttk.Frame(notebook)
        annotations_tab = ttk.Frame(notebook)
        console_tab = ttk.Frame(notebook)  # New console tab

        notebook.add(model_tab, text="Model")
        notebook.add(data_tab, text="Data")
        notebook.add(annotations_tab, text="Annotations")
        notebook.add(console_tab, text="Console")  # Add console tab

        # Add console output area
        console_frame = ttk.Frame(console_tab)
        console_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add scrollbars
        scrollbar_y = ttk.Scrollbar(console_frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(console_frame, orient='horizontal')
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add the console text widget
        self.console_text = tk.Text(console_frame, wrap=tk.NONE, 
                                    yscrollcommand=scrollbar_y.set,
                                    xscrollcommand=scrollbar_x.set,
                                    bg='black', fg='white', font=('Consolas', 10))
        self.console_text.pack(fill="both", expand=True)
        
        # Configure scrollbars
        scrollbar_y.config(command=self.console_text.yview)
        scrollbar_x.config(command=self.console_text.xview)
        
        # Add console control buttons
        console_buttons_frame = ttk.Frame(console_tab)
        console_buttons_frame.pack(fill="x", padx=5, pady=5)
        
        clear_console_button = ttk.Button(console_buttons_frame, text="Clear Console", 
                                         command=self.clear_console)
        clear_console_button.pack(side=tk.LEFT, padx=5)
        
        save_console_button = ttk.Button(console_buttons_frame, text="Save Console Output", 
                                        command=self.save_console_output)
        save_console_button.pack(side=tk.LEFT, padx=5)
        
        # Redirect stdout to the console text widget
        self.stdout_redirector = TextRedirector(self.console_text)
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout_redirector

        # Add model status display
        self.model_status_frame = ttk.LabelFrame(model_tab, text="Current Model Status")
        self.model_status_frame.pack(pady=5, padx=5, fill="x")

        # Model type indicator
        self.model_type_label = ttk.Label(self.model_status_frame, text="Model type: None")
        self.model_type_label.pack(pady=5)

        # Model path
        self.model_path_label = ttk.Label(self.model_status_frame, text="Model path: None")
        self.model_path_label.pack(pady=5)

        # Add these as class attributes
        self.model_path = None
        self.model_name = None

        # Model tab buttons
        self.train_button = tk.Button(model_tab, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.load_model_button = tk.Button(model_tab, text="Load Pretrained Model", command=self.load_pretrained_model)
        self.load_model_button.pack(pady=10)

        self.save_button = tk.Button(model_tab, text="Save Model", command=self.save_model)
        self.save_button.pack(pady=10)

        self.clear_button = tk.Button(model_tab, text="Clear Model", command=self.clear_model)
        self.clear_button.pack(pady=10)

        # Model type selection
        model_type_frame = ttk.LabelFrame(model_tab, text="Model Type")
        model_type_frame.pack(pady=10)

        self.model_type = tk.StringVar(value="detecto")  # Default to Detecto
        ttk.Radiobutton(model_type_frame, text="Use Detecto",
                        variable=self.model_type,
                        value="detecto",
                        command=self.toggle_model_type).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(model_type_frame, text="Use YOLO",
                        variable=self.model_type,
                        value="yolo",
                        command=self.toggle_model_type).pack(side=tk.LEFT, padx=10)

        # Data tab
        self.patch_button = tk.Button(data_tab, text="Generate Patches", command=self.make_patches)
        self.patch_button.pack(pady=10)

        self.splat_button = tk.Button(data_tab, text="Generate splattered images from kernels",
                                      command=self.splat_images)
        self.splat_button.pack(pady=10)

        self.noise_button = tk.Button(data_tab, text="Add Noise", command=self.add_noise)
        self.noise_button.pack(pady=10)

        # Annotations tab
        self.create_xml_button = tk.Button(annotations_tab, text="Create annotation xml files via model",
                                           command=self.create_annotation_files)
        self.create_xml_button.pack(pady=10)

        self.create_xml_button = tk.Button(annotations_tab, text="Convert xml annotation to yolo",
                                           command=self.convert_xml_to_yolo)
        self.create_xml_button.pack(pady=10)

        self.filter_xmls_button = tk.Button(annotations_tab, text="Filter objects from xml",
                                            command=self.filter_xml_objects)
        self.filter_xmls_button.pack(pady=10)

        self.filterout_xmls_button = tk.Button(annotations_tab, text="Remove objects from xml",
                                               command=self.filterout_xml_objects)
        self.filterout_xmls_button.pack(pady=10)

        self.rename_xmls_button = tk.Button(annotations_tab, text="Rename objects in xml",
                                            command=self.rename_xml_objects)
        self.rename_xmls_button.pack(pady=10)

        self.merge_xmls_button = tk.Button(annotations_tab, text="Merge objects in xml", command=self.merge_xml_objects)
        self.merge_xmls_button.pack(pady=10)

        # Hyperparameters
        hyperparameters_frame = ttk.LabelFrame(model_tab, text="Hyperparameters")
        hyperparameters_frame.pack(pady=10)

        epochs_label = tk.Label(hyperparameters_frame, text="Epochs:")
        epochs_label.pack(side=tk.LEFT, padx=5)
        self.epochs_entry = tk.Entry(hyperparameters_frame)
        self.epochs_entry.insert(0, "5")  # Default value for epochs
        self.epochs_entry.pack(side=tk.LEFT, padx=5)

        learning_rate_label = tk.Label(hyperparameters_frame, text="Learning Rate:")
        learning_rate_label.pack(side=tk.LEFT, padx=5)
        self.learning_rate_entry = tk.Entry(hyperparameters_frame)
        self.learning_rate_entry.insert(0, "0.0005")  # Default value for learning rate
        self.learning_rate_entry.pack(side=tk.LEFT, padx=5)

        gamma_label = tk.Label(hyperparameters_frame, text="LR Decay:")
        gamma_label.pack(side=tk.LEFT, padx=5)
        self.gamma_entry = tk.Entry(hyperparameters_frame)
        self.gamma_entry.insert(0, "0.7")  # Default value for learning rate
        self.gamma_entry.pack(side=tk.LEFT, padx=5)

        # region Test model section

        # New section for Test Model
        test_model_frame = ttk.LabelFrame(model_tab, text="Test Model")
        test_model_frame.pack(pady=10, fill="x", padx=10)

        # Minscore
        minscore_label = tk.Label(test_model_frame, text="Min Score (%):")
        minscore_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.minscore_entry = tk.Entry(test_model_frame)
        self.minscore_entry.insert(0, "50")
        self.minscore_entry.grid(row=0, column=1, padx=5, pady=5)

        # Patch size
        patch_size_label = tk.Label(test_model_frame, text="Patch Size:")
        patch_size_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.patch_size_entry = tk.Entry(test_model_frame)
        self.patch_size_entry.insert(0, "1300")
        self.patch_size_entry.grid(row=1, column=1, padx=5, pady=5)

        # Stride
        stride_label = tk.Label(test_model_frame, text="Stride:")
        stride_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.stride_entry = tk.Entry(test_model_frame)
        self.stride_entry.insert(0, "1200")
        self.stride_entry.grid(row=2, column=1, padx=5, pady=5)

        # NMS Threshold
        nms_threshold_label = tk.Label(test_model_frame, text="NMS Threshold:")
        nms_threshold_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.nms_threshold_entry = tk.Entry(test_model_frame)
        self.nms_threshold_entry.insert(0, "0.5")
        self.nms_threshold_entry.grid(row=3, column=1, padx=5, pady=5)

        # Test Model Button
        self.test_button = tk.Button(test_model_frame, text="Test Model on Image", command=self.test_model_on_image)
        self.test_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.test_button = tk.Button(test_model_frame, text="Test Model on Random Images",
                                     command=self.test_model_on_random_images)
        self.test_button.grid(row=4, column=2, columnspan=2, pady=10)

        # Labels
        labels_frame = ttk.LabelFrame(data_tab, text="Labels")
        labels_frame.pack(pady=10)

        self.labels_entry = tk.Text(labels_frame, height=3, wrap=tk.WORD)  # Set height and wrap properties
        self.labels_entry.pack(side=tk.LEFT, padx=5)

        labels_buttons_frame = ttk.Frame(labels_frame)
        labels_buttons_frame.pack(side=tk.LEFT, padx=5)

        self.set_labels_button = tk.Button(labels_buttons_frame, text="Set Labels", command=self.get_labels_from_user)
        self.set_labels_button.pack(pady=5)

        self.load_labels_button = tk.Button(labels_buttons_frame, text="Load Labels", command=self.load_labels)
        self.load_labels_button.pack(pady=5)
        
        # Load the model if a model path is provided
        if model_path:
            self.load_pretrained_model(model_path)


    # endregion

    # region Model Operations

    def toggle_model_type(self):
        self.use_yolo = (self.model_type.get() == "yolo")
        self.update_model_status()
        print('use yolo:', self.use_yolo)

    def train_model(self):
        # Get the number of epochs and learning rate from the entry fields
        num_epochs = int(self.epochs_entry.get())
        learning_rate = float(self.learning_rate_entry.get())
        gamma = float(self.gamma_entry.get())

        # Create and train the model
        if self.use_yolo:
            if self.model_yolo is None:
                # open drop down asking for pretrained or blank with different models
                yolo_names = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                              "yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml", "yolov8x.yaml"]
                yolo_window = tk.Toplevel(self.root)
                yolo_window.title("YOLO Selection")

                yolo_var = tk.StringVar()
                yolo_var.set(yolo_names[0])  # default value

                yolo_combobox = ttk.Combobox(yolo_window, textvariable=yolo_var)
                yolo_combobox['values'] = yolo_names
                yolo_combobox.pack()

                def ok_button_clicked():
                    selected_yolo = yolo_var.get()
                    yolo_window.destroy()
                    self.model_yolo = YOLO(selected_yolo)

                ok_button = tk.Button(yolo_window, text="OK", command=ok_button_clicked)
                ok_button.pack()
                yolo_window.wait_window(yolo_window)  # wait for the window to be closed

            yaml_path = filedialog.askopenfilename(title='Select the data.yaml file',
                                                   filetypes=[("YAML files", "*.yaml")])
            
            # Check the yaml file for labels that don't match our current labels
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check if any labels in the data.yaml file are not in our current labels
            yaml_labels = data['names']
            unknown_labels = [label for label in yaml_labels if label not in self.labels]
            
            if unknown_labels:
                messagebox.showerror("Unknown Labels", 
                                     f"The following labels in the training data are not in the current model's labels: {', '.join(unknown_labels)}\n\n"
                                     f"Please update your labels before training.")
                return
            
            imgsz = tk.simpledialog.askinteger("image size",
                                               "Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.",
                                               initialvalue=1300)
            self.model_yolo.train(data=yaml_path, epochs=num_epochs, imgsz=imgsz, lr0=learning_rate)

            self.labels = data['names']
            self.labels_entry.delete('1.0', 'end')  # Clear existing content
            self.labels_entry.insert('1.0', ', '.join(self.labels))

        else:
            if self.model is None:
                self.model = core.Model(self.labels)
                # Ask for folder locations
            images_path = filedialog.askdirectory(title="Select training images folder")

            # Check if Validation folder exists in images_path
            validation_path = os.path.join(images_path, 'Validation')

            # If Validation folder doesn't exist, ask for separate validation folder
            if not os.path.exists(validation_path):
                val_images_path = filedialog.askdirectory(title="Select validation images folder")
            else:
                val_images_path = validation_path

            # Check for unknown labels in the training data
            unknown_labels = self.get_unknown_labels_in_folder(images_path)
            if unknown_labels:
                messagebox.showerror("Unknown Labels", 
                                     f"The following labels in the training data are not in the current model's labels: {', '.join(unknown_labels)}\n\n"
                                     f"Please update your labels before training.")
                return
            
            # Also check validation data for unknown labels
            if val_images_path != validation_path:  # Only check if it's a separate folder
                unknown_labels_val = self.get_unknown_labels_in_folder(val_images_path)
                if unknown_labels_val:
                    messagebox.showerror("Unknown Labels", 
                                         f"The following labels in the validation data are not in the current model's labels: {', '.join(unknown_labels_val)}\n\n"
                                         f"Please update your labels before training.")
                    return

            print('images path\n',images_path)
            # Load the training data
            train_dataset = core.Dataset(images_path, images_path)
            # Load the validation data
            val_dataset = core.Dataset(val_images_path, val_images_path)
            losses = self.model.fit(train_dataset, val_dataset, epochs=num_epochs, learning_rate=learning_rate,
                                    gamma=gamma)
            messagebox.showinfo("Training Complete", f"Model training completed successfully. Loss: {losses}")
            # Add update_model_status() call after successful training
            if self.use_yolo:
                if self.model_yolo is not None:
                    self.update_model_status()
            else:
                if self.model is not None:
                    self.update_model_status()

    def update_model_status(self):
        """Update the model status display"""
        if self.use_yolo:
            if self.model_yolo is not None:
                model_name = self.model_yolo.model.name if hasattr(self.model_yolo.model,
                                                                   'name') else "Custom YOLO model"
                self.model_type_label.config(text="Model type: YOLO")
                self.model_path_label.config(text=f"Model path: {self.model_path if self.model_path else 'N/A'}")
            else:
                self.model_type_label.config(text="Model type: None")
                self.model_path_label.config(text="Model path: None")
        else:
            if self.model is not None:
                self.model_type_label.config(text="Model type: Detecto")
                self.model_path_label.config(text=f"Model path: {self.model_path if self.model_path else 'N/A'}")
            else:
                self.model_type_label.config(text="Model type: None")
                self.model_path_label.config(text="Model path: None")

    def clear_model(self):
        self.model = None
        self.model_yolo = None
        self.update_model_status()
        messagebox.showinfo("Model Cleared", "Model has been cleared successfully.")

    def load_pretrained_model(self, model_path=None):
        if not model_path:
            model_path = filedialog.askopenfilename(title="Select pretrained model file",
                                                    filetypes=[("Model files", "*.pth;*.pt")])
        if model_path:
            self.model_path = model_path  # Store the model path
            withoutext, ext = os.path.splitext(model_path)
            print(ext)
            label_path = withoutext + '.txt'
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    self.labels = [x.strip() for x in f.read().split(",")]
            self.labels_entry.delete('1.0', 'end')  # Clear existing content
            self.labels_entry.insert('1.0', ', '.join(self.labels))

            print('make sure labels are loaded correctly')
            if ext == '.pth':
                self.model = core.Model.load(model_path, self.labels)
                self.update_model_status()
            elif ext == '.pt':
                self.model_yolo = YOLO(model_path)
                self.update_model_status()

    def save_model(self):
        if self.use_yolo:
            if self.model_yolo is None:
                messagebox.showerror("Error", "No model to save. Train a model or load a pretrained model first.")
                return
            save_path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("Model files", "*.pt")])
            if save_path:
                self.model_yolo.save(save_path)
                label_path = save_path.replace(".pt", ".txt")
                with open(label_path, "w") as f:
                    f.write(", ".join(map(str, self.labels)))
                messagebox.showinfo("Model Saved", "Model and labels saved successfully.")
        else:

            if self.model is None:
                messagebox.showerror("Error", "No model to save. Train a model or load a pretrained model first.")
                return
            save_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("Model files", "*.pth")])
            if save_path:
                self.model.save(save_path)
                label_path = save_path.replace(".pth", ".txt")
                with open(label_path, "w") as f:
                    f.write(", ".join(map(str, self.labels)))
                messagebox.showinfo("Model Saved", "Model and labels saved successfully.")
        self.model_path = save_path
        self.update_model_status()

    # endregion

    # region XML Operations

    def filter_xml_objects(self):
        source_folder = filedialog.askdirectory(title="Select folder containing xmls")
        keep_list = tk.simpledialog.askstring("Enter the tags to keep space separated",
                                              prompt="Enter the tags to keep space separated",
                                              initialvalue="DCS INST SP")
        keep_list = keep_list.split()
        print(keep_list)
        if source_folder:
            for filename in os.listdir(source_folder):
                if filename.endswith(".xml"):
                    xml_path = os.path.join(source_folder, filename)

                    filter_annotation_file(xml_path, keep_list)

    def filterout_xml_objects(self):
        source_folder = filedialog.askdirectory(title="Select folder containing xmls")
        keep_list = tk.simpledialog.askstring("Enter the tags to remove space separated",
                                              prompt="Enter the tags to remove space separated",
                                              initialvalue="DCS INST SP")
        remove_list = keep_list.split()
        print(remove_list)
        if source_folder:
            for filename in os.listdir(source_folder):
                if filename.endswith(".xml"):
                    xml_path = os.path.join(source_folder, filename)

                    filterout_annotation_file(xml_path, remove_list)

    def rename_xml_objects(self):
        source_folder = filedialog.askdirectory(title="Select folder containing xmls")
        rename_list = tk.simpledialog.askstring("Enter the tags to rename use arrows",
                                                prompt="Enter the tags to rename use arrows -> and space",
                                                initialvalue="inst->INST dcs->DCS")
        rename_list = rename_list.split()
        rename_list = [item.split("->") for item in rename_list]
        print(rename_list)
        if source_folder:
            for filename in os.listdir(source_folder):
                if filename.endswith(".xml"):
                    xml_path = os.path.join(source_folder, filename)
                    rename_labels(xml_path, rename_list)

    def merge_xml_objects(self):
        source_folder = filedialog.askdirectory(title="Select folder containing xmls to extract objects from")
        target_folder = filedialog.askdirectory(title="Select folder containing xmls to append to")

        if source_folder and target_folder:
            for filename in os.listdir(source_folder):
                if filename.endswith(".xml"):
                    try:
                        source_xml_path = os.path.join(source_folder, filename)
                        target_xml_path = os.path.join(target_folder, filename)
                        merge_xml_objects(source_xml_path, target_xml_path)
                    except:
                        print('error on ', filename)

    def create_annotation_files(self):
        source_folder = filedialog.askdirectory(title="Select folder containing images")
        output_folder = source_folder
        min_score = tk.simpledialog.askinteger("Minimum Score percent", "Enter a Minimum Score percent:",
                                               initialvalue=75) / 100

        if source_folder:
            for filename in os.listdir(source_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(source_folder, filename)
                    img = cv2.imread(image_path)
                    height, width, channels = img.shape
                    size = (int(height), int(width))
                    labels, boxes, scores = model_predict_on_mosaic(img, self.model)
                    create_annotation_file(image_path, size, labels, boxes, scores, output_folder, minscore=min_score)

            remove_orphan_images(source_folder)

    def convert_xml_to_yolo(self):
        print('make sure labels are loaded')
        source_folder = filedialog.askdirectory(title="Select folder containing XMLs")
        output_folder = filedialog.askdirectory(title="Select output folder")
        data_split = tk.simpledialog.askfloat("Data split", "Enter ratio of that data goes to training",
                                              initialvalue=0.9)

        self.load_labels_auto(source_folder)

        create_yolov8_dataset(source_folder, output_folder, self.labels, train_ratio=data_split)

    # endregion

    # region Image Processing

    def splat_images(self):
        # self.root.withdraw()  # Hide the main window
        splat_gui_window = tk.Toplevel(self.root)
        SplatImagesGUI(splat_gui_window)

    def on_close(self, window):
        self.root.deiconify()  # Show the main window again when the secondary window is closed
        window.destroy()

    def add_noise(self):
        path = filedialog.askdirectory(title="Select folder containing images")
        scale = tk.simpledialog.askfloat("Noise", "Enter a scale for noise power", initialvalue=0.5)
        add_noise(path, scale)

    def make_patches(self):
        source_folder = filedialog.askdirectory(title="Select folder containing big images")
        destination_folder = filedialog.askdirectory(title="Select folder to send patches to")
        res = tk.simpledialog.askinteger("Patch size", "Enter a patch size:", initialvalue=1300)
        process_and_save_patches(source_folder, destination_folder, res)

    def test_model_on_image(self):
        # Ask for an image to test
        image_path = filedialog.askopenfilename(title="Select an image for testing",
                                                filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg")))
        self.test_model_mosaic(image_path)

    def test_model_on_random_images(self):
        # Ask for an image folder to test
        root_path = filedialog.askdirectory(title="Select a folder containing images for testing")
        count = tk.simpledialog.askinteger(title="image count",
                                           prompt="Enter the number of images to run inference on: ")
        min_width = tk.simpledialog.askinteger(title="min image width", prompt="Enter the minimum image width (px): ")

        # Get a list of all image files (png and jpg) in the folder and its subfolders
        image_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(('.png', '.jpg')):
                    image_files.append(os.path.join(root, file))

        print('Number of images:', len(image_files))
        if image_files:
            for i in range(count):
                if not image_files:
                    print("No more images available.")
                    break

                random_index = random.randint(0, len(image_files) - 1)
                random_image_path = image_files[random_index]

                # Check the image width
                with Image.open(random_image_path) as img:
                    width, _ = img.size

                if width >= min_width:
                    print(f"Selected random image: {random_image_path}")
                    self.test_model_mosaic(random_image_path)
                else:
                    print(f"Skipped image (width < {min_width}): {random_image_path}")

                # Remove the processed image from the list
                image_files.pop(random_index)
        else:
            print("No images found in the selected folder.")

    def test_model_mosaic(self, image_path):
        minscore = int(self.minscore_entry.get()) / 100
        patch_size = int(self.patch_size_entry.get())
        stride = int(self.stride_entry.get())
        nms_threshold = float(self.nms_threshold_entry.get())

        if image_path:
            image = cv2.imread(image_path)
            if self.model and self.use_yolo == False:
                labels, boxes, scores = model_predict_on_mosaic(image, self.model,
                                                                square_size=patch_size, stride=stride,
                                                                threshold=nms_threshold)
            elif self.model_yolo and self.use_yolo == True:
                labels, boxes, scores = model_predict_on_mosaic(image, self.model_yolo,
                                                                threshold=nms_threshold, yolo=True,
                                                                label_names=self.labels)

            # Overlay boxes and labels on the image
            img_with_boxes = self.plot_pic(image, labels, boxes, scores, minscore=minscore)

            # Use system temp directory for output
            output_image_path = os.path.join(tempfile.gettempdir(), 'result_image.png')
            cv2.imwrite(output_image_path, img_with_boxes)

            # Open the saved image using the default image viewer
            if os.path.exists(output_image_path):
                os.startfile(output_image_path)
            else:
                messagebox.showerror("Error", f"Failed to save result image at {output_image_path}")

    def plot_pic(self, img, labels, boxes, scores, size=5, minscore=.3):
        img = copy.copy(img)
        # plot_pic(img,labels,boxes,scores)
        # Define some colors for the boxes and labels

        for label, box, score in zip(labels, boxes, scores):
            if score > minscore:
                # Extract the coordinates of the box
                my_list = box
                my_list = [int(x) for x in my_list]
                x1, y1, x2, y2 = my_list

                # Draw a rectangle around the box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                # Add a label above the box
                cv2.putText(img, label + ":0." + str(int(float(score * 1000))), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            .7,
                            (255, 0, 0), thickness=2)

        # Display the image with the boxes and labels overlayed using Matplotlib
        # fig, ax = plt.subplots(figsize=(size, size))
        # ax.imshow(img)
        # plt.show()
        return img

    # endregion

    # region Label Management

    def load_labels(self):
        folder_path = filedialog.askdirectory(title="Select folder containing xmls")

        all_object_names = []

        # Iterate through XML files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)

                # Load XML data
                with open(file_path, 'r') as xml_file:
                    xml_data = xml_file.read()

                # Parse XML data
                root = ET.fromstring(xml_data)

                # Extract object names
                object_names = [obj.find('name').text for obj in root.findall('.//object')]
                all_object_names.extend(object_names)

        all_object_names_set = set(all_object_names)
        all_object_names = list(all_object_names_set)
        all_object_names.sort()
        # Update the content of the Text widget
        self.labels_entry.delete('1.0', 'end')  # Clear existing content
        self.labels_entry.insert('1.0', ', '.join(all_object_names))  # Insert the new content

    def load_labels_auto(self, folder_path):

        all_object_names = []

        # Iterate through XML files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)

                # Load XML data
                with open(file_path, 'r') as xml_file:
                    xml_data = xml_file.read()

                # Parse XML data
                root = ET.fromstring(xml_data)

                # Extract object names
                object_names = [obj.find('name').text for obj in root.findall('.//object')]
                all_object_names.extend(object_names)

        all_object_names_set = set(all_object_names)
        all_object_names = list(all_object_names_set)
        all_object_names.sort()
        self.labels = all_object_names

    def get_labels_from_user(self):
        # Get the labels from the Text widget and split them by newline
        labels_text = self.labels_entry.get("1.0", "end-1c")  # Get all the text
        split_labels = re.split(r'[\n, ]+', labels_text)
        print(split_labels)
        # Filter out any empty labels
        self.labels = split_labels

        messagebox.showinfo("Labels Set", f"Labels set to:\n{', '.join(self.labels)}")

    def get_unknown_labels_in_folder(self, folder_path):
        """Check if the folder contains XML files with labels that are not in self.labels"""
        unknown_labels = set()
        
        # Iterate through XML files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                
                # Load XML data
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    
                    # Extract object names
                    for obj in root.findall('.//object'):
                        name = obj.find('name').text
                        if name not in self.labels:
                            unknown_labels.add(name)
                except ET.ParseError:
                    print(f"Error parsing XML file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
        
        return list(unknown_labels)

    # endregion

    def clear_console(self):
        """Clear the console text widget"""
        self.console_text.delete('1.0', tk.END)
    
    def save_console_output(self):
        """Save the console output to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.console_text.get('1.0', tk.END))
            messagebox.showinfo("Console Output Saved", f"Console output saved to {file_path}")

    # Restore original stdout when the application closes
    def __del__(self):
        try:
            sys.stdout = self.original_stdout
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
