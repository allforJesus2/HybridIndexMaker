import os
import random
from PIL import Image, ImageFilter
import xml.etree.cElementTree as ET
import re
import numpy as np
import cv2
import glob

def apply_transformations(image, p_mirror_x=0.5, p_mirror_y=0.5, uniform_scale=True, min_scale=0.8, max_scale=1.2,
                          black_threshold=50, color_probability=0.5, p_rotate=0.5):
    """Apply random transformations to the input image."""
    if color_probability > random.random():
        color = generate_random_color()
        image = replace_black_with_color(image, color, black_threshold)

    # Apply random 90-degree rotation
    if random.random() < p_rotate:
        rotation_choices = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        rotation = random.choice(rotation_choices)
        image = image.transpose(rotation)

    # Define transformations
    x = random.random()
    if x < p_mirror_x:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    x = random.random()
    if x < p_mirror_y:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # Apply random scaling
    if uniform_scale:
        scale_factor = random.uniform(min_scale, max_scale)  # Random scale factor
        new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
    else:
        scale_factor_x = random.uniform(min_scale, max_scale)  # Random scale
        scale_factor_y = random.uniform(min_scale, max_scale)  # Random scale
        new_size = (int(image.size[0] * scale_factor_y), int(image.size[1] * scale_factor_x))

    image = image.resize(new_size)
    return image

def replace_black_with_color(img, replacement_color, threshold):
    # Convert to grayscale
    gray_img = img.convert('L')

    # Convert to numpy array
    np_img = np.array(gray_img)

    # Convert to binary
    binary_img = (np_img > threshold).astype(np.uint8) * 255

    # Create a new RGB image
    result = np.stack([binary_img, binary_img, binary_img], axis=-1)

    # Replace black (0, 0, 0) with the specified color
    black_pixels = np.all(result == [0, 0, 0], axis=-1)
    result[black_pixels] = replacement_color

    # Create a new image from the array
    output_img = Image.fromarray(result.astype(np.uint8))
    return output_img


def generate_random_color(
        red_range=(0, 255),
        blue_range=(0, 255),
        green_range=(0, 30),
        brightness_scale=(0, 1.0)):

    # Generate random values for red and blue components
    red = random.randint(red_range[0], red_range[1])
    blue = random.randint(blue_range[0], blue_range[1])

    # Generate a random value for the green component, ensuring it is below 30
    green = random.randint(green_range[0], green_range[1])

    # Apply brightness scale
    brightness_factor = random.uniform(brightness_scale[0], brightness_scale[1])
    adjusted_red = int(red * brightness_factor)
    adjusted_blue = int(blue * brightness_factor)
    adjusted_green = int(green * brightness_factor)

    # Ensure color values stay within valid range
    adjusted_red = max(0, min(255, adjusted_red))
    adjusted_blue = max(0, min(255, adjusted_blue))
    adjusted_green = max(0, min(255, adjusted_green))

    return (adjusted_red, adjusted_green, adjusted_blue)

def add_noise(path, scale):
    # Path to the folder containing PNG images

    # Loop through all PNG images in the folder
    for file in os.listdir(path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)

            # Add random colored speckle noise
            noise = np.random.normal(0, random.random(), img.shape)
            noisy_img = img + img * noise.astype(np.float32) * scale

            # Save the noisy image with the same filename
            cv2.imwrite(os.path.join(path, file), noisy_img)

def place_object(new_image, input_image, object_name, existing_objects, output_size, transforms,
    bbox_scale_min=1.0, bbox_scale_max=1.0):
    """Place the input image on the new image and return object data."""
    input_image = apply_transformations(input_image, **transforms)

    # Keep trying until a non-overlapping position is found
    max_attempts = 300
    attempt = 0
    while attempt < max_attempts:
        x_offset = random.randint(0, output_size - input_image.size[0])
        y_offset = random.randint(0, output_size - input_image.size[1])

        # Check for overlap with existing objects
        overlap = False
        for obj in existing_objects:
            obj_xmin, obj_ymin, obj_xmax, obj_ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
            new_xmin, new_ymin, new_xmax, new_ymax = x_offset, y_offset, x_offset + input_image.size[0], y_offset + \
                                                                         input_image.size[1]

            if (new_xmin < obj_xmax and new_xmax > obj_xmin and
                    new_ymin < obj_ymax and new_ymax > obj_ymin):
                overlap = True
                break

        if not overlap:
            break

        attempt += 1

    # If no non-overlapping position is found after max_attempts, skip this object
    if attempt == max_attempts:
        return None

    new_image.paste(input_image, (x_offset, y_offset))
    if object_name != "NOT":  # Don't create bounding box for NOT objects
        bbox_scale = random.uniform(bbox_scale_min, bbox_scale_max)
        # Calculate the center of the bounding box
        center_x = x_offset + input_image.size[0] / 2
        center_y = y_offset + input_image.size[1] / 2

        # Calculate the new width and height of the bounding box
        new_width = input_image.size[0] * bbox_scale
        new_height = input_image.size[1] * bbox_scale

        # Calculate the new bounding box coordinates
        xmin = max(0, int(center_x - new_width / 2))
        ymin = max(0, int(center_y - new_height / 2))
        xmax = min(output_size, int(center_x + new_width / 2))
        ymax = min(output_size, int(center_y + new_height / 2))

        object_data = {
            "name": object_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }
        return object_data


import random
import numpy as np
from PIL import Image, ImageEnhance

def random_hue(image, min=0, max=180):
    # Convert image to HSV
    image_hsv = image.convert('HSV')
    data = np.array(image_hsv)
    # Randomly change the hue
    data[:, :, 0] = (data[:, :, 0] + random.randint(min, max)) % 180
    return Image.fromarray(data, 'HSV').convert('RGB')

def random_brightness(image, min=0.5, max=1.5):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(min, max)  # Random factor between 0.5 and 1.5
    return enhancer.enhance(factor)

def random_contrast(image, min=0.5, max=1.5):
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(min, max)  # Random factor between 0.5 and 1.5
    return enhancer.enhance(factor)


def random_invert(image, invert_probability):

    if random.random() < invert_probability:
        # Convert the image to an array
        image_array = np.array(image)

        # Invert the colors
        inverted_array = ~image_array

        # Convert the array back to an image
        inverted_image = Image.fromarray(inverted_array)
        return inverted_image
    else:
        return image
def random_blur(new_image, blur_probability, blur_max):
    if random.random() < blur_probability:
        blur_radius = random.uniform(0, blur_max)
        return new_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return new_image


def generate_output_image(root_folder, output_folder, **kwargs):
    output_size = kwargs.get('output_size', 1300)  # Default value if not provided
    not_weight = kwargs.get('not_weight', 0.2)  # Default value
    yolo = kwargs.get('yolo', False)  # Default value
    transforms = {
        'p_mirror_x': kwargs.get('p_mirror_x', 0.5),
        'p_mirror_y': kwargs.get('p_mirror_y', 0.5),
        'min_scale': kwargs.get('min_scale', 0.9),
        'max_scale': kwargs.get('max_scale', 1.2),
        'uniform_scale': kwargs.get('uniform_scale', False),
        'black_threshold': kwargs.get('black_threshold', 50),
        'color_probability': kwargs.get('color_probability', 0.0),
        'p_rotate': kwargs.get('p_rotate', 0.5),

    }
    not_folder = kwargs.get('not_folder')
    # New parameter for random canvas color probability

    brightness_range = kwargs.get('brightness_range', (0.5, 1.5))  # Default value
    contrast_range = kwargs.get('contrast_range', (0.5, 1.5))  # Default value
    hue_range = kwargs.get('hue_range', (0, 180))  # Default value
    invert_probability = kwargs.get('invert_probability', 0.0)  # Default value

    bbox_scale_min = kwargs.get('bbox_scale_min', 1.0)
    bbox_scale_max = kwargs.get('bbox_scale_max', 1.0)

    splat_variance = kwargs.get('splats_variance', 0.2)  # Default value
    count = kwargs.get('splats_per_image', 50)  # Default value
    random_variance = random.uniform(1-splat_variance, 1+splat_variance)
    count = int(count*random_variance)

    blur_probability = kwargs.get('blur_probability', 0.2)
    blur_max = kwargs.get('blur_max', 1)

    """Generate an output image with 100 random objects and create the annotation file."""
    input_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith((".png", ".PNG", ".jpg", ".JPG")):
                input_files.append(os.path.join(root, file))

    new_image = Image.new("RGB", (output_size, output_size), (255,255,255))
    object_data_list = []
    # Get NOT kernel files
    not_files = []
    if not_folder:
        for root, _, files in os.walk(not_folder):
            for file in files:
                if file.endswith((".png", ".PNG", ".jpg", ".JPG")):
                    not_files.append(os.path.join(root, file))


    # Group input files by object name
    object_groups = {}
    for file_path in input_files:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        name = re.sub(r'\d+$', '', name)  # Remove trailing numbers
        object_groups.setdefault(name, []).append(file_path)

    # Place NOT objects
    not_count = int(count * not_weight)
    for _ in range(not_count):
        if not_files:
            file_path = random.choice(not_files)
            input_image = Image.open(file_path)
            place_object(new_image, input_image, "NOT", object_data_list, output_size, transforms)

    # Place other objects
    remaining_count = count - not_count
    for _ in range(remaining_count):
        object_name, object_files = random.choice(list(object_groups.items()))
        file_path = random.choice(object_files)
        input_image = Image.open(file_path)
        object_data = place_object(new_image, input_image, object_name, object_data_list, output_size, transforms,
                                   bbox_scale_min=bbox_scale_min,bbox_scale_max=bbox_scale_max)
        if object_data:
            object_data_list.append(object_data)


    new_image = random_hue(new_image, min=hue_range[0],max=hue_range[1])
    new_image = random_brightness(new_image, min=brightness_range[0], max=brightness_range[1])
    new_image = random_contrast(new_image, min=contrast_range[0], max=contrast_range[1])
    new_image = random_invert(new_image, invert_probability)
    new_image = random_blur(new_image, blur_probability, blur_max)

    output_path = os.path.join(output_folder, f"{random.randint(1, 1000000)}.png")
    new_image.save(output_path)

    if yolo:
        create_yolo_annotation_file_from_objects(object_data_list, output_path, output_size)
    else:
        create_annotation_file_from_objects(object_data_list, output_path, output_size)

def create_yolo_annotation_file_from_objects(objects, output_path, output_size):
    annotation_lines = []
    classes = set()

    for obj in objects:
        class_name = obj["name"]
        classes.add(class_name)

    classes = sorted(classes)

    for obj in objects:
        class_name = obj["name"]
        classes.add(class_name)

        xmin = obj["xmin"]
        ymin = obj["ymin"]
        xmax = obj["xmax"]
        ymax = obj["ymax"]

        # Convert bounding box coordinates to YOLO format
        x_center = (xmin + xmax) / (2 * output_size[0])
        y_center = (ymin + ymax) / (2 * output_size[1])
        bbox_width = (xmax - xmin) / output_size[0]
        bbox_height = (ymax - ymin) / output_size[1]

        # Create the annotation line in YOLO format
        annotation_line = f"{class_ids[class_name]} {x_center} {y_center} {bbox_width} {bbox_height}"
        annotation_lines.append(annotation_line)

    # Save the annotation lines to a text file
    annotation_path = os.path.splitext(output_path)[0] + ".txt"
    with open(annotation_path, "w") as f:
        f.write("\n".join(annotation_lines))

    # Save the class names to a separate file
    class_names_path = os.path.join(os.path.dirname(output_path), "classes.txt")
    with open(class_names_path, "w") as f:
        f.write("\n".join(sorted(classes)))

def create_annotation_file_from_objects(objects, output_path, output_size):
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = "compiled archive"

    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(output_path)

    path = ET.SubElement(root, "path")
    path.text = output_path

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(output_size)
    height = ET.SubElement(size, "height")
    height.text = str(output_size)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    for obj in objects:
        object_elem = ET.SubElement(root, "object")
        name = ET.SubElement(object_elem, "name")
        name.text = obj["name"]
        pose = ET.SubElement(object_elem, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object_elem, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(object_elem, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(object_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj["xmin"])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj["ymin"])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj["xmax"])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj["ymax"])

    tree = ET.ElementTree(root)
    xml_path = os.path.splitext(output_path)[0] + ".xml"
    tree.write(xml_path)

def remove_orphan_images(directory_path, suffix='.xml'):
    # Define the directory path
    # Find all PNG and JPG files in the directory
    image_files = glob.glob(os.path.join(directory_path, '*.[jp][pn]g'))

    for image_file in image_files:
        # Construct the expected name of the corresponding XML file
        xml_file_name = os.path.splitext(image_file)[0] + suffix

        # Check if the XML file exists
        if not os.path.exists(xml_file_name):
            # Delete the image file if no XML counterpart exists
            os.remove(image_file)
            print(f'Deleted {image_file} because no corresponding XML file was found.')

    print('Process completed.')

def create_annotation_file(image_path, size, labels, boxes, scores, output_dir, minscore=0.5):
    # Create the root element
    root = ET.Element("annotation")

    # Add folder element
    folder = ET.SubElement(root, "folder")
    folder.text = "images"

    # Add filename element
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)

    # Add path element
    path = ET.SubElement(root, "path")
    path.text = image_path

    # Add image source
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    # Add image size (assuming you have access to the image dimensions)
    # Replace this with the actual image dimensions
    img_width, img_height = size
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    objects_added = False

    # Add objects
    for label, box, score in zip(labels, boxes, scores):
        if score > minscore:
            objects_added = True  # Set flag to True when an object is added

            object_elem = ET.SubElement(root, "object")
            name = ET.SubElement(object_elem, "name")
            name.text = label
            pose = ET.SubElement(object_elem, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(object_elem, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(object_elem, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(object_elem, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(box[0]))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(box[1]))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(box[2]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(box[3]))

    # Check if any objects were added
    if not objects_added:
        print("No objects found above the minimum score threshold. Skipping file creation.")
        return  # Return early without creating the file

    # Create the output file path
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_dir, f"{filename_without_ext}.xml")

    # Write the XML tree to the output file
    tree = ET.ElementTree(root)
    tree.write(output_file)

    print(f"Annotation file saved: {output_file}")

def create_yolov5_annotation_file(image_path, size, all_labels, labels, boxes, scores, output_dir, minscore=0.5):
    # Ensure labels are integers representing class IDs
    # Assuming boxes are in the format [xmin, ymin, xmax, ymax]

    img_width, img_height = size

    label_id = {}
    for i, label in enumerate(all_labels):
        label_id[label] = i

    # Filter objects based on score threshold
    filtered_objects = [(label_id[label], box, score) for label, box, score in zip(labels, boxes, scores) if
                        score > minscore]

    if not filtered_objects:
        print("No objects found above the minimum score threshold. Skipping file creation.")
        return

    # Prepare YOLOv5 format annotations
    yolov5_annotations = []
    for label_id, box, _ in filtered_objects:
        xmin, ymin, xmax, ymax = box

        # Normalize bounding box coordinates
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Append annotation line
        yolov5_annotations.append(f"{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Create the output file path
    filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_dir, f"{filename_without_ext}.txt")

    # Write the annotations to the output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolov5_annotations))

    print(f"YOLOv5 annotation file saved: {output_file}")

def filter_annotation_file(input_file, keep_list):
    # Parse the input XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Iterate over the objects
    objects_to_remove = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in keep_list:
            objects_to_remove.append(obj)

    # Remove the unwanted objects
    for obj in objects_to_remove:
        root.remove(obj)

    # Write the modified XML tree to the output file
    output_file = input_file
    tree.write(output_file)

def filterout_annotation_file(input_file, remove_list):
    # Parse the input XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Iterate over the objects
    objects_to_remove = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in remove_list:
            objects_to_remove.append(obj)

    # Remove the unwanted objects
    for obj in objects_to_remove:
        root.remove(obj)

    # Write the modified XML tree to the output file
    output_file = input_file
    tree.write(output_file)

def rename_labels(input_file, rename_list):
    # Parse the input XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Iterate over the objects
    for obj in root.findall("object"):
        name = obj.find("name")
        label = name.text

        for pair in rename_list:
            if label == pair[0]:
                name.text = pair[1]

    # Write the modified XML tree to the output file
    output_file = input_file
    tree.write(output_file)

def merge_xml_objects(source_file, target_file):
    try:
        # Parse the source and target XML files
        source_tree = ET.parse(source_file)
        target_tree = ET.parse(target_file)

        # Get the root element of both trees
        source_root = source_tree.getroot()
        target_root = target_tree.getroot()

        # Confirm the root element's tag is 'annotation'
        if target_root.tag != "annotation":
            raise ValueError("Target file's root element is not '<annotation>'.")

        # Extract <object> elements from the source
        source_objects = source_root.findall('.//object')

        # Append each <object> from the source to the target's <annotation>
        for obj in source_objects:
            target_root.append(obj)

        # Write the modified target tree back to the target file
        target_tree.write(target_file, encoding='utf-8', xml_declaration=True)
        print(f"Objects successfully merged into {target_file}")
    except FileNotFoundError:
        print(f"Error: File not found. Please check the file paths for '{source_file}' and '{target_file}'.")
    except ET.ParseError:
        print(f"Error: Unable to parse one of the XML files. Ensure they are well-formed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")