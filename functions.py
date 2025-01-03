
import fitz
from PyPDF2 import PdfMerger
import copy
import re
import hashlib
import time
import pickle
import openpyxl
import os
import torch
from new_fns import *
from easyocr_mosaic import HardOCR

# Standalone Functions

# region Utilities

def generate_color(string):
    # Generate a hash value for the input string using MD5
    hash_value = hashlib.md5(string.encode('utf-8')).hexdigest()

    # Take the first six characters of the hash value
    hex_code = hash_value[:6]

    # Convert the hex code to RGB values
    r, g, b = int(hex_code[:2], 16), int(hex_code[2:4], 16), int(hex_code[4:], 16)

    # Return the RGB values as a tuple
    return (r, g, b)

def calculate_center(box):
    """Calculate the center of a box."""
    x_min, y_min, x_max, y_max = box.tolist()
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return np.array([center_x, center_y])

def calculate_distance(center1, center2):
    """Calculate the Euclidean distance between two centers."""
    return np.linalg.norm(center1 - center2)

def point_in_box(x, y, box):
    xmin, ymin, xmax, ymax = box
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
    return False

def overlap_bbox(obox, dbox, extension, include_inside):
    #this fn works as follows
    #if include inside is true, we ONLY check for complete enclosed items
    #if it's not we will ignore completely enclosed items
    #so if its not completely enlcosed but there is overlap we return true else we return false


    # box 1 is a text box, box2 is an object box
    # Extract coordinates of box1
    x1_min, y1_min, x1_max, y1_max = obox

    # Extract coordinates of box2
    x2_min, y2_min, x2_max, y2_max = dbox

    # Extend box2 by the given extension
    x2_min_ext = max(x2_min - extension, 0)
    y2_min_ext = max(y2_min - extension, 0)
    x2_max_ext = x2_max + extension
    y2_max_ext = y2_max + extension
    if include_inside:
        # Check for overlap. includes completely enclosed boxes
        if (x1_min <= x2_max_ext and x1_max >= x2_min_ext and y1_min <= y2_max_ext and y1_max >= y2_min_ext):
            return True
        else:
            return False
    else:
        # Check if box1 is completely inside un-extended box2
        if x2_min <= x1_min and x2_max >= x1_max and y2_min <= y1_min and y2_max >= y1_max:
            return False
        else:
            # Check for overlap.
            if (x1_min <= x2_max_ext and x1_max >= x2_min_ext and y1_min <= y2_max_ext and y1_max >= y2_min_ext):
                return True
            else:
                return False

def save_easyocr_results_pickle(results, filename='easyocr_results.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_easyocr_results_pickle(filename='easyocr_results.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("File not found.")
        return []

def pil_to_cv2(pil_image):
    """
    Convert a PIL image to a cv2 image.
    :param pil_image: PIL image object
    :return: cv2 image object
    """
    # Convert PIL image to numpy array
    numpy_image = np.array(pil_image)

    # If the image has a transparency channel (RGBA)
    if numpy_image.shape[-1] == 4:
        # Convert RGBA to RGB by dropping the transparency channel
        numpy_image = numpy_image[:, :, :3]

    # Convert numpy array to cv2 image
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return cv2_image
# endregion


# region File Operations

def compile_excels(input_folder, output_folder=None, prefix='page', timestamp=True, recursive=False):
    """
    Compile multiple Excel files into a single Excel file.

    Args:
    input_folder (str): Path to the folder containing input Excel files.
    output_folder (str, optional): Path to save the output Excel file. If None, uses input_folder.
    prefix (str, optional): Prefix of Excel files to compile. Defaults to 'page'.
    timestamp (bool, optional): Whether to include a timestamp in the output filename. Defaults to True.
    recursive (bool, optional): Whether to search for Excel files recursively in the input folder. Defaults to False.

    Returns:
    str: Path to the compiled Excel file.
    """
    if output_folder is None:
        output_folder = input_folder

    if timestamp:
        timestamp_str = f"_{int(time.time())}"
    else:
        timestamp_str = ""

    output_filename = os.path.join(output_folder, f"compiled_output{timestamp_str}.xlsx")

    output_workbook = openpyxl.Workbook()
    output_sheet = output_workbook.active
    first_file = True

    if recursive:
        for root, dirs, files in os.walk(input_folder):
            for file_name in files:
                if file_name.endswith(".xlsx") and file_name.startswith(prefix):
                    file_path = os.path.join(root, file_name)
                    workbook = openpyxl.load_workbook(file_path)
                    sheet = workbook.active

                    if first_file:
                        for row in sheet.iter_rows(min_row=1, max_row=1):
                            for cell in row:
                                output_sheet[cell.coordinate].value = cell.value
                        first_file = False

                    for row in sheet.iter_rows(min_row=2):
                        output_sheet.append([cell.value for cell in row])
    else:
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".xlsx") and file_name.startswith(prefix):
                file_path = os.path.join(input_folder, file_name)
                workbook = openpyxl.load_workbook(file_path)
                sheet = workbook.active

                if first_file:
                    for row in sheet.iter_rows(min_row=1, max_row=1):
                        for cell in row:
                            output_sheet[cell.coordinate].value = cell.value
                    first_file = False

                for row in sheet.iter_rows(min_row=2):
                    output_sheet.append([cell.value for cell in row])

    output_workbook.save(output_filename)
    print('excels compiled to: ', output_filename)
    return output_filename

def merge_pdf(directory_path):
    # Set the directory path
    #directory_path = r"C:\Users\dcaoili\OneDrive - Samuel Engineering\Documents\PIDS - WORKING\missouri cobalt 22024\New folder (2)\2024-04-17 Area 620"

    # Create a PDF merger object
    pdf_merger = PdfMerger()

    # Loop through all the files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            # Open the file and append it to the PDF merger object
            print(f"opening {filename}")
            with open(os.path.join(directory_path, filename), 'rb') as pdf_file:
                pdf_merger.append(pdf_file)

    # Write the merged PDF to a file
    with open(os.path.join(directory_path, 'merged.pdf'), 'wb') as output_file:
        pdf_merger.write(output_file)

def pdf2png(pdf_file, target_width):
    doc = fitz.open(pdf_file)
    page = doc[0]  # Assume the first page is representative of the document

    # Calculate the required DPI to achieve the target width
    page_width = page.rect.width
    dpi = int(target_width * 72 / page_width)

    # Create a new folder to store the PNG images
    folder_name = os.path.join(os.path.dirname(pdf_file), os.path.splitext(os.path.basename(pdf_file))[0] + "_images")
    os.makedirs(folder_name, exist_ok=True)

    # Iterate through the pages and save them as PNG images
    for page in doc:
        # Render the page as a pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

        # Save the pixmap as a PNG image in the new folder
        png_file = os.path.join(folder_name, f"page_{page.number + 1}.png")
        pix.save(png_file)
        print(png_file)
    return folder_name
# endregion


# region Text Processing

def merge_common_substrings(str1, str2):
    # Split the strings into lists of words
    words1 = str1.split()
    words2 = str2.split()

    merged_words = []
    i, j = 0, 0

    while i < len(words1) and j < len(words2):
        if words1[i] == words2[j]:
            merged_words.append(words1[i])
            i += 1
            j += 1
        else:
            merged_word = ""
            if i < len(words1):
                merged_word += words1[i]
            if j < len(words2):
                merged_word += "/" + words2[j]
            merged_words.append(merged_word)
            i += 1
            j += 1

    # Append any remaining words from either string
    merged_words.extend(words1[i:])
    merged_words.extend(["/" + word for word in words2[j:]])
    last_word = merged_words[-1:][0]
    print('last word', last_word)
    joined_words = " ".join(merged_words)
    if not any(char.isdigit() for char in last_word):
        joined_words = joined_words+"S"
    return joined_words

def condense_hyphen_string(s):
    if not s:
        return ""
    words = s.split()
    result = []

    for word in words:
        if '/' in word:
            parts = word.split('/')
            prefixes = [part.split('-')[:-1] for part in parts]
            suffixes = [part.split('-')[-1] for part in parts]

            if all(prefix == prefixes[0] for prefix in prefixes):
                condensed = '-'.join(prefixes[0] + ['/'.join(suffixes)])
                result.append(condensed)
            else:
                result.append(word)
        else:
            result.append(word)

    return ' '.join(result)
# endregion


# region Image Processing


def draw_detection_boxes(img, labels, boxes, scores, size=5, minscore=.5):
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
            cv2.putText(img, label + ":0." + str(int(float(score * 1000))), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (255, 0, 0), thickness=2)

    # Display the image with the boxes and labels overlayed using Matplotlib
    # fig, ax = plt.subplots(figsize=(size, size))
    # ax.imshow(img)
    # plt.show()
    return img

def remove_objects_from_image(img, inst_data, ocr_results):
    # Create a copy of the original image
    masked_img = img.copy()

    # Process object results
    for data in inst_data:
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, data['box'])

        # Draw a white filled rectangle to cover the object
        cv2.rectangle(masked_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Process OCR results
    for result in ocr_results:
        box = result[0]
        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])

        # Draw a white filled rectangle to cover the OCR text
        cv2.rectangle(masked_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    return masked_img

def plot_ocr_results(img, results):
    img_ocr_results = img.copy()
    for result in results:
        # Get the recognized text
        text = result[1]
        box = result[0]
        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])

        pt1 = (x1, y1)
        pt2 = (x2, y2)
        # draw boxes for viewing results
        cv2.rectangle(img_ocr_results, pt1, pt2, (0, 0, 255), 2)
        # Add the recognized text on top of the box
        cv2.putText(img_ocr_results, text, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, .7, (0, 0, 255), 2)

    # plot_pic2(img_ocr_results,20)
    return img_ocr_results



# endregion


# region OCR and Text Analysis

def get_text_in_ocr_results(re_pattern, results):
    pattern = re.compile(re_pattern)
    text_list = []
    for result in results:
        # we need a list because we might want to overwrite
        text = result[1]
        if pattern.match(text):
            text_list.append(result)
    # returning a list of results that match the text
    return text_list

def get_comment(ocr_results, dbox, box_expand, include_inside=False, remove_pattern='^$'):
    # dbox is a detecto box
    # obox is an easyOCR box
    comment = ''
    object_box = dbox.tolist()
    object_box = [int(x) for x in object_box]


    for item in ocr_results:
        #we cant do for obox, word, score in ocr because ocr_results changes lenght i.e. color or a word is appened
        obox, word, confidence = item


        text_box = [obox[0][0], obox[0][1], obox[2][0], obox[2][1]]

        if overlap_bbox(text_box, object_box, box_expand, include_inside):
            comment += f" {word}"
    return comment

# endregion


# region Line Processing

def process_line_data(img, ocr_results, re_line):
    """Process line-related data from an image and OCR results.

    Returns:
        dict: Keys are line text, values are dict containing 'image' and 'color'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=1, threshold=20, minLineLength=40,
                            maxLineGap=80)

    line_groups = {}
    line_boxes = {}
    all_line_groups = group_lines(lines)
    img_line_groups_removed = white_out_line_groups(img, all_line_groups, line_thickness=5)
    ocr_lines = get_text_in_ocr_results(re_line, ocr_results)

    for line_group in all_line_groups:
        for box, text, confidence in ocr_lines:
            if box_intersects_line(box, line_group, scale=1.5):
                line_groups.setdefault(text, []).append(line_group)
                line_boxes.setdefault(text, []).append(box)

    line_data = {}
    for line_text in line_groups:
        color = generate_color(line_text)
        img = place_lines_on_image_and_flood_fill(
            img_line_groups_removed,
            line_groups[line_text],
            color,
            line_boxes[line_text]
        )
        line_data[line_text] = {
            'image': img,
            'color': color
        }

    return line_data


def get_lines_from_box(box, line_data, img_scale=1.0):
    lines = []
    for line_text, data in line_data.items():
        mode_color = region_mode_color(box, data['image'], img_scale)
        expected_color = data['color']

        mode_color = np.array(mode_color)
        expected_color = np.array(expected_color)

        if np.all(np.abs(mode_color - expected_color) <= 5):
            lines.append(line_text)

    return sorted(lines)

def place_lines_on_image_and_flood_fill(img_line_groups_removed, line_group, line_color, line_box, line_thickness=5):
    """
    Places lines and performs flood fill on image regions.

    Args:
        img_line_groups_removed: Base image with other lines removed
        line_group: Group of lines to draw
        line_color: Color to use for lines
        line_box: Bounding boxes for text regions
        line_thickness: Thickness of drawn lines
    """
    img = img_line_groups_removed.copy()

    # Draw lines
    for line in line_group:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), (0,0,0), line_thickness)

    # Create seed points for flood fill around text boxes
    seed_pixel_boxes = [(box, line_color) for box in line_box]

    # Perform flood fill from text regions
    filled_img = multi_region_flood_fill(seed_pixel_boxes, img, expand_scale=1.2)

    return filled_img

def multi_region_flood_fill(seed_pixel_boxes, line_img, expand_scale):
    # Create a new image to perform the flood fill on
    img = line_img.copy()

    # Get the size of the image
    height, width, channels = img.shape

    # Initialize the pixel stacks with the outer boundary pixels in each box
    pixel_stacks = []
    for i, seed_box in enumerate(seed_pixel_boxes):
        box, color = seed_box

        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        expand_h = int(abs(x1-x2)*(expand_scale-1)/2)
        expand_v = int(abs(y1-y2)*(expand_scale-1)/2)
        x1, y1, x2, y2 = x1-expand_h, y1-expand_v, x2+expand_h, y2+expand_v
        # Create a new list for each pixel stack
        pixel_stack = []

        # Draw and fill the rectangle using cv2.rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)

        # Iterate over the top and bottom rows of the box and append their pixels
        for x in range(x1, x2 + 1):
            pixel_stack.append((x, y1, color))
            pixel_stack.append((x, y2, color))

        # Iterate over the left and right columns of the box (excluding corners) and append their pixels
        for y in range(y1, y2 + 1):
            pixel_stack.append((x1, y, color))
            pixel_stack.append((x2, y, color))

        # Append the pixel stack for this box to the list of pixel stacks
        pixel_stacks.append(pixel_stack)

    # While there is still an element in any of the lists:
    # An empty list returns false
    while pixel_stacks:
        # Enumerate gives us an index n
        for pixel_stack in pixel_stacks:
            # print("chekcing stack "+str(n))
            start_size = len(pixel_stack)
            # i goes from the bottom to the top
            for i in range(start_size):
                pixel = pixel_stack.pop(0)
                x, y, color = pixel
                # Get neighbors
                neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for neighbor in neighbors:
                    nx, ny = neighbor
                    # Check if neighbor is within image bounds
                    if (0 <= nx < width) and (0 <= ny < height):
                        # Get the color of the neighboring pixel
                        neighbor_color = tuple(img[ny, nx])
                        # If the neighboring pixel is black
                        if neighbor_color == (0, 0, 0):
                            # Add pixel to pixel list
                            # print("append")
                            pixel_stack.append((nx, ny, color))
                            # Set color on filled image
                            img[ny, nx] = color

        # Remove empty sublists
        pixel_stacks = [x for x in pixel_stacks if x]

    # Return the filled image
    return img

def region_mode_color(box, img, img_scale):
    x_min, y_min, x_max, y_max = box.tolist()
    roi = img[int(y_min / img_scale):int(y_max / img_scale), int(x_min / img_scale):int(x_max / img_scale)]

    # Create a mask for non-black and non-white pixels
    non_black_white_mask = np.all((roi != [0, 0, 0]) & (roi != [255, 255, 255]), axis=2)

    # Extract non-black, non-white pixel colors
    non_black_white_pixels = roi[non_black_white_mask]

    # Find the most popular color among non-black, non-white pixels
    unique_colors, counts = np.unique(non_black_white_pixels, axis=0, return_counts=True)
    try:
        most_popular_color = unique_colors[np.argmax(counts)]
    except:
        return [0,0,0]

    return most_popular_color


# endregion


# region Instrument Processing



def return_inst_data(prediction_data, img, reader, rs, expand=0.0, radius=180,
                     inst_labels=None,
                     other_labels=None,
                     min_scores=None,
                     offset=None, ocr_results=None, comment_box_expand=30,
                     tag_label_groups=None,
                     re_line=None, capture_ocr=False):

    all_data = []
    group_inst = []
    group_other = []
    got_one = False

    # Separate the groups
    for label, box, score in prediction_data:

        try:
            if score < min_scores[label]:
                continue
        except Exception as e:
            print(e, '. Maybe try setting minscores --> settings >> set object minscores')

        if label in inst_labels:
            group_inst.append((label, box, score))
        if label in other_labels:
            group_other.append((label, box, score))

    line_images = {}
    line_colors = {}
    if ocr_results and re_line:
        line_images, line_colors = process_line_data(img, ocr_results, re_line)

    for label, box, score in group_inst:

        # get lines_lines_from_box figures out what lines the current instrument has by using region_mode_color(box, img, img_scale)
        lines = ['']
        if re_line:
            lines = get_lines_from_box(box, line_images, line_colors)
        #lines is simply a list of strings

        tag, tag_no = '', ''
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        inst_center = calculate_center(box)

        x_expand = int((x_max - x_min) * expand / 2)
        y_expand = int((y_max - y_min) * expand / 2)

        crop_img = img[(y_min - y_expand):(y_max + y_expand), (x_min - x_expand):(x_max + x_expand)]
        try:
            results = reader.readtext(crop_img, **rs)
        except Exception as e:
            print('error', e)
            continue

        if results:
            if not got_one:
                filename = 'instrument_capture.png'
                cv2.imwrite(filename, crop_img)
                got_one = True

            tag = results[0][1]
            tag_no = ' '.join([box[1] for box in results[1:]])

        # Only use tag_label_groups filtering if it's provided and we have a tag
        valid_types = ['']
        if tag_label_groups and tag:
            valid_types = get_valid_types_for_tag(tag, tag_label_groups)
            print('valid types ', valid_types)

        # If tag_label_groups is None or no valid types were found,
        # find_closest_other will work like the original version
        inst_type = find_closest_other(inst_center, group_other, label, radius, valid_types, tag_label_groups)

        comment = ''
        offset_tensor = torch.tensor([offset[0], offset[1], offset[0], offset[1]])
        offset_box = box + offset_tensor
        if ocr_results:
            comment = get_comment(ocr_results, offset_box, comment_box_expand)

        data = {'tag': tag, 'tag_no': tag_no, 'score': score, 'box': offset_box, 'label': label, 'type': inst_type,
                'comment': comment, 'line': lines}
        all_data.append(data)

    return all_data

def get_valid_types_for_tag(tag, tag_label_groups):
    # For each key in tag_label_groups, split it into individual tags
    # and check if our tag starts with any of them
    for tag_group, valid_types in tag_label_groups.items():
        tag_prefixes = tag_group.split()
        if tag in tag_prefixes:
            return valid_types
    return None  # Return None if no matching group found

def find_closest_other(inst_center, group_other, current_label, radius, valid_types=[''], tag_label_groups=None):
    """
    Find the closest item in group_other that doesn't have the same label and is in valid_types if specified.

    Args:
        inst_center: Center coordinates of the current instrument
        group_other: List of (label, box, score) tuples for other objects
        current_label: Label of the current instrument to exclude
        radius: Maximum distance to consider
        valid_types: Optional list of valid type labels to consider
    """
    min_distance = radius
    closest_label = ''



    for label, box, score in group_other:

        # Skip items with the same label as the current instrument
        if label == current_label:
            continue

        if tag_label_groups:
            # Skip if we have valid_types and this label isn't in it
            if valid_types is None:
                continue
            print('label ', label)
            print('valid_types ', valid_types)
            if label not in valid_types:
                continue

        other_center = calculate_center(box)
        distance = calculate_distance(inst_center, other_center)
        if distance < min_distance:
            min_distance = distance
            closest_label = label

    return closest_label

# endregion

