
import fitz
from PyPDF2 import PdfMerger
import copy
import re
import hashlib
import time
import pickle
import tkinter as tk
from tkinter import ttk
import openpyxl
import cv2
import os
import torch
import numpy as np
from new_fns import *
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

def return_inst_data(prediction_data, img, reader, rs, expand=0.0, radius=180,
                     inst_labels=None,
                     other_labels=None,
                     min_scores=None,
                     offset=None, ocr_results=None, comment_box_expand=30,
                     tag_label_groups=None,
                     re_line=None):

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

    # region make line image
    # make colors
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length,
                                 maxLineGap=max_line_gap)

    line_dictionary = {}

    line_groups = group_lines(lines)
    img_line_groups_removed = white_out_line_groups(img, line_groups, line_thickness=5)


    line_texts = get_text_in_ocr_results(re_line, ocr_results)

    for line_group in line_groups:

        for line_text in line_texts:
            box = line_text[0]
            if box_intersects_line(box, line_group):
                if line_dictionary.get(line_text):
                    line_dictionary[line_text].append(line_group)
                else:
                    line_dictionary[line_text] = [line_group]

    # add colors
    color_dictionary = {}
    for line_text in line_dictionary:
        color_dictionary[line_text] = generate_color(line_text)

    image_dictionary = {}
    for line_text in line_dictionary:
        one_line_img = place_lines_on_image_and_flood_fill(img_line_groups_removed,
                                            line_dictionary[line_text],
                                            color_dictionary[line_text])
        image_dictionary[line_text] = one_line_img
    
    #make single dict
    line_data = {}
    for line in line_dictionary:
        line_data['lines'] = line_dictionary[line]
        line_data['color'] = color_dictionary[line]
        line_data['img'] = image_dictionary[line]

    # We then iterate though all the registered lines placing them back on the image one at a time
    # That line is drawn in with the line color and we use the flood fill algorithm to color the image and assign instruments the line number
    # end region
    '''
    for label, box, score in group_inst:

        #lines is simply a list of strings
        #lines = get_lines_from_box(box, line_data)

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
                'comment': comment}
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

def calculate_center(box):
    """Calculate the center of a box."""
    x_min, y_min, x_max, y_max = box.tolist()
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return np.array([center_x, center_y])

def calculate_distance(center1, center2):
    """Calculate the Euclidean distance between two centers."""
    return np.linalg.norm(center1 - center2)


#functions relevant to auto generate index
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

def process_images_and_data(img_T_replaced, img_no_equipment, img, ocr_results_with_rotated_text,
                            re_lines, re_equipment, erosion, shrink_factor, lower_thresh, line_box_expand, services,
                            equipment_boxes, equipment_box_expand, include_inside, img_scale):
    # Process line images
    processed_img_lines = process_image(img_T_replaced, erosion, shrink_factor, lower_thresh=lower_thresh)
    lines = get_text_in_ocr_results(re_lines, ocr_results_with_rotated_text)
    line_colors = return_hash_colors(lines)

    line_img = generate_painted_img(processed_img_lines, lines, line_colors, shrink_factor, expand_scale=line_box_expand)


    service_colors = return_service_colors(services, line_img, img_scale)
    processed_service_img = process_image(img_no_equipment, erosion, shrink_factor, lower_thresh=lower_thresh)

    #if we could just generate services_in and services_out as text lists with indexes equal to lines that'd be great

    # Process service in lines. idea is to return a subset of lines and colors, so we can repaint the image
    # services_in = get_services(lines, line_colors, services, service_colors, 'service_in')

    services_in, services_in_colors, services_in_txt = return_serviced_lines(lines, line_colors, services, service_colors, 'service_in')
    service_in_img = generate_painted_img(processed_service_img, services_in, services_in_colors, shrink_factor,
                                          expand_scale=line_box_expand)

    # Process service out lines where each index corresponds ot a service out for that line/line_color index
    services_out, services_out_colors, services_out_txt = return_serviced_lines(lines, line_colors, services, service_colors, 'service_out')
    service_out_img = generate_painted_img(processed_service_img, services_out, services_out_colors, shrink_factor,
                                           expand_scale=line_box_expand)

    # Process equipment images
    processed_img = process_image(img, erosion, shrink_factor, lower_thresh=lower_thresh)
    equipments, equipment_img = generate_equipment(processed_img, ocr_results_with_rotated_text, re_equipment,
                                                   shrink_factor, equipment_boxes, equipment_box_expand,
                                                   include_inside)
    # cv2.imwrite(os.path.join(results_folder, 'equipment_img.png'), equipment_img)





    services_in = list(zip(services_in_colors, services_in_txt))
    services_out = list(zip(services_out_colors, services_out_txt))


    return line_img, equipment_img, service_in_img, service_out_img, lines, line_colors, equipments, services_in, services_out

def process_image(img, erosion_kernel, shrink_factor, lower_thresh):
    print('copying img')
    line_img = copy.copy(img)

    # Convert the RGB image to grayscale
    print('to grayscale')
    line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to get a binary image
    print('thresholding')
    _, line_img = cv2.threshold(line_img, lower_thresh, 255, cv2.THRESH_BINARY)

    # Convert the binary image back to RGB
    print('converting back to rgb')
    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)

    # Define erosion kernel
    # kernel_size = 12
    kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)

    # Perform erosion
    print('doing erosion')
    line_img = cv2.erode(line_img, kernel, iterations=1)

    # FOR FASTER PROCESSING, NOTE HALF THE ORIGINAL SIZE
    # shrink_factor = 6
    line_img = cv2.resize(line_img, None, fx=1 / shrink_factor, fy=1 / shrink_factor, interpolation=cv2.INTER_LINEAR)

    return line_img
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
def return_hash_colors(ocr_results):
    # convert the line name into a color and fold it into the line object
    colors = []
    for result in ocr_results:
        result_id = result[1]
        print(result_id)
        hash_color = generate_color(result_id)
        colors.append(hash_color)
    return colors
def generate_painted_img(processed_img, ocr_results, colors, shrink_factor, expand_scale=1):
    # re_pattern = r'.*\"-[A-Z]{2,3}-[A-Z\d]{3,5}-.*'
    # re_pattern = r'.*\"-[A-Z]{1,5}-[A-Z\d]{3,5}-.*'
    #rocessed_service_img, lines, lines_with_service_in, shrink_factor,

    scale_results = copy.deepcopy(ocr_results)

    # also note that easy ocr gives 4 points as a box. So line[0] = p1, p2, p3, p4 where each point is a [x,y] pair
    # eg. line[0] = [[17, 5], [71, 5], [71, 17], [17, 17]]
    # have the boxes change size to reflect, eventually we might be able to skip this step
    for result in scale_results:
        result[0] = [[int(x / shrink_factor), int(y / shrink_factor)] for x, y in result[0]]
        # print("after "+str(line[0]))

    # we put the seed pixel locations in
    # note, x1, y1, x2, y2 = line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]
    seed_pixel_boxes = []
    for result, color in zip(scale_results, colors):
        pixel_box = [result[0], color]  # [box, color]
        seed_pixel_boxes.append(pixel_box)

    # print(seed_pixels)
    print("doing flood fill evenly")
    # generate the colored line image
    colored_img = flood_fill_evenly(seed_pixel_boxes, processed_img, expand_scale)
    return colored_img
def return_service_colors(services, line_img, img_scale):
    # services = [[text, label, box]...]
    service_colors = []
    for service in services:
        text, lb, box = service
        service_color = region_mode_color(box, line_img, img_scale)
        service_colors.append(service_color)

    return service_colors
def region_mode_color(box, img, img_scale,):
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
def generate_color(string):
    # Generate a hash value for the input string using MD5
    hash_value = hashlib.md5(string.encode('utf-8')).hexdigest()

    # Take the first six characters of the hash value
    hex_code = hash_value[:6]

    # Convert the hex code to RGB values
    r, g, b = int(hex_code[:2], 16), int(hex_code[2:4], 16), int(hex_code[4:], 16)

    # Return the RGB values as a tuple
    return (r, g, b)

def generate_equipment(processed_img, ocr_results, re_pattern, shrink_factor,
                       detecto_equipment_boxes, equipment_box_expand, include_inside):
    #generate_equipment(line_img, shrink_factor, re_pattern, detecto_equipment_boxes, equipment_box_expand, include_inside, ocr_results):
    #generate_equipment(processed_img, ocr_results_with_rotated_text, re_equipment, shrink_factor,
    #                   equipment_boxes, equipment_box_expand, include_inside)
    # img=img_no_lines
    # equipment_img = copy.copy(img)
    # get equipment text
    # assign it a hash color
    # find detecto boxes that intersect
    # use those boxes with that color
    # use equipment b
    # re_pattern = r'^[A-Z]{1,2}\d{5}-.*'
    ocr_equipment = get_text_in_ocr_results(re_pattern, ocr_results)  # where equipment = [box, name, score]

    # convert the line name into a color
    for equipment in ocr_equipment:
        print('looping through ocr equipment')
        equipment_id = equipment[1]

        # print(equipment_id)
        hashcolor = generate_color(equipment_id)
        equipment.append(hashcolor)

        # APPEND comment here?     #box is a detecto box
        dbox = convert_ocr_box_to_detecto_box(equipment[0])
        equipment_text_comment = get_comment(ocr_results, dbox, 20, include_inside=False)
        equipment.append(equipment_text_comment)



        #share

        print(equipment)
        # [[(2671, 106), (2863, 106), (2863, 148), (2671, 148)], 'PP-1130', 0.9728249629949269, (123, 187, 216), 'PP-1130~PS1 PIPELINE PUMPS~']
    # detecto equipment boxes, ocr equipment, extension range
    # now OCR equipment will have a extra comment appended

    share_equipment_comments(ocr_equipment)

    #make_detecto_equipment_boxes_black(processed_img,detecto_equipment_boxes)

    ocr_equipment = update_ocr_box(detecto_equipment_boxes, ocr_equipment, equipment_box_expand)

    #add_overlapping boxes to ocr equipment the idea is to merge certain boxes like if you have L shaped tank
    # but it might be broken because some equipment doens'
    #add_overlapping_boxes(ocr_equipment, detecto_equipment_boxes)

    # should we make a shrink coppy?
    for equipment in ocr_equipment:
        print('equipment debug', equipment)
        equipment[0] = [[int(x / shrink_factor), int(y / shrink_factor)] for x, y in equipment[0]]
    # we put the seed pixel locations in
    # note, x1, y1, x2, y2 = line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]
    seed_pixel_boxes = []
    for equipment in ocr_equipment:
        pixel_box = [equipment[0], equipment[3]]  # derived from easyocr [box, label, score, color]
        seed_pixel_boxes.append(pixel_box)

    # print(seed_pixels)
    print("doing flood fill evenly")
    # generate the colored line image
    equipment_img = flood_fill_evenly(seed_pixel_boxes, processed_img, 1)

    # line_img = cv2.resize(line_img, None, fx=shrink_factor, fy=shrink_factor, interpolation=cv2.INTER_LINEAR)

    return (ocr_equipment, equipment_img)
def return_serviced_lines(lns, lcolors, srvs, scolors, option):
    lines_with_service = []
    lines_with_service_color = []
    lines_with_service_txt = []
    services = []
    #note list index is being utilized
    for ln, lcolor in zip(lns, lcolors):
        for srv, scolor in zip(srvs, scolors):
            print(srv)
            stxt, slabel, sbox = srv
            if option == slabel:#service_in or service_out
                print(f'lcolor {lcolor}, scolor {scolor}')
                if lcolor == tuple(scolor):
                    #service = stxt
                    #ln[1] = stxt
                    lines_with_service.append(ln) # jsut for painting image
                    lines_with_service_color.append(lcolor)
                    lines_with_service_txt.append(stxt)
                    break

    return lines_with_service, lines_with_service_color, lines_with_service_txt

def flood_fill_evenly(seed_pixel_boxes, line_img, expand_scale):
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
def convert_ocr_box_to_detecto_box(obox):
    dbox = np.zeros(4)  # Initialize a NumPy array of size 4 with zeros
    dbox[0] = obox[0][0]
    dbox[1] = obox[0][1]
    dbox[2] = obox[2][0]
    dbox[3] = obox[2][1]
    return dbox
def share_equipment_comments(ocr_equipment):
    seen = {}

    for equipment in ocr_equipment:
        label = equipment[1]
        comment = equipment[4]

        if label not in seen:
            seen[label] = comment
        else:
            seen[label] += '~' + comment

    for equipment in ocr_equipment:
        label = equipment[1]
        if label in seen:
            equipment[4] = seen[label]
def update_ocr_box(detecto_equipment_boxes, ocr_equipment, expand_amount):
    # data format of detecto box: x1d = int(box[0])    y1d = int(box[1])    x2d = int(box[2])    y2d = int(box[3])
    # data format of ocr box: x1o, y1o, x2o, y2o = ocr_equipment[0][0][0], ocr_equipment[0][0][1], ocr_equipment[0][2][0], ocr_equipment[0][2][1]
    # if a detecto box when expanded touches a ocr box, replace that ocr box with the detecto box
    def expand(dbox, amount):
        expanded_box = []
        expanded_box += [int(dbox[0] - amount)]  # x1
        expanded_box += [int(dbox[1] - amount)]  # y1
        expanded_box += [int(dbox[2] + amount)]  # x2
        expanded_box += [int(dbox[3] + amount)]  # y2
        return expanded_box

    def overlap(dbox, obox):
        x1d, y1d, x2d, y2d = dbox
        x1o, y1o, x2o, y2o = obox[0][0], obox[0][1], obox[2][0], obox[2][1]
        # Check for x-axis overlap
        x_overlap = (x1d <= x2o) and (x2d >= x1o)
        # Check for y-axis overlap
        y_overlap = (y1d <= y2o) and (y2d >= y1o)
        # If there is overlap on both axes, the boxes overlap
        if x_overlap and y_overlap:
            return True
        else:
            return False

    def shift_box(dbox, obox):
        print(f'shifting {dbox} to {obox}')
        x1d, y1d, x2d, y2d = dbox
        # Update each point in obox with the new coordinates
        obox[0] = [x1d, y1d]
        obox[1] = [x2d, y1d]
        obox[2] = [x2d, y2d]
        obox[3] = [x1d, y2d]

        return obox

    # Initialize a list to keep track of detecto boxes that have been matched
    matched_detecto_boxes = []
    matched_ocr_boxes = []
    for i, dbox in enumerate(detecto_equipment_boxes):
        expanded_dbox = expand(dbox, expand_amount)
        # matched = False

        for n, equipment in enumerate(ocr_equipment):
            if overlap(expanded_dbox, equipment[0]):
                equipment[0] = shift_box(dbox, equipment[0])
                matched_detecto_boxes.append(i)
                matched_ocr_boxes.append(n)
                # matched = True
                break


    print(f'all ocr equip boxes: {ocr_equipment}')
    #remove unmatched ocr boxes
    ocr_equipment = [ocr for i, ocr in enumerate(ocr_equipment) if i in matched_ocr_boxes]
    print(f'trimed ocr equip boxes: {ocr_equipment}')

    # Iterate over unmatched detecto boxes and append them as new OCR equipment
    extra_ocr_equipment = []
    for i, dbox in enumerate(detecto_equipment_boxes):
        if i not in matched_detecto_boxes:  # for unmatched equipment boxes
            for equipment in ocr_equipment:  # if one of the ocr_equipment (since boxes for some have been grown) overlaps
                if overlap(dbox, equipment[0]):
                    equip2 = equipment.copy()  # make a copy of the ocr equipment
                    equip2[0] = shift_box(dbox, equipment[0])  # change the box position
                    extra_ocr_equipment.append(equip2)  # append it to this list

    ocr_equipment += extra_ocr_equipment  # merege lists

    return ocr_equipment
def figure_out_if_instrument_has_sis(data, img, shrink_factor):
    # box, tag, tag_no, label, line_id, comment, valve_type, valve_size, inst_alarm, sis = data
    # page, pid, box, tag, tag_no, label, line_id, comment, valve_type, valve_size, inst_alarm
    # print(data)

    for x in data:
        if x['tag'] == 'SIS':
            tag_no = x['tag_no']
            print(f'found a sis {tag_no}')
            start = get_box_center(x['box'])
            img_copy = copy.copy(img)
            flood_fill_till_inst_reached(start, tag_no, data, img_copy, shrink_factor)
def get_box_center(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y
def flood_fill_till_inst_reached(start, tag_no, data, img, shrink_factor):
    # do we need to make a copy of each image?

    start_time = time.time()
    # Create a new image to perform the flood fill on
    # img = copy.deepcopy(line_img)
    # Get the size of the image
    color = (0, 255, 0)
    height, width, channels = img.shape

    start = [int(value / shrink_factor) for value in start]
    # Make a corresponding list or queue and put the initial pixel in it
    active_pixel_list = []
    for pixel in [start]:
        index = []
        index.append(pixel)
        # Add the initial pixel to the active pixel list
        active_pixel_list.append(index)
        x, y = pixel
        print("x:", x, "y:", y)
        # Change the colors on the filled image to the seed colors
        img[y, x] = color

    # While there is still an element in any of the lists:
    # An empty list returns false
    while active_pixel_list:

        if time.time() - start_time > 2:
            print("Timeout reached.")
            return False

            # Enumerate gives us an index n
        for n, pixel_stack in enumerate(active_pixel_list):
            # print("chekcing stack "+str(n))
            start_size = len(pixel_stack)
            # i goes from the bottom to the top
            for i in range(start_size):
                pixel = active_pixel_list[n].pop(0)
                x, y = pixel

                # if curr_x, curr_y in a region break the loop
                for element in data:
                    if element['label'] == 'inst':
                        # shirnk box
                        # print('inst')
                        box = [int(value / shrink_factor) for value in element['box']]
                        if point_in_box(x, y, box):
                            # print(element[9])
                            element['sis'] = element['sis'] + f'{tag_no} '
                            print('found a sis inst pair!')
                            # assuming data is mutable
                            return True

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
                            active_pixel_list[n].append([nx, ny])
                            # Set color on filled image
                            img[ny, nx] = color

        # Remove empty sublists
        active_pixel_list = [x for x in active_pixel_list if x]

    # Return the filled image
    # return img
def point_in_box(x, y, box):
    xmin, ymin, xmax, ymax = box
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return True
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

