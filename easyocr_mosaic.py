import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def join_strings(str1, str2):
    # Find the length of the shorter string
    min_length = min(len(str1), len(str2))

    # Start with the full length of the shorter string
    overlap = min_length

    # Find the largest overlap
    while overlap > 0:
        if str1[-overlap:] == str2[:overlap]:
            return str1 + str2[overlap:]
        overlap -= 1

    # If no overlap is found, simply concatenate the strings
    return str1 + str2
def stitch_words(ocr_results):
    # Sort results by y-coordinate (top to bottom)
    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])

    # Group results by approximate y-coordinate (same line)
    line_threshold = 10  # Adjust based on your image resolution
    lines = defaultdict(list)
    for result in sorted_results:
        y_coord = result[0][0][1]
        line_key = y_coord // line_threshold
        lines[line_key].append(result)

    stitched_results = []
    for line in lines.values():
        # Sort words in the line by x-coordinate (left to right)
        line.sort(key=lambda x: x[0][0][0])

        # Stitch words in the line
        stitched_line = []
        for i, word in enumerate(line):
            if i == 0:
                stitched_line.append(word)
                continue

            prev_word = stitched_line[-1]
            if should_stitch(prev_word, word):
                stitched_word = stitch_two_words(prev_word, word)
                stitched_line[-1] = stitched_word
            else:
                stitched_line.append(word)

        stitched_results.extend(stitched_line)

    return stitched_results


def should_stitch(word1, word2):
    # Check if the words are close enough horizontally
    x1_max = max(coord[0] for coord in word1[0])
    x2_min = min(coord[0] for coord in word2[0])
    distance = x2_min - x1_max

    # Adjust this threshold based on your image resolution and font size
    distance_threshold = 20

    return distance <= distance_threshold


def stitch_two_words(word1, word2):
    # Combine the bounding boxes
    new_box = [
        word1[0][0],  # Top-left
        word2[0][1],  # Top-right
        word2[0][2],  # Bottom-right
        word1[0][3],  # Bottom-left
    ]

    # Combine the text
    new_text = join_strings(word1[1], word2[1])

    # Combine the confidence scores (you may want to adjust this logic)
    new_confidence = (word1[2] + word2[2]) / 2

    return (new_box, new_text, new_confidence)


def stitch_text(text1, text2):
    # Find the overlap
    min_overlap = 3  # Minimum characters to consider for overlap
    max_overlap = min(len(text1), len(text2))

    for i in range(max_overlap, min_overlap - 1, -1):
        if text1[-i:].lower() == text2[:i].lower():
            return text1 + text2[i:]

    # If no significant overlap found, just concatenate with a space
    return text1 + " " + text2


# Modify the main HardOCR function to use the new stitching algorithm
def HardOCR(image, reader, reader_settings,
            sub_img_size=1300, stride=1250):
    rs = reader_settings

    current_time1 = time.time()
    ocr_results = perform_ocr_on_subimages(image, reader, rs, sub_img_size=sub_img_size, stride=stride)
    current_time2 = time.time()
    print(f'OCR time elapsed: {current_time2 - current_time1}')

    current_time1 = time.time()
    stitched_results = stitch_words(ocr_results)
    current_time2 = time.time()
    print(f'Stitching time elapsed: {current_time2 - current_time1}')

    current_time1 = time.time()
    final_results = correct_for_rotated_text2(image, stitched_results, reader, rs)
    current_time2 = time.time()
    print(f'Rotation correction time elapsed: {current_time2 - current_time1}')

    return final_results

# START PARALLELE STUFF
def perform_ocr_on_subimages_parallel(img, reader, rs,
                                      sub_img_size=1300, stride=1250):
    sub_images, offsets = split_image(img, sub_img_size, stride)

    with ThreadPoolExecutor() as executor:
        future_to_subimg = {executor.submit(process_subimage, sub_img, offset, reader, rs): i
                            for i, (sub_img, offset) in enumerate(zip(sub_images, offsets))}

        ocr_results = []
        for future in as_completed(future_to_subimg):
            i = future_to_subimg[future]
            print(f'\rprocessing subimage {i + 1} of {len(sub_images)}', end='')
            ocr_results.extend(future.result())

    return ocr_results


def process_subimage(sub_img, offset, reader, rs):
    sub_results = reader.readtext(sub_img, **rs)
    adjusted_results = []
    for result in sub_results:
        box = result[0]
        adjusted_box = adjust_box_coordinates_with_offset(box, offset)
        result = list(result)
        result[0] = adjusted_box
        adjusted_results.append(result)
    return adjusted_results


def process_overlapping_boxes_parallel(img, ocr_results, reader):
    overlapping_groups, non_overlapping_results = find_overlapping_boxes(ocr_results)

    with ThreadPoolExecutor() as executor:
        future_to_group = {executor.submit(process_group, img, group, reader): i
                           for i, group in enumerate(overlapping_groups)}

        for future in as_completed(future_to_group):
            i = future_to_group[future]
            print(f'\rprocessing overlap group {i + 1} of {len(overlapping_groups)}', end='')
            non_overlapping_results.extend(future.result())

    return non_overlapping_results


def process_group(img, group, reader):
    roi_coords = extract_roi_from_group(group)
    roi_ocr_results = ocr_on_roi(img, roi_coords, reader)
    roi_offset = roi_coords[0]
    return adjust_roi_results_offset(roi_ocr_results, roi_offset)


def correct_for_rotated_text2_parallel(img, results, reader, rs):
    with ThreadPoolExecutor() as executor:
        future_to_result = {executor.submit(correct_result, img, result, reader, rs): i
                            for i, result in enumerate(results)}

        corrected_results = []
        for future in as_completed(future_to_result):
            corrected_results.append(future.result())

    return corrected_results


def correct_result(img, result, reader, rs):
    result_list = list(result)
    box = result[0]
    x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
    if abs(y1 - y2) > abs(x1 - x2):
        try:
            crop_img = img[y1:y2, x1:x2]
            rotated_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
            sub_results = reader.readtext(rotated_img, **rs)
            result_list[1] = sub_results[0][1]
        except:
            print(f'*** no text detected couldnt rotate {sub_results}')
    return result_list

# END PARALLEL STUFF

def adjust_box_coordinates_with_offset(box, offset):
    return [
        (box[0][0] + offset[0], box[0][1] + offset[1]),
        (box[1][0] + offset[0], box[1][1] + offset[1]),
        (box[2][0] + offset[0], box[2][1] + offset[1]),
        (box[3][0] + offset[0], box[3][1] + offset[1])
    ]
def split_image(img, sub_img_size, stride):
    # Get the size of the input image
    h, w = img.shape[:2]

    # Loop through the image and extract sub-images
    sub_images = []
    offsets = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            sub_img = img[y:y + sub_img_size, x:x + sub_img_size]

            # Pad the sub-image if necessary
            pad_y = sub_img_size - sub_img.shape[0]
            pad_x = sub_img_size - sub_img.shape[1]
            if pad_y > 0 or pad_x > 0:
                sub_img = cv2.copyMakeBorder(sub_img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            sub_images.append(sub_img)
            offsets.append((x, y))

    return sub_images, offsets


def perform_ocr_on_subimages(img, reader, rs,
                             sub_img_size=1300, stride=1250):
    # Load the image using OpenCV
    # img = cv2.imread(img_path)

    # Split the image into sub-images
    sub_images, offsets = split_image(img, sub_img_size, stride)

    # Perform OCR on each sub-image using EasyOCR
    ocr_results = []
    for i, (sub_img, offset) in enumerate(zip(sub_images, offsets)):
        print('\r', f'performing ocr on image {i} of {len(sub_images)}', end='')
        sub_results = reader.readtext(sub_img, **rs)
        # Adjust the box coordinates with the offset
        adjusted_results = []
        for result in sub_results:
            box = result[0]
            adjusted_box = adjust_box_coordinates_with_offset(box, offset)
            result = list(result)
            result[0] = adjusted_box
            adjusted_results.append(result)  # Appending adjusted box and text

        ocr_results.extend(adjusted_results)

    return ocr_results


def create_annotated_image(img_path, results):
    # Load the original image
    img = cv2.imread(img_path)

    # Draw bounding boxes and text annotations on the image based on OCR results
    for result in results:
        box = result[0]
        text = result[1]
        box = convert_to_ints(box)
        cv2.polylines(img, [np.array(box)], True, (0, 255, 0), 2)
        cv2.putText(img, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1, cv2.LINE_AA)

    # Return the annotated image
    return img


def convert_to_ints(data):
    result = [(int(item[0]), int(item[1])) for item in data]
    return result


def check_overlap(box1, box2):
    # Check if two boxes overlap by comparing their coordinates
    x1_min, y1_min = min(box1[0][0], box1[1][0], box1[2][0], box1[3][0]), min(box1[0][1], box1[1][1], box1[2][1],
                                                                              box1[3][1])
    x1_max, y1_max = max(box1[0][0], box1[1][0], box1[2][0], box1[3][0]), max(box1[0][1], box1[1][1], box1[2][1],
                                                                              box1[3][1])
    x2_min, y2_min = min(box2[0][0], box2[1][0], box2[2][0], box2[3][0]), min(box2[0][1], box2[1][1], box2[2][1],
                                                                              box2[3][1])
    x2_max, y2_max = max(box2[0][0], box2[1][0], box2[2][0], box2[3][0]), max(box2[0][1], box2[1][1], box2[2][1],
                                                                              box2[3][1])

    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def find_overlapping_boxes(data):
    groups = []

    while len(data) > 0:
        current_group = []
        current_box = data.pop(0)
        current_group.append(current_box)

        i = 0
        while i < len(data):
            if check_overlap(current_box[0], data[i][0]):
                current_group.append(data.pop(i))
                i -= 1
            i += 1

        groups.append(current_group)

    data_non_overlapping = []
    overlapping_groups = []

    for group in groups:
        if len(group) == 1:
            data_non_overlapping.append(group[0])
        else:
            overlapping_groups.append(group)

    return overlapping_groups, data_non_overlapping


def extract_roi_from_group(group):
    # for e in group:
    #    print(e[1])
    # Get the bounding box coordinates for the group of overlapping boxes
    x_coords = [box[0][0][0] for box in group]
    y_coords = [box[0][0][1] for box in group]
    min_x = min(x_coords)
    min_y = min(y_coords)
    # print(box[1][2][0])
    # print(group)
    max_x = max([box[0][2][0] for box in group])
    max_y = max([box[0][2][1] for box in group])

    # Extract the region of interest (ROI) based on the coordinates
    roi = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    return roi


def ocr_on_roi(img, roi, reader):
    # Extract the ROI from the image
    x_min, y_min = roi[0]
    x_max, y_max = roi[2]
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    roi_img = img[y_min:y_max, x_min:x_max]
    # display_roi_image(roi_img)
    # Perform OCR on the ROI using EasyOCR
    roi_results = reader.readtext(roi_img)

    return roi_results


def adjust_roi_results_offset(roi_results, roi_offset):
    # Adjust the box coordinates of the ROI OCR results with the ROI offset
    adjusted_results = []
    for result in roi_results:
        box = result[0]
        adjusted_box = adjust_box_coordinates_with_offset(box, roi_offset)
        result = list(result)
        result[0] = adjusted_box
        adjusted_results.append(result)  # Appending adjusted box and text

    return adjusted_results


def process_overlapping_boxes(img, ocr_results, reader):
    # img = cv2.imread(img_path)
    overlapping_groups, non_overlapping_results = find_overlapping_boxes(ocr_results)
    length = len(overlapping_groups)
    # Process each group of overlapping boxes
    for i, group in enumerate(overlapping_groups):
        print('\r', f'processing overlap group {i} of {length}',end='')
        # Extract ROI coordinates from the group of overlapping boxes
        roi_coords = extract_roi_from_group(group)

        # Perform OCR on the ROI
        roi_ocr_results = ocr_on_roi(img, roi_coords, reader)

        # Adjust ROI OCR results' offsets
        roi_offset = roi_coords[0]  # Offset for adjusting ROI OCR results
        adjusted_roi_results = adjust_roi_results_offset(roi_ocr_results, roi_offset)

        # Extend the results with adjusted ROI OCR results
        non_overlapping_results.extend(adjusted_roi_results)

    return non_overlapping_results


# Function to display ROI image
def display_roi_image(roi_img):
    # Convert BGR image to RGB for displaying with Matplotlib
    roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

    # Display the ROI image using Matplotlib
    plt.imshow(roi_img_rgb)
    plt.axis('off')  # Turn off axis labels and ticks
    plt.title('Region of Interest (ROI) Image')
    plt.show()


def correct_for_rotated_text2(img, results, reader, rs):
    # img = cv2.imread(img)
    corrected_results = []
    for result in results:
        # we need a list because we might want to overwrite
        result_list = list(result)
        # Get the bounding box
        # print(result)
        box = result[0]
        x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
        # if a vertical box
        if abs(y1 - y2) > abs(x1 - x2):
            try:
                crop_img = img[y1:y2, x1:x2]
                # here we want to rotate the text
                rotated_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
                # re-read
                sub_results = reader.readtext(rotated_img, **rs)
                # overwrite text

                result_list[1] = sub_results[0][1]
            except:
                print(f'*** no text detected couldnt rotate {sub_results}')
                continue
        corrected_results.append(result_list)
    return corrected_results


