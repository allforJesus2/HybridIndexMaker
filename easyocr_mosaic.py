import cv2
import time
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


