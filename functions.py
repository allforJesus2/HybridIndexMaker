import torch
import cv2
import os
import fitz
import re
import numpy as np
import os
from PyPDF2 import PdfMerger

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

def merge_common_substring_with_single_chars(str1, str2):
    # Step 1: Find the longest common substring (match_text)
    match_text = ""
    for i in range(len(str1)):
        for j in range(i + len(match_text), len(str1) + 1):
            if str1[i:j] in str2:
                match_text = str1[i:j]
            else:
                break

    # Step 2: Remove match_text from both strings
    str1 = str1.replace(match_text, '')
    str2 = str2.replace(match_text, '')

    # Step 3: Split the remaining text on whitespace and '/'
    str1_words = re.split(r'\s|/', str1)
    str2_words = re.split(r'\s|/', str2)

    # Step 4: Check for any single characters and join them
    single_chars = []
    for word in str1_words + str2_words:
        if len(word) == 1:
            single_chars.append(word)
    single_chars_str = '/'.join(single_chars)

    # Step 5: Append single_chars_str to match_text and return
    return match_text + single_chars_str

def pdf2png(pdf_file, dpi):
    doc = fitz.open(pdf_file)

    # Create a new folder to store the PNG images
    folder_name = os.path.join(os.path.dirname(pdf_file), os.path.splitext(os.path.basename(pdf_file))[0] + "_images")
    os.makedirs(folder_name, exist_ok=True)

    # Iterate through the pages and save them as PNG images
    for page in doc:
        # Render the page as a pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        #width, height = page.rect.width * dpi / 72, page.rect.height * dpi / 72

        # Save the pixmap as a PNG image in the new folder
        png_file = os.path.join(folder_name, f"page_{page.number + 1}.png")
        pix.save(png_file)
        print(png_file)
def model_predict_on_mozaic(img, model, square_size=1300, stride=1200):
    sub_images, offsets = split_image(img, square_size, stride)
    print("length of sub_images " + str(len(sub_images)))


    all_labels = []
    all_boxes = []
    all_scores = []

    for i, image in enumerate(sub_images):
        sub_labels, sub_boxes, sub_scores = model.predict(image)
        print(f"image {i} : {sub_labels}")

        for box in sub_boxes:
            box[0] = box[0] + offsets[i][0]  # x1
            box[1] = box[1] + offsets[i][1]  # y1
            box[2] = box[2] + offsets[i][0]  # x2
            box[3] = box[3] + offsets[i][1]  # y2


        all_labels.append(sub_labels)
        all_boxes.append(sub_boxes)
        all_scores.append(sub_scores)

    labels, boxes, scores = merge(all_labels, all_boxes, all_scores)
    labels, boxes, scores = remove_overlapping_boxes(labels, boxes, scores, 0.5)
    return labels, boxes, scores
def split_image(img, sub_img_size=1300, stride=1200):
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
            # lot_pic(sub_img,5)
            offsets.append((x, y))

    return sub_images, offsets
def merge(all_labels, all_boxes, all_scores):
    labels = []
    boxes = []
    scores = []

    for i in range(len(all_labels)):
        sub_labels = all_labels[i]
        sub_boxes = all_boxes[i]
        sub_scores = all_scores[i]

        for j in range(len(sub_labels)):
            labels.append(sub_labels[j])
            boxes.append(sub_boxes[j].tolist())
            scores.append(sub_scores[j])

    boxes = torch.tensor(boxes, dtype=torch.float32)

    return labels, boxes, torch.tensor(scores, dtype=torch.float32)
def remove_overlapping_boxes(labels, boxes, scores, threshold):
    to_remove = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            overlap_ratio = overlap(boxes[i], boxes[j])
            if overlap_ratio > threshold:
                # print("comparing "+labels[j]+":"+str(scores[j])+" to "+labels[i]+":"+str(scores[i]))
                if scores[i] > scores[j]:
                    to_remove.append(j)
                    # print("need to remove "+str(labels[j])+" at "+str(boxes[j])+" with score "+str(scores[j]))
                else:
                    to_remove.append(i)
                    # print("need to remove "+str(labels[i])+" at "+str(boxes[i])+" with score "+str(scores[i]))
    labels = [label for i, label in enumerate(labels) if i not in to_remove]
    boxes = [box for i, box in enumerate(boxes) if i not in to_remove]
    scores = [score for i, score in enumerate(scores) if i not in to_remove]
    return labels, boxes, scores

def overlap(box1, box2):
    # box1 is the text
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    x_left = max(x1a, x1b)
    y_top = max(y1a, y1b)
    x_right = min(x2a, x2b)
    y_bottom = min(y2a, y2b)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # if box1 is completely within box2

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2a - x1a) * (y2a - y1a)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - intersection_area
    overlap_ratio = intersection_area / union_area

    return overlap_ratio

def return_inst_data(prediction_data, img, clip, reader, minscore, correct_fn):
    all_data = []
    for label, box, score in prediction_data:
        if (label == 'inst' or label == 'dcs') and score > minscore:
            tag, tag_no = '', ''
            x_min, y_min, x_max, y_max = box.tolist()

            # Crop the image to the box region
            crop_img = img[int(y_min) + clip:int(y_max) - clip, int(x_min) + clip:int(x_max) - clip]


            # enhance
            crop_img = cv2.resize(crop_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            try:
                # Perform OCR on the cropped image using EasyOCR
                #result = reader.readtext(crop_img)#, min_size=10, low_text=0.4, link_threshold=.4,
                #                         text_threshold=0.3, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()')

                result = reader.readtext(crop_img, min_size=10, low_text=0.3, link_threshold=0.2,
                                         text_threshold=0.3, width_ths=6.0, decoder='beamsearch',
                                         allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()')


            except Exception as e:
                print('error',e)
                continue

            # Extract the recognized text from the EasyOCR result
            # pdb.set_trace()
            if result:
                tag = result[0][1]
                tag_no = ' '.join([box[1] for box in result[1:]])

                ''' SAVE LITTLE BOXES
                dest_folder = 'inst_crop_folder'
                os.makedirs(dest_folder, exist_ok=True)
                filename = tag+"_"+tag_no+'.png'
                dest_file_path = os.path.join(dest_folder, filename)
                cv2.imwrite(dest_file_path, crop_img)
                '''

                if correct_fn:
                    print('doing correct')
                    tag, tag_no = correct_fn(tag, tag_no)
            else:
                continue

            data = {'tag':tag, 'tag_no':tag_no, 'label':label}
            all_data.append(data)

    return all_data
    # tag, tag_no, label, line_id, xv_type, valve_type, size, inst_alarm

