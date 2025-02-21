import cv2
import time
from collections import defaultdict

def HardOCR(image, reader, reader_settings,
            sub_img_size=1300, stride=1250):
    rs = reader_settings

    current_time1 = time.time()
    ocr_results = perform_ocr_on_subimages(image, reader, rs, sub_img_size=sub_img_size, stride=stride)
    current_time2 = time.time()
    print(f'OCR time elapsed: {current_time2 - current_time1}')

    current_time1 = time.time()
    stitched_results = revaluate_overlapping_regions(ocr_results, reader, image, reader_settings)
    current_time2 = time.time()
    print(f'Stitching time elapsed: {current_time2 - current_time1}')

    current_time1 = time.time()
    final_results = correct_for_rotated_text(image, stitched_results, reader, rs)
    current_time2 = time.time()
    print(f'Rotation correction time elapsed: {current_time2 - current_time1}')

    return final_results


def split_image(img, sub_img_size, stride):
    # Get the size of the input image
    h, w = img.shape[:2]

    # Loop through the image and extract sub-images
    sub_images = []
    offsets = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Calculate actual size of this sub-image
            actual_h = min(sub_img_size, h - y)
            actual_w = min(sub_img_size, w - x)

            # Only process sub-image if it has meaningful size
            if actual_h > 0 and actual_w > 0:
                sub_img = img[y:y + actual_h, x:x + actual_w]
                sub_images.append(sub_img)
                offsets.append((x, y))

    return sub_images, offsets


def perform_ocr_on_subimages(img, reader, rs, sub_img_size=1300, stride=1250):
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
            adjusted_results.append(result)

        ocr_results.extend(adjusted_results)

    return ocr_results


def adjust_box_coordinates_with_offset(box, offset):
    return [
        (box[0][0] + offset[0], box[0][1] + offset[1]),
        (box[1][0] + offset[0], box[1][1] + offset[1]),
        (box[2][0] + offset[0], box[2][1] + offset[1]),
        (box[3][0] + offset[0], box[3][1] + offset[1])
    ]


def correct_for_rotated_text(img, results, reader, rs):
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


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""

    # Convert boxes to x1,y1,x2,y2 format
    def get_bounds(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    box1_x1, box1_y1, box1_x2, box1_y2 = get_bounds(box1)
    box2_x1, box2_y1, box2_x2, box2_y2 = get_bounds(box2)

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def get_combined_box(box1, box2):
    """Get a bounding box that contains both input boxes."""
    xs = [p[0] for p in box1 + box2]
    ys = [p[1] for p in box1 + box2]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def revaluate_overlapping_regions(results, reader, img, rs, min_overlap_threshold=0.01, max_overlap_threshold=0.98):
    """
    Revaluate overlapping regions in OCR results.

    Args:
        results: List of OCR results, each containing [box, text, confidence]
        reader: EasyOCR reader instance
        img: Original image
        rs: Reader settings
        min_overlap_threshold: Minimum IoU to consider boxes as overlapping
        max_overlap_threshold: Threshold above which to keep only highest confidence

    Returns:
        List of processed OCR results with overlaps handled
    """
    if not results:
        return []

    # Create groups of overlapping boxes
    groups = []
    used_indices = set()

    for i in range(len(results)):
        if i in used_indices:
            continue

        current_group = [i]
        box1 = results[i][0]

        for j in range(i + 1, len(results)):
            if j in used_indices:
                continue

            box2 = results[j][0]
            iou = calculate_iou(box1, box2)

            if iou > min_overlap_threshold:
                current_group.append(j)
                used_indices.add(j)

        if len(current_group) > 1:  # Only add groups with overlaps
            groups.append(current_group)
            used_indices.add(i)

    # Process each group
    final_results = []
    processed_indices = set()

    # First, add all non-overlapping results
    for i in range(len(results)):
        if i not in used_indices:
            final_results.append(results[i])

    # Process overlapping groups
    for group in groups:
        group_results = [results[i] for i in group]

        # Check if any pair has high overlap
        high_overlap = False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                iou = calculate_iou(group_results[i][0], group_results[j][0])
                if iou > max_overlap_threshold:
                    high_overlap = True
                    break
            if high_overlap:
                break

        if high_overlap:
            # Keep result with highest confidence
            best_result = max(group_results, key=lambda x: x[2])
            final_results.append(best_result)
        else:
            # Get combined box for all results in group
            combined_box = group_results[0][0]
            for result in group_results[1:]:
                combined_box = get_combined_box(combined_box, result[0])

            # Extract region from image
            x1 = int(min(p[0] for p in combined_box))
            y1 = int(min(p[1] for p in combined_box))
            x2 = int(max(p[0] for p in combined_box))
            y2 = int(max(p[1] for p in combined_box))

            region = img[y1:y2, x1:x2]

            # Perform OCR on combined region
            new_results = reader.readtext(region, **rs)

            # Adjust coordinates
            for result in new_results:
                box = result[0]
                adjusted_box = adjust_box_coordinates_with_offset(box, (x1, y1))
                result = list(result)
                result[0] = adjusted_box
                final_results.append(result)

    return final_results
