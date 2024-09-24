import cv2
import os
import torch

def model_predict_on_mosaic(img, model, square_size=1300, stride=1200, yolo=False, label_names=None, threshold=0.5):
    sub_images, offsets = split_image(img, square_size, stride)
    print("length of sub_images " + str(len(sub_images)))

    all_labels = []
    all_boxes = []
    all_scores = []

    for i, image in enumerate(sub_images):
        if yolo == False:
            sub_labels, sub_boxes, sub_scores = model.predict(image)
        else:
            results = model.predict(image)
            sub_labels, sub_boxes, sub_scores = convert_results(results[0].boxes, label_names)

        # sub_boxes (Tensor): A tensor of shape (N, 4) representing the bounding boxes.
        # sub_scores (Tensor): A tensor of shape (N,) representing the confidence scores.
        # sub_labels a list of strings that match up with a box and score.
        # For example sub_labels[N] (i.e. sub_labels[5] = 'Cat'), sub_boxes[N], and sub_scores[N] go together to define one object
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
    print('beginning to remove overlapping boxes')
    labels, boxes, scores = remove_overlapping_boxes(labels, boxes, scores, threshold)
    print('finished  removing overlapss')



    return labels, boxes, scores

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

def split_image(img, sub_img_size=1300, stride=1200):
    # Get the size of the input image
    h, w = img.shape[:2]

    # If the image size is smaller than sub_img_size, return the image itself
    if h <= sub_img_size and w <= sub_img_size:
        pad_y = max(sub_img_size - h, 0)
        pad_x = max(sub_img_size - w, 0)
        img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return [img], [(0, 0)]


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

def remove_overlapping_boxes(labels, boxes, scores, threshold):
    """
    Apply Non-Maximum Suppression (NMS) to the bounding boxes using fully vectorized operations.

    Args:
        labels (list): List of labels corresponding to the bounding boxes.
        boxes (Tensor): Tensor of shape (N, 4) representing the bounding boxes.
        scores (Tensor): Tensor of shape (N,) representing the confidence scores.
        threshold (float): Overlap threshold for NMS.

    Returns:
        labels (list): List of labels after NMS.
        boxes (Tensor): Tensor of shape (M, 4) representing the bounding boxes after NMS.
        scores (Tensor): Tensor of shape (M,) representing the confidence scores after NMS.
    """
    if len(boxes) == 0:
        return labels, boxes, scores

    # Get coordinates of bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Compute areas of bounding boxes
    areas = (x2 - x1) * (y2 - y1)

    # Sort by scores
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        # The index of the highest scoring box
        i = order[0]
        keep.append(i)

        # If only one box is left, break
        if order.numel() == 1:
            break

        # Get coordinates of intersection rectangle
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        # Compute the area of intersection rectangle
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        # Compute IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = torch.where(ovr <= threshold)[0]
        order = order[inds + 1]

    keep = torch.tensor(keep)
    return [labels[i] for i in keep], boxes[keep], scores[keep]

def process_and_save_patches(source_folder, destination_folder, size):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Construct full file paths
            src_path = os.path.join(source_folder, filename)
            img = cv2.imread(src_path)

            # Split the image into patches
            patches, _ = split_image(img, sub_img_size=size, stride=size)

            # Save each patch
            base_name = os.path.splitext(filename)[0]  # Remove file extension
            for i, patch in enumerate(patches):
                patch_filename = f"{base_name}_patch_{i}.jpg"
                dst_path = os.path.join(destination_folder, patch_filename)
                cv2.imwrite(dst_path, patch)

    print("All patches have been processed and saved.")

def convert_results(yolo_box_results, label_names):
    labels = []
    boxes = []
    scores = []

    for result in yolo_box_results:
        box = result.xyxy
        label_id = int(result.cls)
        label = label_names[label_id]
        score = result.conf

        labels.append(label)
        boxes.append(box)
        scores.append(score)

    return labels, boxes, scores