import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class ConvolutionReplacer:

    def __init__(self, kernel_directory, scale, rotate):
        self.rotate = rotate
        self.kernel_directory = kernel_directory
        self.scale = scale
        self.kernel_images, self.kernel_images_scaled = self.kernel_data_load(
            self.kernel_directory)

    def load_images_from_directory(self, directory):
        images = {}
        images_scaled = {}
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                file_path = os.path.join(directory, filename)
                root_name, _ = os.path.splitext(filename)
                images[root_name] = cv2.imread(file_path)
                # scaling
                images_scaled[root_name] = cv2.resize(images[root_name], None, fx=self.scale, fy=self.scale)
        return images, images_scaled

    def detect_objects(self, source_image, kernel_data, threshold):

        source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        detections = []

        for label, kernel_image in kernel_data.items():

            print(f'label {label}')
            rotated_kernel = kernel_image.copy()

            rotations = [0] if not self.rotate else range(self.rotate)

            for rotation in rotations:

                print(f'>>> starting rotation {rotation}')
                # print('Convert the kernel image to grayscale')
                rotated_kernel_gray = cv2.cvtColor(rotated_kernel, cv2.COLOR_BGR2GRAY)

                # print('Perform cross-correlation using matchTemplate')
                result = cv2.matchTemplate(source_gray, rotated_kernel_gray, cv2.TM_CCOEFF_NORMED)

                # print('Find all matching locations above the threshold')
                locations = np.where(result >= threshold)
                h, w = rotated_kernel_gray.shape

                # print('Iterate through matching instances')
                for loc in zip(*locations[::-1]):
                    top_left = loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    score = result[loc[1], loc[0]]
                    detections.append((top_left, bottom_right, label, rotation, score))

                rotated_kernel = cv2.rotate(rotated_kernel, 0)

        return detections

    def non_max_suppression(self, detections, overlap_threshold):
        detections.sort(key=lambda x: x[3], reverse=True)

        final_detections = []

        while len(detections) > 0:
            top_left1, bottom_right1, label1, rotation1, score1 = detections[0]
            final_detections.append((top_left1, bottom_right1, label1, rotation1, score1))
            detections.pop(0)

            to_remove = []
            for i, (top_left2, bottom_right2, label2, rotation2, score2) in enumerate(detections):
                x1 = max(top_left1[0], top_left2[0])
                y1 = max(top_left1[1], top_left2[1])
                x2 = min(bottom_right1[0], bottom_right2[0])
                y2 = min(bottom_right1[1], bottom_right2[1])

                intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
                area1 = (bottom_right1[0] - top_left1[0] + 1) * (bottom_right1[1] - top_left1[1] + 1)
                area2 = (bottom_right2[0] - top_left2[0] + 1) * (bottom_right2[1] - top_left2[1] + 1)
                iou = intersection_area / float(area1 + area2 - intersection_area)

                if iou > overlap_threshold:
                    to_remove.append(i)

            to_remove.sort(reverse=True)
            for index in to_remove:
                detections.pop(index)

        return final_detections

    def detect(self, source_image, threshold=0.81, overlap_threshold=0.3):

        source_image_scaled = cv2.resize(source_image, None, fx=self.scale, fy=self.scale)

        # kernel_images, replace_images,
        detections = self.detect_objects(source_image_scaled, self.kernel_images_scaled, threshold)

        final_detections = self.non_max_suppression(detections, overlap_threshold)

        result_boxes_image = self.draw_boxes(source_image_scaled, final_detections)

        final_detections_rescaled = self.rescale_detections(final_detections)

        return result_boxes_image, final_detections_rescaled

    def rescale_detections(self, final_detections):
        scaled_detections = []  # Create a new list to store scaled detections
        for top_left, bottom_right, label, rotation, score in final_detections:
            scaled_top_left = (top_left[0] / self.scale, top_left[1] / self.scale)
            scaled_bottom_right = (bottom_right[0] / self.scale, bottom_right[1] / self.scale)
            scaled_detections.append((scaled_top_left, scaled_bottom_right, label, rotation, score))

        return scaled_detections  # Return the updated detections

    def draw_boxes(self, source_image, final_detections):
        result_boxes_image = source_image.copy()

        for top_left, bottom_right, label, rotation, score in final_detections:
            cv2.rectangle(result_boxes_image, top_left, bottom_right, (0, 150, 0), 2)
            cv2.putText(result_boxes_image, f'{label} ({score:.2f}), R{str(rotation)}', (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 150, 0), 2)

        return result_boxes_image

    def replace_objects(self, source_image, replace_images, final_detections):
        # Create a copy of the source image to draw rectangles and labels on
        source_image_replaced = source_image.copy()

        for top_left, bottom_right, label, rotation, score in final_detections:

            replace_image = replace_images.get(label)

            if replace_image is not None:

                if rotation == 1:
                    replace_image = cv2.rotate(replace_image, 0)
                if rotation == 2:
                    replace_image = cv2.rotate(replace_image, 1)
                if rotation == 3:
                    replace_image = cv2.rotate(replace_image, 2)

                # print('source image is probably a different size, so were gonna stick it in the middle')
                replace_height, replace_width = replace_image.shape[:2]

                # Calculate coordinates to center the replacement image over the detected region
                center_x = int((top_left[0] + bottom_right[0] - replace_width) / 2)
                center_y = int((top_left[1] + bottom_right[1] - replace_height) / 2)

                # Ensure the replacement coordinates fit within the source image boundaries
                max_x = min(center_x + replace_width, source_image_replaced.shape[1])
                max_y = min(center_y + replace_height, source_image_replaced.shape[0])

                # Calculate the region where the replacement image will fit
                replace_region = source_image_replaced[center_y:max_y, center_x:max_x]

                # Calculate the region where the replacement image will be placed
                target_region = replace_image[:replace_region.shape[0], :replace_region.shape[1]]

                # Replace the target region with the replacement image
                source_image_replaced[center_y:max_y, center_x:max_x] = target_region

        return source_image_replaced

    def kernel_data_load(self, kernel_directory):
        kernel_images, kernel_images_scaled = self.load_images_from_directory(kernel_directory)
        return kernel_images, kernel_images_scaled