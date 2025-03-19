import torch
import cv2
from functions import (
    calculate_center, calculate_distance, filter_ocr_results,
    calculate_overlap, get_comment, process_line_data,
    get_lines_from_box
)
from utilities.easyocr_mosaic import HardOCR
from model_predict_mosaic import model_predict_on_mosaic

class InstrumentProcessor:
    def __init__(self, 
                 model_inst=None,
                 model_inst_path=None,
                 detection_labels=None,
                 reader=None,
                 instrument_reader_settings=None,
                 reader_settings=None):
        self.model_inst = model_inst
        self.model_inst_path = model_inst_path
        self.detection_labels = detection_labels
        self.reader = reader
        self.instrument_reader_settings = instrument_reader_settings
        self.reader_settings = reader_settings

    def process_instruments(self, 
                          img,
                          prediction_data,
                          expand=1.0,
                          radius=180,
                          inst_labels=None,
                          other_labels=None,
                          min_scores=None,
                          offset=None,
                          comment_box_expand=30,
                          tag_label_groups=None,
                          capture_ocr=True,
                          reader_sub_img_size=1300,
                          reader_stride=1250,
                          filter_ocr_threshold=0.9,
                          line_params=None,
                          text_corrections=None):
        """Process and extract instrument data from image.
        
        Args:
            img: Input image
            prediction_data: Model prediction data
            expand: Box expansion factor
            radius: Maximum radius for finding closest other object
            inst_labels: List of instrument labels to process
            other_labels: List of other object labels to consider
            min_scores: Minimum confidence scores per label
            offset: Optional (x,y) offset to apply to boxes
            comment_box_expand: Pixels to expand comment box
            tag_label_groups: Groups of valid tag labels
            capture_ocr: Whether to perform OCR
            reader_sub_img_size: Size of OCR sub-images
            reader_stride: Stride for OCR processing
            filter_ocr_threshold: Threshold for filtering OCR results
            line_params: Parameters for line detection
            text_corrections: Text correction rules to apply
            
        Returns:
            List of dictionaries containing processed instrument data
        """
        all_data = []
        group_inst = []
        group_other = []
        got_one = False
        detecto_boxes = []
        pred_boxes = []

        # Process prediction data and group instruments/other objects
        for label, box, score, visual_elements in prediction_data:
            try:
                if score < min_scores[label]:
                    continue
            except Exception as e:
                print(e, '. Maybe try setting minscores --> settings >> set object minscores')

            if label in inst_labels:
                group_inst.append((label, box, score, visual_elements))
                detecto_boxes.append(box)
            if label in other_labels:
                group_other.append((label, box, score, visual_elements))

            if offset is not None:
                offset_tensor = torch.tensor([offset[0], offset[1], offset[0], offset[1]])
                offset_box = box + offset_tensor
                pred_boxes.append(offset_box)
            else:
                pred_boxes.append(box)

        if not group_inst:
            return

        # Perform OCR if requested
        local_ocr_results = None
        if capture_ocr:
            print('getting local ocr')
            local_ocr_results = HardOCR(img, self.reader, self.reader_settings, 
                                      sub_img_size=reader_sub_img_size,
                                      stride=reader_stride)
            print('unfiltered local ocr results:\n', local_ocr_results)
            local_ocr_results = filter_ocr_results(local_ocr_results, pred_boxes, 
                                                 overlap_threshold=filter_ocr_threshold)
            print('filtered local ocr results:\n', local_ocr_results)

        # Process line data
        line_data = process_line_data(
            img,
            local_ocr_results,
            **line_params.__dict__,
            detecto_boxes=detecto_boxes,
        )

        # Process each instrument
        for label, box, score, visual_elements in group_inst:
            # Get line information
            if line_params.simple:
                lines = line_data
            elif line_data:
                lines = get_lines_from_box(box, line_data, img_scale=line_params.line_img_scale)
                if lines and len(lines) == 1:
                    lines = lines[0]
            else:
                lines = ''

            # Apply text corrections to lines
            if isinstance(lines, list):
                updated_lines = []
                for l in lines:
                    updated_lines.append(text_corrections.apply_corrections(l))
                lines = updated_lines
            else:
                lines = text_corrections.apply_corrections(lines)

            # Process instrument tag and number
            tag, tag_no = self._process_instrument_tag(box, img, expand)

            # Apply text corrections
            if text_corrections:
                tag = text_corrections.apply_corrections(tag)
                tag_no = text_corrections.apply_corrections(tag_no)

            # Find instrument type
            inst_center = calculate_center(box)
            valid_types = ['']
            if tag_label_groups and tag:
                valid_types = self._get_valid_types_for_tag(tag, tag_label_groups)

            inst_type = self._find_closest_other(inst_center, group_other, label, radius, 
                                               valid_types, tag_label_groups)

            # Get comment if OCR results available
            comment = ''
            if offset is not None:
                offset_tensor = torch.tensor([offset[0], offset[1], offset[0], offset[1]])
                offset_box = box + offset_tensor
            else:
                offset_box = box

            if local_ocr_results:
                print('getting local ocr comment')
                comment = get_comment(local_ocr_results, box, comment_box_expand)
                if text_corrections and comment:
                    comment = text_corrections.apply_corrections(comment)

            # Create data dictionary
            data = {
                'tag': tag,
                'tag_no': "'" + tag_no,
                'score': score,
                'box': offset_box,
                'label': label,
                'type': inst_type,
                'comment': comment,
                'line': lines,
                'visual_elements': visual_elements
            }
            all_data.append(data)

        return all_data

    def _process_instrument_tag(self, box, img, expand):
        """Process instrument tag and number from image region."""
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        x_expand = int((x_max - x_min) * (expand - 1) / 2)
        y_expand = int((y_max - y_min) * (expand - 1) / 2)

        crop_img = img[(y_min - y_expand):(y_max + y_expand), 
                      (x_min - x_expand):(x_max + x_expand)]
        
        tag = ''
        tag_no = ''
        
        try:
            results = self.reader.readtext(crop_img, **self.instrument_reader_settings)
            if results:
                filename = 'temp/instrument_capture.png'
                cv2.imwrite(filename, crop_img)
                
                tag = results[0][1]
                tag_no = ' '.join([box[1] for box in results[1:]])
                
        except Exception as e:
            print('error in instrument OCR:', e)
            
        return tag, tag_no

    def _get_valid_types_for_tag(self, tag, tag_label_groups):
        """Get valid types for a given tag based on tag label groups."""
        for tag_group, valid_types in tag_label_groups.items():
            tag_prefixes = tag_group.split()
            if tag in tag_prefixes:
                return valid_types
        return None

    def _find_closest_other(self, inst_center, group_other, current_label, radius, 
                          valid_types=[''], tag_label_groups=None):
        """Find closest valid object to instrument center."""
        min_distance = radius
        closest_label = ''

        for label, box, score, _ in group_other:
            if label == current_label:
                continue

            if tag_label_groups:
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

    def predict_on_mosaic(self, img, threshold=0.5, square_size=1300, stride=1200):
        """Run model prediction on image using mosaic approach."""
        if not hasattr(self, 'model_inst') or self.model_inst is None:
            print("Loading object detection model...")
            self.model_inst = Model.load(self.model_inst_path, self.detection_labels)
            
        return model_predict_on_mosaic(
            img,
            self.model_inst,
            threshold=threshold,
            square_size=square_size,
            stride=stride
        ) 