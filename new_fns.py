import cv2
import numpy as np

def scale_ocr_box_additive(ocr_results, scale):
    new_results = []
    for box, text, score in ocr_results:
        new_box = scale_box_additive(box, scale)
        new_result = (new_box, text, score)
        new_results.append(new_result)

    return new_results
def scale_box_additive(box, scale):
    """
    Scale a box by adding padding proportional to its area.
    Scale of 1.0 results in no padding.

    Args:
        box: List of coordinates [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        scale: Scale factor where 1.0 means no change

    Returns:
        List of coordinates for the scaled box
    """
    # Extract coordinates
    box_x1, box_y1 = box[0]
    box_x2, box_y2 = box[2]

    # Calculate current width and height
    width = box_x2 - box_x1
    height = box_y2 - box_y1

    # Calculate area
    area = width * height

    # Calculate padding based on scale relative to 1.0
    # When scale is 1.0, (scale - 1) is 0, so no padding is added
    padding = (scale - 1) * (area ** 0.5)

    # Add padding equally to all sides
    new_x1 = int(box_x1 - padding)
    new_y1 = int(box_y1 - padding)
    new_x2 = int(box_x2 + padding)
    new_y2 = int(box_y2 + padding)

    return [(new_x1, new_y1), (new_x2, new_y1), (new_x2, new_y2), (new_x1, new_y2)]


def group_lines(lines, max_distance=20, check_line_points=False):
    """
    Groups lines whose endpoints are within max_distance of each other.
    Optional check for points near line segments.

    Args:
        lines: numpy array of shape (N, 1, 4) where each line is [x1, y1, x2, y2]
        max_distance: maximum distance between endpoints to consider lines as part of same group
        check_line_points: if True, also check distances from endpoints to line segments
    """
    if lines is None or len(lines) == 0:
        return []

    lines = lines.reshape(-1, 4)
    n_lines = len(lines)
    groups = []
    used = set()

    def point_to_line_distance(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            return np.linalg.norm(point_vec)

        t = np.dot(point_vec, line_vec) / (line_length * line_length)
        t = max(0, min(1, t))
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)

    def endpoint_distance(line1, line2):
        p1 = np.array([[line1[0], line1[1]], [line1[2], line1[3]]])
        p2 = np.array([[line2[0], line2[1]], [line2[2], line2[3]]])

        endpoint_dists = np.sqrt(np.sum((p1[:, np.newaxis] - p2) ** 2, axis=2))
        min_dist = np.min(endpoint_dists)

        if check_line_points:
            line1_to_2 = min(point_to_line_distance(p1[0], p2[0], p2[1]),
                             point_to_line_distance(p1[1], p2[0], p2[1]))
            line2_to_1 = min(point_to_line_distance(p2[0], p1[0], p1[1]),
                             point_to_line_distance(p2[1], p1[0], p1[1]))
            min_dist = min(min_dist, line1_to_2, line2_to_1)

        return min_dist

    def find_connected_lines(line_idx, current_group):
        if line_idx in used:
            return

        used.add(line_idx)
        current_group.append(line_idx)

        for i in range(n_lines):
            if i not in used:
                if endpoint_distance(lines[line_idx], lines[i]) <= max_distance:
                    find_connected_lines(i, current_group)

    for i in range(n_lines):
        if i not in used:
            current_group = []
            find_connected_lines(i, current_group)
            groups.append(current_group)

    return [lines[group].tolist() for group in groups]

def white_out_line_groups(img, line_groups, line_thickness=5):
    """
    Draws white lines over the specified line groups in the image.

    Args:
        img: Input image (numpy array)
        line_groups: List of lists where each inner list contains lines [x1, y1, x2, y2]
        line_thickness: Thickness of the white lines to draw

    Returns:
        Modified image with white lines drawn over the specified groups
    """
    # Create a copy to avoid modifying the original image
    result = img.copy()

    # White color in BGR
    WHITE = (255, 255, 255)

    # Draw white lines for each group
    for group in line_groups:
        for line in group:
            x1, y1, x2, y2 = map(int, line)  # Ensure coordinates are integers
            cv2.line(result, (x1, y1), (x2, y2), WHITE, line_thickness)

    return result


def box_intersects_line(box, line_group, scale=1.0):
    """
    Checks if any line in the line group intersects with the given box.

    Args:
        box: Tuple of (x1, y1, x2, y2) representing the box corners
        line_group: List of lines where each line is [x1, y1, x2, y2]
        scale: Float value to scale the box (1.0 means original size, 2.0 means double size)

    Returns:
        Boolean indicating whether any line intersects with the box
    """
    box_x1, box_y1, box_x2, box_y2 = box

    # Ensure box coordinates are in correct order (top-left, bottom-right)
    box_x1, box_x2 = min(box_x1, box_x2), max(box_x1, box_x2)
    box_y1, box_y2 = min(box_y1, box_y2), max(box_y1, box_y2)

    # Calculate box center
    center_x = (box_x1 + box_x2) / 2
    center_y = (box_y1 + box_y2) / 2

    # Calculate box dimensions
    width = box_x2 - box_x1
    height = box_y2 - box_y1

    # Calculate scaled dimensions
    scaled_width = width * scale
    scaled_height = height * scale

    # Calculate new box coordinates maintaining the center point
    box_x1 = center_x - (scaled_width / 2)
    box_x2 = center_x + (scaled_width / 2)
    box_y1 = center_y - (scaled_height / 2)
    box_y2 = center_y + (scaled_height / 2)

    def ccw(A, B, C):
        """Returns true if points are arranged counter-clockwise"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def line_segments_intersect(line1_start, line1_end, line2_start, line2_end):
        """Returns true if two line segments intersect"""
        return ccw(line1_start, line2_start, line2_end) != ccw(line1_end, line2_start, line2_end) and \
            ccw(line1_start, line1_end, line2_start) != ccw(line1_start, line1_end, line2_end)

    def point_in_box(point):
        """Returns true if point is inside box"""
        x, y = point
        return box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2

    # Check each line in the group
    for line in line_group:
        x1, y1, x2, y2 = line
        line_start = (x1, y1)
        line_end = (x2, y2)

        # Check if either endpoint is inside the box
        if point_in_box(line_start) or point_in_box(line_end):
            return True

        # Check intersection with each box edge
        box_edges = [
            ((box_x1, box_y1), (box_x2, box_y1)),  # Top edge
            ((box_x2, box_y1), (box_x2, box_y2)),  # Right edge
            ((box_x2, box_y2), (box_x1, box_y2)),  # Bottom edge
            ((box_x1, box_y2), (box_x1, box_y1))  # Left edge
        ]

        for edge_start, edge_end in box_edges:
            if line_segments_intersect(line_start, line_end, edge_start, edge_end):
                return True

    return False

def scale_line_group(line_group, scale):
    """Scale the coordinates of a line group by the given factor."""
    return [
        [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)]
        for x1, y1, x2, y2 in line_group
    ]

def scale_box(box, scale):
    """Scale the coordinates of a bounding box by the given factor.
    Box format is [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]"""
    return [
        (int(point[0] * scale), int(point[1] * scale))
        for point in box
    ]


def remove_lines_that_intersect_box(lines, boxes):
    """
    Remove lines that intersect with any of the detection boxes.

    Args:
        lines: numpy array of lines in format [[x1, y1, x2, y2], ...]
        boxes: tensor or numpy array of boxes in format [[x1, y1, x2, y2], ...]

    Returns:
        filtered_lines: numpy array of lines that don't intersect with boxes
    """
    if lines is None:
        return None

    def ccw(A, B, C):
        """
        Check if three points are listed in counterclockwise order
        """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def line_segments_intersect(line1_start, line1_end, line2_start, line2_end):
        """
        Check if two line segments intersect
        """
        return ccw(line1_start, line2_start, line2_end) != ccw(line1_end, line2_start, line2_end) and \
            ccw(line1_start, line1_end, line2_start) != ccw(line1_start, line1_end, line2_end)

    filtered_lines = []

    # Handle list of tensors
    if isinstance(boxes, list):
        boxes = np.array([box.numpy() if hasattr(box, 'numpy') else box for box in boxes])
    # Handle single tensor
    elif hasattr(boxes, 'numpy'):
        boxes = boxes.numpy()

    # Ensure boxes is 2D array
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)

    for line in lines:
        x1, y1, x2, y2 = line[0]  # HoughLinesP returns nested array
        line_start = (x1, y1)
        line_end = (x2, y2)

        intersects_any_box = False

        for box in boxes:
            # Create four corners of the box
            box_x1, box_y1, box_x2, box_y2 = map(int, box)  # Convert to int for pixel coordinates

            # Check intersection with all four sides of the box
            box_lines = [
                ((box_x1, box_y1), (box_x2, box_y1)),  # Top
                ((box_x2, box_y1), (box_x2, box_y2)),  # Right
                ((box_x2, box_y2), (box_x1, box_y2)),  # Bottom
                ((box_x1, box_y2), (box_x1, box_y1))  # Left
            ]

            for box_line_start, box_line_end in box_lines:
                if line_segments_intersect(line_start, line_end, box_line_start, box_line_end):
                    intersects_any_box = True
                    break

            if intersects_any_box:
                break

        if not intersects_any_box:
            filtered_lines.append(line)

    return np.array(filtered_lines) if filtered_lines else None


def remove_lines_with_endpoints_in_boxes(lines, boxes):
    """
    Remove lines that have either endpoint contained within any of the detection boxes.

    Args:
        lines: numpy array of lines in format [[x1, y1, x2, y2], ...]
        boxes: tensor or numpy array of boxes in format [[x1, y1, x2, y2], ...]

    Returns:
        filtered_lines: numpy array of lines whose endpoints don't fall within boxes
    """
    if lines is None:
        return None

    def point_in_box(x, y, box):
        """
        Check if point (x,y) is inside the box
        """
        box_x1, box_y1, box_x2, box_y2 = map(int, box)  # Convert to int for pixel coordinates
        return (box_x1 <= x <= box_x2) and (box_y1 <= y <= box_y2)

    filtered_lines = []

    # Handle list of tensors
    if isinstance(boxes, list):
        boxes = np.array([box.numpy() if hasattr(box, 'numpy') else box for box in boxes])
    # Handle single tensor
    elif hasattr(boxes, 'numpy'):
        boxes = boxes.numpy()

    # Ensure boxes is 2D array
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)

    for line in lines:
        x1, y1, x2, y2 = line[0]  # HoughLinesP returns nested array

        endpoint_in_box = False

        for box in boxes:
            # Check if either endpoint is in the box
            if point_in_box(x1, y1, box) or point_in_box(x2, y2, box):
                endpoint_in_box = True
                break

        if not endpoint_in_box:
            filtered_lines.append(line)

    return np.array(filtered_lines) if filtered_lines else None