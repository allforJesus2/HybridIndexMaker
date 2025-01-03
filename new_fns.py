def group_lines(lines, max_distance=10):
    """
    Groups lines whose endpoints are within max_distance of each other.

    Args:
        lines: numpy array of shape (N, 1, 4) where each line is [x1, y1, x2, y2]
        max_distance: maximum distance between endpoints to consider lines as part of same group

    Returns:
        List of lists, where each inner list contains indices of lines that belong together
    """
    if lines is None or len(lines) == 0:
        return []

    # Convert lines to regular array of shape (N, 4)
    lines = lines.reshape(-1, 4)
    n_lines = len(lines)

    # Initialize groups
    groups = []
    used = set()

    def endpoint_distance(line1, line2):
        """Calculate minimum distance between any endpoints of two lines"""
        p1 = np.array([[line1[0], line1[1]], [line1[2], line1[3]]])
        p2 = np.array([[line2[0], line2[1]], [line2[2], line2[3]]])

        # Calculate distances between all combinations of endpoints
        distances = np.sqrt(np.sum((p1[:, np.newaxis] - p2) ** 2, axis=2))
        return np.min(distances)

    def find_connected_lines(line_idx, current_group):
        """Recursively find all lines connected to the current line"""
        if line_idx in used:
            return

        used.add(line_idx)
        current_group.append(line_idx)

        # Check all other lines
        for i in range(n_lines):
            if i not in used:
                if endpoint_distance(lines[line_idx], lines[i]) <= max_distance:
                    find_connected_lines(i, current_group)

    # Find groups of connected lines
    for i in range(n_lines):
        if i not in used:
            current_group = []
            find_connected_lines(i, current_group)
            groups.append(current_group)

    # Convert indices to actual lines
    return [lines[group].tolist() for group in groups]


import cv2
import numpy as np


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