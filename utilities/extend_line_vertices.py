import numpy as np
import math
import numpy as np
from scipy.spatial import cKDTree


def extend_line_vertices(lines, distance_threshold=20, extension_length=50, max_neighbors=1):
    """
    Extend line endpoints if they have ≤max_neighbors neighboring vertices within a threshold.

    Args:
        lines (np.ndarray): Input lines from HoughLinesP, shape (N, 1, 4).
        distance_threshold (int): Max pixel distance to consider a vertex "nearby".
        extension_length (int): How far to extend the line (in pixels).
        max_neighbors (int): Maximum number of neighbors before skipping extension.
            If an endpoint has more neighbors than this, it won't be extended.

    Returns:
        np.ndarray: Extended lines with shape (N, 1, 4).
    """
    # Squeeze to (N, 4) and convert to list of tuples for easier manipulation
    lines_squeezed = np.squeeze(lines, axis=1)
    lines_list = [tuple(map(int, line)) for line in lines_squeezed]

    # Collect all endpoints from all lines
    all_endpoints = []
    for line in lines_list:
        x1, y1, x2, y2 = line
        all_endpoints.append((x1, y1))
        all_endpoints.append((x2, y2))

    extended_lines = []
    for idx, line in enumerate(lines_list):
        x1, y1, x2, y2 = line

        # Get current line's endpoints to exclude from neighbor checks
        current_endpoints = {(x1, y1), (x2, y2)}

        # Check neighbors for each endpoint
        new_x1, new_y1 = x1, y1
        new_x2, new_y2 = x2, y2

        # Direction vector of the line
        dx = x2 - x1
        dy = y2 - y1
        line_length = math.hypot(dx, dy)

        if line_length == 0:  # Skip zero-length lines
            extended_lines.append([x1, y1, x2, y2])
            continue

        # Unit vector in the line's direction
        unit_dx = dx / line_length
        unit_dy = dy / line_length

        # --- Process Start Point (x1, y1) ---
        neighbors = 0
        for ep in all_endpoints:
            if ep in current_endpoints:
                continue  # Skip own endpoints
            distance = math.hypot(ep[0] - x1, ep[1] - y1)
            if distance <= distance_threshold:
                neighbors += 1
                if neighbors > max_neighbors:  # Early exit if we exceed max_neighbors
                    break

        if neighbors <= max_neighbors:
            # Extend backward along the line direction
            new_x1 = x1 - unit_dx * extension_length
            new_y1 = y1 - unit_dy * extension_length

        # --- Process End Point (x2, y2) ---
        neighbors = 0
        for ep in all_endpoints:
            if ep in current_endpoints:
                continue  # Skip own endpoints
            distance = math.hypot(ep[0] - x2, ep[1] - y2)
            if distance <= distance_threshold:
                neighbors += 1
                if neighbors > max_neighbors:  # Early exit if we exceed max_neighbors
                    break

        if neighbors <= max_neighbors:
            # Extend forward along the line direction
            new_x2 = x2 + unit_dx * extension_length
            new_y2 = y2 + unit_dy * extension_length

        # Round to integers and clip to image boundaries if needed
        new_line = [
            int(round(new_x1)), int(round(new_y1)),
            int(round(new_x2)), int(round(new_y2))
        ]

        extended_lines.append(new_line)

    # Convert back to original HoughLinesP format (N, 1, 4)
    return np.array(extended_lines, dtype=np.int32).reshape(-1, 1, 4)

def extend_line_vertices_optimized(lines, distance_threshold=20, extension_length=50, max_neighbors=1):
    """
    Optimized version of line endpoint extension using KD-Tree for faster neighbor searches.
    """
    lines_squeezed = np.squeeze(lines, axis=1)

    # Extract all endpoints into a single array for KD-Tree
    endpoints = np.vstack((
        lines_squeezed[:, :2],  # Start points (x1, y1)
        lines_squeezed[:, 2:]  # End points (x2, y2)
    ))

    # Build KD-Tree once for efficient neighbor searches
    tree = cKDTree(endpoints)

    # Calculate unit vectors for all lines at once
    dx = lines_squeezed[:, 2] - lines_squeezed[:, 0]
    dy = lines_squeezed[:, 3] - lines_squeezed[:, 1]
    lengths = np.hypot(dx, dy)

    # Handle zero-length lines
    valid_lengths = lengths > 0
    unit_dx = np.zeros_like(dx)
    unit_dy = np.zeros_like(dy)
    unit_dx[valid_lengths] = dx[valid_lengths] / lengths[valid_lengths]
    unit_dy[valid_lengths] = dy[valid_lengths] / lengths[valid_lengths]

    # Initialize output array
    extended_lines = lines_squeezed.copy()

    # Process start points
    for i in range(len(lines_squeezed)):
        if not valid_lengths[i]:
            continue

        start_point = lines_squeezed[i, :2]
        end_point = lines_squeezed[i, 2:]

        # Find neighbors for start point
        neighbors = tree.query_ball_point(start_point, distance_threshold)
        neighbors = [n for n in neighbors if not np.array_equal(endpoints[n], start_point)
                     and not np.array_equal(endpoints[n], end_point)]

        if len(neighbors) <= max_neighbors:
            extended_lines[i, 0] = start_point[0] - unit_dx[i] * extension_length
            extended_lines[i, 1] = start_point[1] - unit_dy[i] * extension_length

        # Find neighbors for end point
        neighbors = tree.query_ball_point(end_point, distance_threshold)
        neighbors = [n for n in neighbors if not np.array_equal(endpoints[n], start_point)
                     and not np.array_equal(endpoints[n], end_point)]

        if len(neighbors) <= max_neighbors:
            extended_lines[i, 2] = end_point[0] + unit_dx[i] * extension_length
            extended_lines[i, 3] = end_point[1] + unit_dy[i] * extension_length

    return extended_lines.reshape(-1, 1, 4).astype(np.int32)