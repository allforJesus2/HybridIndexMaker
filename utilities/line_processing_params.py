from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class LineProcessingParams:
    """Parameters for line processing in process_line_data function."""

    # Required parameters that match process_line_data signature
    re_line: str
    simple: bool = True
    debug_line: bool = True
    remove_significant_lines_only: bool = True

    # Image processing parameters
    paint_line_thickness: int = 8
    line_join_threshold: int = 20
    line_box_scale: float = 1.2
    erosion_kernel: int = 10
    erosion_iterations: int = 1
    binary_threshold: int = 200
    line_img_scale: float = 1.15
    clean_img: bool = True
    remove_text_before: bool = False
    text_min_score: float = 0.5
    white_out_color: tuple = (255,255,255)

    # Optional parameter groups
    hough_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'rho': 1,
        'theta': 3.14159 / 180,
        'threshold': 120,
        'min_line_length': 50,
        'max_line_gap': 40
    })

    canny_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'low_threshold': 50,
        'high_threshold': 150,
        'aperture_size': 3
    })

    extension_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'merge_threshold': 10,
        'look_ahead': 20,
        'max_neighbors': 2
    })

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not self.simple and (self.hough_params is None or self.canny_params is None):
            raise ValueError("hough_params and canny_params are required when simple mode is False")