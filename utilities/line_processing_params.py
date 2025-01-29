"""
Configuration parameters for line processing in image analysis.
This module contains the dataclass that encapsulates all parameters used in line detection and processing.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LineProcessingParams:
    """Parameters for line processing in process_line_data function.

    This class encapsulates all configuration parameters used in line detection
    and processing, providing type safety and default values.

    Attributes:
        simple_mode (bool): If True, uses simple line detection mode
        debug_line (bool): If True, shows debug visualizations
        remove_significant_lines_only (bool): If True, only removes significant lines
        paint_line_thickness (int): Thickness of lines in pixels
        line_join_threshold (int): Maximum distance to join line segments
        line_box_scale (float): Scale factor for line bounding boxes
        erosion_kernel (int): Size of erosion kernel
        erosion_iterations (int): Number of erosion iterations
        binary_threshold (int): Threshold for binary image conversion
        line_img_scale (float): Scale factor for line image
        hough_params (Dict): Parameters for Hough transform
        canny_params (Dict): Parameters for Canny edge detection
        extension_params (Dict): Parameters for line extension
    """

    # Mode parameters
    simple_mode: bool = True
    debug_line: bool = True
    remove_significant_lines_only: bool = True

    # Image processing parameters
    paint_line_thickness: int = 5
    line_join_threshold: int = 20
    line_box_scale: float = 1.5
    erosion_kernel: int = 2
    erosion_iterations: int = 1
    binary_threshold: int = 200
    line_img_scale: float = 1.0

    # Optional parameter groups with default factory functions
    hough_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'rho': 1,
        'theta': 3.14159 / 180,
        'threshold': 50,
        'min_line_length': 50,
        'max_line_gap': 10
    })

    canny_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'low_threshold': 50,
        'high_threshold': 150,
        'aperture_size': 3
    })

    extension_params: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        'merge_threshold': 20,
        'look_ahead': 3,
        'max_neighbors': 5
    })

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not self.simple_mode and (self.hough_params is None or self.canny_params is None):
            raise ValueError("hough_params and canny_params are required when simple_mode is False")

    @classmethod
    def create_simple(cls) -> 'LineProcessingParams':
        """Create an instance with simple mode settings.

        Returns:
            LineProcessingParams: Instance configured for simple mode
        """
        return cls(simple_mode=True)

    @classmethod
    def create_advanced(cls,
                        hough_threshold: int = 50,
                        line_length: int = 50,
                        line_gap: int = 10) -> 'LineProcessingParams':
        """Create an instance with advanced mode settings.

        Args:
            hough_threshold: Threshold for Hough transform
            line_length: Minimum line length
            line_gap: Maximum line gap

        Returns:
            LineProcessingParams: Instance configured for advanced mode
        """
        params = cls(simple_mode=False)
        params.hough_params.update({
            'threshold': hough_threshold,
            'min_line_length': line_length,
            'max_line_gap': line_gap
        })
        return params

    def update_hough_params(self, **kwargs) -> None:
        """Update Hough transform parameters.

        Args:
            **kwargs: Key-value pairs of parameters to update
        """
        if self.hough_params is None:
            self.hough_params = {}
        self.hough_params.update(kwargs)

    def update_canny_params(self, **kwargs) -> None:
        """Update Canny edge detection parameters.

        Args:
            **kwargs: Key-value pairs of parameters to update
        """
        if self.canny_params is None:
            self.canny_params = {}
        self.canny_params.update(kwargs)

    def update_extension_params(self, **kwargs) -> None:
        """Update line extension parameters.

        Args:
            **kwargs: Key-value pairs of parameters to update
        """
        if self.extension_params is None:
            self.extension_params = {}
        self.extension_params.update(kwargs)