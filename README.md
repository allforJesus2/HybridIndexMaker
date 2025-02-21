# PIDVision.AI

PIDVision.AI is a tool for extracting instrument data and metadata from P&ID drawings using computer vision and OCR.

## How It Works

The basic workflow is:

1. **Region Selection**: Users draw boxes around regions of interest on the P&ID image by clicking and dragging.

2. **Capture Modes**:
   - **Instruments**: Captures groups of instruments using object recognition. Multiple instrument regions can be captured at once.
   - **Service In/Out**: Draw boxes around service text to capture flow direction
   - **Line**: Capture line numbers and specifications
   - **Equipment**: Capture equipment tags and descriptions
   - **PID**: Capture the drawing number/title
   - **Comment**: Capture any additional text annotations

3. **Data Association**: The captured metadata (services, lines, etc.) is automatically associated with the instruments from the captured regions.

4. **Data Review**: Review the captured data in the data window, which shows:
   - Instrument tags and numbers
   - Associated line numbers
   - Service flows
   - Equipment tags
   - Comments

5. **Excel Export**: Once everything looks correct, press 'W' to write the data to an Excel file.

## Key Controls

- **Mouse Controls**:
  - Left click + drag: Draw capture box
  - Right click + drag: Pan image
  - Mouse wheel: Zoom in/out

- **Keyboard Shortcuts**:
  - `N`: Next page
  - `B`: Previous page
  - `P`: Set PID capture mode
  - `A`: Set instrument capture mode
  - `F`: Set line capture mode
  - `E`: Set equipment capture mode
  - `Z`: Set service in capture mode
  - `X`: Set service out capture mode
  - `G`: Set comment capture mode
  - `W`: Write to Excel
  - `C`: Clear instrument group
  - `V`: Vote to normalize tag numbers
  - `S`: Swap services

## Tips

- Use the capture mode indicator in the top-left to confirm your current capture mode
- Hold Shift while capturing services to append with formatting
- Hold Ctrl while capturing services to merge common substrings
- The data window shows all captured instruments and their associated metadata
- Double-click entries in the data window to edit them manually
- Use the View menu to navigate between pages or jump to specific PIDs

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- EasyOCR
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run PIDVision.py

## License

This software requires a valid license key for full functionality. Contact support for licensing information.

## Features

- **Object Detection**: Pre-trained model for detecting various instrument types and equipment
- **OCR Integration**: Uses EasyOCR for accurate text recognition
- **Batch Processing**: Process multiple P&ID pages in sequence
- **Data Validation**: Tools for reviewing and correcting captured data
- **Excel Integration**: Direct export to Excel with configurable formats
- **Instrument Counting**: Automated instrument counting across multiple pages
- **PDF Tools**: Built-in tools for PDF to image conversion and PDF merging

## Advanced Settings

Access advanced settings through the Settings menu:

- **Reader Settings**: Configure OCR parameters for different text types
- **Detection Settings**: Adjust object detection confidence thresholds
- **Group Settings**: Define instrument grouping rules and association radius
- **Line Processing**: Configure line detection and processing parameters

## Troubleshooting

Common issues and solutions:

- If OCR results are poor, try adjusting the Reader Settings
- For missed instruments, check Detection Settings and minimum scores
- If line detection fails, adjust Line Processing parameters
- For memory issues with large drawings, adjust the processing stride and window size

## Support

For technical support or to report issues:
1. Check the troubleshooting section
2. Review settings documentation
3. Contact technical support with:
   - Software version
   - Error messages
   - Sample images (if possible)
   - Steps to reproduce the issue

## Updates

Check for updates regularly to access:
- Improved detection models
- OCR enhancements
- Bug fixes
- New features 