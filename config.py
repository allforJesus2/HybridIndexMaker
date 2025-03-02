import os
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

class Config:
    """
    Configuration manager for PIDVision application.
    Handles loading, saving, and accessing all application settings.
    """
    
    def __init__(self, folder_path: str = None):
        """
        Initialize configuration with default values.
        
        Args:
            folder_path: Optional path to project folder for loading/saving config
        """
        self.folder_path = folder_path
        
        # Initialize with default values
        self._init_default_values()
        
        # Load config from file if folder_path is provided
        if folder_path:
            self.load()
    
    def _init_default_values(self):
        """Initialize all configuration parameters with default values"""
        
        # Image and navigation settings
        self.current_image_index = 0
        self.pid_coords = None
        
        # Model paths
        self.model_inst_path = "models/vortex_large.pth"
        
        # OCR settings
        self.do_local_ocr = False
        self.filter_ocr_threshold = 0.9
        self.reader_stride = 550
        self.reader_sub_img_size = 600
        self.pred_square_size = 1300
        self.pred_stride = 1250
        
        # Detection settings
        self.detection_labels = []
        self.nms_threshold = 0.5
        self.object_box_expand = 1.0
        self.default_min_detection_score = 0.74
        self.min_scores = {}
        
        # Group settings
        self.association_radius = 180
        self.tag_label_groups = {
            "FE FIT": ["CORIOLIS", "MAGNETIC", "PITOT", "TURBINE", "ULTRASONIC", "VORTEX"],
            "PCV TCV LCV SDV AV XV HCV FCV FV PV TV LV": ["BALL", "BUTTERFLY", "DIAPHRAM", "GATE",
                                                          "GLOBE", "KNIFE", "PLUG", "VBALL"],
            "LE LIT LT": ["GWR", "PR"], 
            "PT PIT PI DPIT": ["SEAL"]
        }
        self.group_inst = []
        self.group_other = []
        
        # Comment settings
        self.comment_box_expand = 20
        
        # Line detection settings
        self.re_line = r'.*\"-[A-Z\d]{1,5}-.*'
        self.paint_line_thickness = 5
        self.line_join_threshold = 20
        self.line_box_scale = 1.5
        self.line_img_erosion = 2
        self.line_erosion_iterations = 2
        self.line_img_binary_threshold = 200
        self.line_img_scale = 1.0
        self.simple_line_mode = True
        self.debug_line = True
        self.remove_significant_lines_only = True
        self.remove_text_before = False
        self.text_min_score = 0.5
        self.white_out_color = (255, 255, 255)
        
        # Canny and Hough parameters
        self.canny_params = None
        self.hough_params = None
        self.extension_params = None
        
        # Reader settings
        self.instrument_reader_settings = {
            "low_text": 0.3,
            "min_size": 10,
            "ycenter_ths": 0.5,
            "height_ths": 0.5,
            "width_ths": 6.0,
            "add_margin": -0.1,
            "link_threshold": 0.2,
            "text_threshold": 0.3,
            "mag_ratio": 3.0,
            "allowlist": '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ()',
            "decoder": 'beamsearch',
            "batch_size": 1
        }
        
        self.reader_settings = {
            "low_text": 0.4,
            "min_size": 10,
            "ycenter_ths": 0.5,
            "height_ths": 0.5,
            "width_ths": 0.0,
            "add_margin": 0.1,
            "link_threshold": 0.13,
            "text_threshold": 0.3,
            "mag_ratio": 1.0,
            "allowlist": '',
            "decoder": 'beamsearch',
            "batch_size": 1
        }
        
        # Application settings
        self.write_mode = 'xlwings'
        self.annotations_visible = False
    
    def load(self, folder_path: str = None) -> bool:
        """
        Load configuration from a JSON file in the specified folder.
        
        Args:
            folder_path: Optional path to override the current folder_path
            
        Returns:
            bool: True if configuration was loaded successfully, False otherwise
        """
        if folder_path:
            self.folder_path = folder_path
            
        if not self.folder_path:
            print("No folder path specified for loading configuration")
            return False
            
        config_file = os.path.join(self.folder_path, 'config.json')
        
        # Fall back to attributes.json if config.json doesn't exist (for backward compatibility)
        if not os.path.exists(config_file):
            config_file = os.path.join(self.folder_path, 'attributes.json')
            if not os.path.exists(config_file):
                print(f"Configuration file not found at {config_file}")
                return False
        
        try:
            with open(config_file, 'r') as file:
                config_data = json.load(file)
                
            # Update attributes from loaded data
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
            print(f"Configuration loaded from {config_file}")
            return True
            
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {config_file}")
            return False
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return False
    
    def save(self, folder_path: str = None) -> bool:
        """
        Save configuration to a JSON file in the specified folder.
        
        Args:
            folder_path: Optional path to override the current folder_path
            
        Returns:
            bool: True if configuration was saved successfully, False otherwise
        """
        if folder_path:
            self.folder_path = folder_path
            
        if not self.folder_path:
            print("No folder path specified for saving configuration")
            return False
        
        config_file = os.path.join(self.folder_path, 'config.json')
        backup_file = os.path.join(
            self.folder_path, 
            f'config_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        # Create a backup of the existing config if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            except Exception as e:
                print(f"Warning: Failed to create backup of configuration: {str(e)}")
        
        try:
            # Get all instance variables that don't start with underscore
            config_data = {
                key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and key != 'folder_path'
            }
            
            with open(config_file, 'w') as file:
                json.dump(config_data, file, indent=2)
                
            print(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key name.
        
        Args:
            key: The configuration parameter name
            default: Value to return if key doesn't exist
            
        Returns:
            The configuration value or default if not found
        """
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value by key name.
        
        Args:
            key: The configuration parameter name
            value: The value to set
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        if hasattr(self, key):
            setattr(self, key, value)
            return True
        return False
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.
        
        Args:
            config_dict: Dictionary of configuration parameters to update
        """
        for key, value in config_dict.items():
            self.set(key, value) 