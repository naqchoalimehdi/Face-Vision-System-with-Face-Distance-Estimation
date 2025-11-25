"""Configuration loader with YAML and environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self.config = self._load_yaml_config()
        self._override_with_env_vars()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        # Try to find config relative to project root
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_file = project_root / "config" / "default_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Default config not found at {config_file}. "
                "Please provide config_path or create default_config.yaml"
            )
        
        return config_file
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    def _override_with_env_vars(self):
        """Override config values with environment variables."""
        # Map environment variables to config keys
        env_mappings = {
            'HOST': ('api', 'host'),
            'PORT': ('api', 'port'),
            'DEBUG': ('api', 'debug'),
            'YOLO_MODEL': ('detection', 'model_path'),
            'CONFIDENCE_THRESHOLD': ('detection', 'confidence_threshold'),
            'IOU_THRESHOLD': ('detection', 'iou_threshold'),
            'DEFAULT_CAMERA_ID': ('camera', 'default_id'),
            'FRAME_WIDTH': ('camera', 'width'),
            'FRAME_HEIGHT': ('camera', 'height'),
            'FPS': ('camera', 'fps'),
            'AVERAGE_FACE_WIDTH_CM': ('distance', 'average_face_width_cm'),
            'DISTANCE_METHOD': ('distance', 'method'),
        }
        
        for env_var, config_keys in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_keys, self._convert_type(value))
    
    def _set_nested_value(self, keys: tuple, value: Any):
        """Set a nested dictionary value using a tuple of keys."""
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get('detection', 'confidence_threshold')
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'detection', 'tracking')
            
        Returns:
            Dictionary of section configuration
        """
        return self.config.get(section, {})
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base: Dict, updates: Dict):
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save config. If None, overwrites original.
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"


# Global config instance
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file. Only used on first call.
        
    Returns:
        ConfigLoader instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    
    return _global_config


def reset_config():
    """Reset global configuration instance."""
    global _global_config
    _global_config = None
