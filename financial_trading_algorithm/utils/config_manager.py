import configparser
import os
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = configparser.ConfigParser()
        self.project_root = self._get_project_root()
        self._load_config()
        self._initialized = True
    
    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / '.git').exists():
                return parent
            if parent.name == 'Financial-Trading-Algorithm-1':
                return parent
        raise FileNotFoundError("Could not find project root directory")
    
    def _load_config(self):
        """Load configuration from config.ini file."""
        config_path = self.project_root / 'financial_trading_algorithm' / 'config.ini'
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        self.config.read(config_path)
        
        # Replace ${PROJECT_ROOT} with actual path in all values
        for section in self.config.sections():
            for key, value in self.config.items(section):
                if '${PROJECT_ROOT}' in value:
                    self.config[section][key] = value.replace('${PROJECT_ROOT}', str(self.project_root))
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get a configuration value."""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return fallback
            raise KeyError(f"Configuration not found for section '{section}' key '{key}'")
    
    def get_path(self, section: str, key: str) -> Path:
        """Get a path configuration value as a Path object."""
        path_str = self.get(section, key)
        return Path(path_str).resolve()
    
    def get_int(self, section: str, key: str, fallback: Optional[int] = None) -> int:
        """Get an integer configuration value."""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return fallback
            raise KeyError(f"Configuration not found for section '{section}' key '{key}'")
    
    def get_float(self, section: str, key: str, fallback: Optional[float] = None) -> float:
        """Get a float configuration value."""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return fallback
            raise KeyError(f"Configuration not found for section '{section}' key '{key}'")
    
    def get_bool(self, section: str, key: str, fallback: Optional[bool] = None) -> bool:
        """Get a boolean configuration value."""
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return fallback
            raise KeyError(f"Configuration not found for section '{section}' key '{key}'")
    
    def get_dict(self, section: str) -> Dict[str, str]:
        """Get all key-value pairs in a section as a dictionary."""
        try:
            return dict(self.config.items(section))
        except configparser.NoSectionError:
            raise KeyError(f"Configuration section '{section}' not found") 