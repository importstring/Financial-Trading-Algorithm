import os
import json
from pathlib import Path
from typing import Optional, Dict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging
from .config_manager import ConfigManager

class APIKeyManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIKeyManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = ConfigManager()
        self.api_keys_path = self.config.get_path('Paths', 'api_keys')
        self.api_keys_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self._initialize_encryption()
        self._load_keys()
        self._initialized = True
    
    def _initialize_encryption(self):
        """Initialize encryption with a secure key derivation function."""
        # Use environment variable for master key, or generate one if not exists
        master_key = os.environ.get('API_MASTER_KEY')
        if not master_key:
            master_key = base64.b64encode(os.urandom(32)).decode('utf-8')
            logging.warning("No master key found in environment. Generated new key.")
            
        # Generate salt if not exists
        salt_path = self.api_keys_path / '.salt'
        if not salt_path.exists():
            salt = os.urandom(16)
            with open(salt_path, 'wb') as f:
                f.write(salt)
        else:
            with open(salt_path, 'rb') as f:
                salt = f.read()
        
        # Create key derivation function
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # Generate encryption key
        key = base64.b64encode(kdf.derive(master_key.encode()))
        self.cipher_suite = Fernet(key)
    
    def _load_keys(self):
        """Load encrypted API keys from storage."""
        self.keys = {}
        keys_file = self.api_keys_path / 'encrypted_keys.json'
        
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    encrypted_data = json.load(f)
                    
                for service, encrypted_key in encrypted_data.items():
                    try:
                        decrypted_key = self.cipher_suite.decrypt(encrypted_key.encode()).decode()
                        self.keys[service] = decrypted_key
                    except Exception as e:
                        logging.error(f"Failed to decrypt key for {service}: {e}")
            except Exception as e:
                logging.error(f"Failed to load API keys: {e}")
    
    def get_key(self, service: str) -> Optional[str]:
        """Get decrypted API key for a service."""
        return self.keys.get(service)
    
    def set_key(self, service: str, key: str) -> bool:
        """Securely store an API key."""
        try:
            # Encrypt the key
            encrypted_key = self.cipher_suite.encrypt(key.encode()).decode()
            self.keys[service] = key
            
            # Save to file
            keys_file = self.api_keys_path / 'encrypted_keys.json'
            encrypted_data = {}
            
            if keys_file.exists():
                with open(keys_file, 'r') as f:
                    encrypted_data = json.load(f)
            
            encrypted_data[service] = encrypted_key
            
            with open(keys_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
            
            return True
        except Exception as e:
            logging.error(f"Failed to set API key for {service}: {e}")
            return False
    
    def delete_key(self, service: str) -> bool:
        """Delete an API key."""
        try:
            if service in self.keys:
                del self.keys[service]
                
                keys_file = self.api_keys_path / 'encrypted_keys.json'
                if keys_file.exists():
                    with open(keys_file, 'r') as f:
                        encrypted_data = json.load(f)
                    
                    if service in encrypted_data:
                        del encrypted_data[service]
                        
                        with open(keys_file, 'w') as f:
                            json.dump(encrypted_data, f, indent=2)
                
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to delete API key for {service}: {e}")
            return False
    
    def rotate_keys(self) -> bool:
        """Rotate encryption key and re-encrypt all API keys."""
        try:
            # Store current keys
            current_keys = self.keys.copy()
            
            # Generate new encryption key
            self._initialize_encryption()
            
            # Re-encrypt all keys with new cipher
            self.keys = {}
            for service, key in current_keys.items():
                if not self.set_key(service, key):
                    raise Exception(f"Failed to re-encrypt key for {service}")
            
            return True
        except Exception as e:
            logging.error(f"Failed to rotate API keys: {e}")
            return False 