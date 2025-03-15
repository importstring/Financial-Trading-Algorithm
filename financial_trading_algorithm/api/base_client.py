from typing import Optional, Dict, Any, Tuple
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..utils.config_manager import ConfigManager
from ..utils.api_key_manager import APIKeyManager
from .connection_manager import ConnectionManager

class BaseAPIClient:
    """Base class for API clients with common functionality."""
    
    def __init__(self, service_name: str):
        """
        Initialize base API client.
        
        Args:
            service_name: Name of the service (e.g., 'openai', 'perplexity')
        """
        self.service_name = service_name
        self.config = ConfigManager()
        self.api_key_manager = APIKeyManager()
        self.connection_manager = ConnectionManager()
        
        # Get API key
        self.api_key = self.api_key_manager.get_key(service_name)
        if not self.api_key:
            logging.error(f"No API key found for {service_name}")
            raise ValueError(f"No API key found for {service_name}")
        
        # Set up logging
        self.logger = logging.getLogger(f"{service_name}_client")
        self.logger.setLevel(logging.INFO)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _make_request(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with retries and error handling.
        
        Args:
            url: API endpoint URL
            method: HTTP method
            headers: Request headers
            json_data: JSON request body
            params: URL parameters
            
        Returns:
            API response as dictionary
        """
        if headers is None:
            headers = {}
        headers['Authorization'] = f"Bearer {self.api_key}"
        
        try:
            response = await self.connection_manager.make_request(
                url=url,
                method=method,
                headers=headers,
                json=json_data,
                params=params
            )
            return response
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def _validate_response(self, response: Dict[str, Any], required_keys: set) -> bool:
        """
        Validate API response has required keys.
        
        Args:
            response: API response dictionary
            required_keys: Set of required keys
            
        Returns:
            True if valid, False otherwise
        """
        return required_keys.issubset(response.keys())
    
    async def _handle_rate_limit(
        self,
        func,
        *args,
        max_retries: int = 3,
        initial_wait: float = 1.0,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Handle rate limiting with exponential backoff.
        
        Args:
            func: Async function to call
            *args: Function arguments
            max_retries: Maximum number of retries
            initial_wait: Initial wait time in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, success)
        """
        retries = 0
        wait_time = initial_wait
        
        while retries < max_retries:
            try:
                result = await func(*args, **kwargs)
                return result, True
            except Exception as e:
                if 'rate_limit' in str(e).lower():
                    retries += 1
                    if retries < max_retries:
                        self.logger.warning(
                            f"Rate limit hit, waiting {wait_time}s before retry {retries}/{max_retries}"
                        )
                        await asyncio.sleep(wait_time)
                        wait_time *= 2
                    continue
                self.logger.error(f"API call failed: {e}")
                return None, False
        
        self.logger.error("Max retries reached for rate limit")
        return None, False
    
    def _format_error(self, error: Exception) -> Dict[str, str]:
        """
        Format error for consistent error handling.
        
        Args:
            error: Exception object
            
        Returns:
            Dictionary with error details
        """
        return {
            'error': str(error),
            'type': error.__class__.__name__,
            'service': self.service_name
        }
    
    async def close(self):
        """Close any open connections."""
        await self.connection_manager.close_async_session()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            asyncio.create_task(self.close())
        except Exception:
            pass 