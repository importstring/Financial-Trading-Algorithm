import aiohttp
import asyncio
from typing import Optional, Dict, Any
from urllib3.poolmanager import PoolManager
from openai import OpenAI
import logging
from ..utils.config_manager import ConfigManager
from ..utils.api_key_manager import APIKeyManager

class ConnectionManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config = ConfigManager()
        self.api_key_manager = APIKeyManager()
        
        # Get connection settings
        self.max_connections = self.config.get_int('Performance', 'max_connections')
        self.timeout = self.config.get_int('Performance', 'connection_timeout')
        self.max_retries = self.config.get_int('Performance', 'max_retries')
        
        # Initialize connection pools
        self._initialize_pools()
        self._initialized = True
    
    def _initialize_pools(self):
        """Initialize connection pools for different services."""
        # HTTP connection pool
        self.http_pool = PoolManager(
            num_pools=4,
            maxsize=self.max_connections,
            timeout=self.timeout,
            retries=self.max_retries
        )
        
        # Async session for non-OpenAI APIs
        self.async_session = None
        
        # OpenAI client with pooling
        openai_key = self.api_key_manager.get_key('openai')
        if openai_key:
            self.openai_client = self._create_pooled_openai_client(openai_key)
        else:
            self.openai_client = None
            logging.warning("OpenAI API key not found")
    
    def _create_pooled_openai_client(self, api_key: str) -> OpenAI:
        """Create OpenAI client with connection pooling."""
        return OpenAI(
            api_key=api_key,
            http_client=self.http_pool,
            max_retries=self.max_retries,
            timeout=self.timeout
        )
    
    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create an async session with connection pooling."""
        if self.async_session is None or self.async_session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            self.async_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=True
            )
        return self.async_session
    
    async def close_async_session(self):
        """Close the async session if it exists."""
        if self.async_session and not self.async_session.closed:
            await self.async_session.close()
            self.async_session = None
    
    def get_openai_client(self) -> Optional[OpenAI]:
        """Get the OpenAI client with connection pooling."""
        if self.openai_client is None:
            openai_key = self.api_key_manager.get_key('openai')
            if openai_key:
                self.openai_client = self._create_pooled_openai_client(openai_key)
            else:
                logging.warning("OpenAI API key not found")
        return self.openai_client
    
    async def make_request(self, url: str, method: str = 'GET', **kwargs) -> Dict[str, Any]:
        """Make an HTTP request using the connection pool."""
        session = await self.get_async_session()
        try:
            async with session.request(method, url, **kwargs) as response:
                return await response.json()
        except Exception as e:
            logging.error(f"Request failed: {e}")
            raise
    
    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, 'async_session') and self.async_session:
            asyncio.create_task(self.close_async_session()) 