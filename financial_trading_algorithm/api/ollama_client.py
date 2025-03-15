from typing import Dict, Any, Optional, List, Tuple
from .base_client import BaseAPIClient
import logging
import json
import asyncio
import aiohttp

class OllamaClient(BaseAPIClient):
    """Client for interacting with Ollama API."""
    
    def __init__(self):
        """Initialize Ollama client."""
        super().__init__('ollama')
        
        # Get model configuration
        self.model = self.config.get('API', 'ollama_model')
        
        # API endpoint (default to localhost)
        self.api_url = "http://localhost:11434/api"
    
    async def query(
        self,
        prompt: str,
        system_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query Ollama API with retry and error handling.
        
        Args:
            prompt: User prompt
            system_role: Optional system role prompt
            
        Returns:
            Dictionary with response and status
        """
        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._handle_rate_limit(
                self._make_completion_request,
                messages=messages
            )
            
            if response[1]:  # If successful
                return {
                    "content": response[0],
                    "status": True
                }
            return {
                "error": response[0],
                "status": False
            }
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Ollama query failed: {error_info}")
            return {"error": error_info, "status": False}
    
    async def _make_completion_request(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, bool]:
        """
        Make a completion request to Ollama.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (response text, success status)
        """
        try:
            # Format messages for Ollama
            prompt = "\n".join(msg["content"] for msg in messages)
            
            json_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = await self._make_request(
                url=f"{self.api_url}/generate",
                method='POST',
                json_data=json_data
            )
            
            if not response or 'response' not in response:
                raise ValueError("Invalid response from Ollama")
                
            return response['response'], True
            
        except Exception as e:
            self.logger.error(f"Ollama completion request failed: {e}")
            raise
    
    async def reason(self, query: str) -> Dict[str, Any]:
        """
        Use Ollama for chain-of-thought reasoning.
        
        Args:
            query: Query to reason about
            
        Returns:
            Dictionary with reasoning results
        """
        prompt = f"""
        Let's approach this step by step:
        
        Query: {query}
        
        1. First, let's break down the key components:
        2. Then, analyze each component:
        3. Consider relevant factors:
        4. Draw connections:
        5. Form conclusions:
        
        Provide your reasoning in this format:
        Components: <key components identified>
        Analysis: <analysis of each component>
        Factors: <relevant factors considered>
        Connections: <relationships identified>
        Conclusion: <final reasoning>
        Confidence: <number between 0 and 1>
        """
        
        try:
            response = await self.query(
                prompt=prompt,
                system_role="You are a logical reasoning expert."
            )
            
            if response["status"]:
                # Parse response into sections
                content = response["content"]
                sections = {}
                current_section = None
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if ':' in line and line.split(':')[0].strip() in [
                        'Components', 'Analysis', 'Factors',
                        'Connections', 'Conclusion', 'Confidence'
                    ]:
                        current_section = line.split(':')[0].strip()
                        sections[current_section] = line.split(':', 1)[1].strip()
                    elif current_section:
                        sections[current_section] += ' ' + line
                
                # Extract confidence score
                confidence = 0.5  # Default confidence
                if 'Confidence' in sections:
                    try:
                        confidence = float(sections['Confidence'])
                    except ValueError:
                        pass
                
                return {
                    "reasoning": sections,
                    "confidence": confidence,
                    "status": True
                }
            
            return {
                "error": "Failed to complete reasoning",
                "status": False
            }
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Reasoning failed: {error_info}")
            return {"error": error_info, "status": False}
    
    async def analyze_plan(self, query: str) -> Dict[str, Any]:
        """
        Generate and analyze a plan using Ollama.
        
        Args:
            query: Query to plan for
            
        Returns:
            Dictionary with plan analysis
        """
        prompt = f"""
        Let's create and analyze a plan for this query:
        
        {query}
        
        Follow these steps:
        1. Identify objectives
        2. Break down into tasks
        3. Analyze dependencies
        4. Assess risks
        5. Propose timeline
        
        Format your response as:
        Objectives: <key objectives>
        Tasks: <list of tasks>
        Dependencies: <task dependencies>
        Risks: <potential risks>
        Timeline: <estimated timeline>
        Priority: <high/medium/low>
        """
        
        try:
            response = await self.query(
                prompt=prompt,
                system_role="You are a strategic planning expert."
            )
            
            if response["status"]:
                # Parse response into sections
                content = response["content"]
                sections = {}
                current_section = None
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if ':' in line and line.split(':')[0].strip() in [
                        'Objectives', 'Tasks', 'Dependencies',
                        'Risks', 'Timeline', 'Priority'
                    ]:
                        current_section = line.split(':')[0].strip()
                        sections[current_section] = line.split(':', 1)[1].strip()
                    elif current_section:
                        sections[current_section] += ' ' + line
                
                return {
                    "plan": sections,
                    "status": True
                }
            
            return {
                "error": "Failed to generate plan",
                "status": False
            }
            
        except Exception as e:
            error_info = self._format_error(e)
            self.logger.error(f"Plan generation failed: {error_info}")
            return {"error": error_info, "status": False} 