"""
Gemini OpenAI-compatible wrapper using LiteLLM.

This wrapper allows Google Gemini to work with MarkItDown's llm_client interface
by providing an OpenAI-compatible API.

Example usage:
    from gemini_wrapper import create_gemini_client
    
    client = create_gemini_client(api_key="your-gemini-api-key")
    md = MarkItDown(llm_client=client, llm_model="gemini-1.5-flash")
    result = md.convert("image.jpg")
    print(result.text_content)
"""

import os
from typing import Optional, Dict, Any, List
import litellm


class ChatCompletion:
    """
    Mimics OpenAI's ChatCompletion interface to work with Gemini via LiteLLM.
    """

    def __init__(self, api_key: str):
        """
        Initialize ChatCompletion wrapper.
        
        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        litellm.api_key = api_key

    def create(
        self,
        model: str = "gemini-1.5-flash",
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Gemini via LiteLLM.
        
        Args:
            model: Model name (e.g., "gemini-1.5-flash", "gemini-pro")
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional parameters to pass to LiteLLM
        
        Returns:
            Response object in OpenAI-compatible format with structure:
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "response text"
                        }
                    }
                ]
            }
        
        Raises:
            RuntimeError: If the Gemini API call fails
        """
        if messages is None:
            messages = []

        # Build request parameters for LiteLLM
        request_params = {
            "model": f"gemini/{model}",  # LiteLLM requires 'gemini/' prefix
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        # Add any additional parameters
        request_params.update(kwargs)

        try:
            # Call Gemini via LiteLLM
            response = litellm.completion(**request_params)
            
            # Ensure response is in OpenAI-compatible format
            return response
            
        except litellm.exceptions.APIError as e:
            raise RuntimeError(
                f"Gemini API error: {str(e)}. "
                f"Make sure your API key is valid and you have quota available."
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Gemini: {str(e)}")


class GeminiOpenAICompatibleClient:
    """
    A wrapper that makes Google Gemini compatible with OpenAI's Python client interface.
    
    This allows Gemini to be used as a drop-in replacement for OpenAI in MarkItDown
    and other applications that expect an OpenAI-compatible client.
    
    Example:
        client = GeminiOpenAICompatibleClient(api_key="your-key")
        # Use with MarkItDown
        md = MarkItDown(llm_client=client, llm_model="gemini-1.5-flash")
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GOOGLE_API_KEY 
                    environment variable.
        
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not provided. Either pass api_key parameter "
                    "or set GOOGLE_API_KEY environment variable. "
                    "Get your key at: https://aistudio.google.com/app/apikeys"
                )
        
        self.api_key = api_key
        
        # Initialize nested Chat object to mimic OpenAI client structure
        # OpenAI client has client.chat.completions.create()
        # We're creating client.chat.create() which works with MarkItDown
        self.chat = ChatCompletion(self.api_key)


def create_gemini_client(api_key: Optional[str] = None) -> GeminiOpenAICompatibleClient:
    """
    Create a Gemini OpenAI-compatible client.
    
    This is a convenience function for creating a Gemini client. It's functionally
    equivalent to directly instantiating GeminiOpenAICompatibleClient.
    
    Args:
        api_key: Optional Gemini API key. If not provided, will use GOOGLE_API_KEY
                environment variable.
    
    Returns:
        GeminiOpenAICompatibleClient instance
    
    Raises:
        ValueError: If no API key is available
    
    Example:
        # Using environment variable
        client = create_gemini_client()
        
        # Using explicit API key
        client = create_gemini_client(api_key="your-key")
    """
    return GeminiOpenAICompatibleClient(api_key=api_key)


# Supported Gemini models (as of the current LiteLLM version)
SUPPORTED_GEMINI_MODELS = [
    "gemini-1.5-flash",      # Fastest, good for most tasks
    "gemini-1.5-pro",        # More capable, better quality
    "gemini-1.0-pro",        # Older model, still available
    "gemini-1.0-pro-vision", # Vision capabilities
]
