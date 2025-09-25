"""
API model wrapper for KnowMT-Bench
Supports GPT-4, GPT-4o-mini, DeepSeek-R1 and other API-based models
"""

import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIModel:
    """API model wrapper for loading and running inference with API-based models"""
    
    SUPPORTED_MODELS = {
        "gpt-4o": {
            "model_id": "gpt-4o",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1"
        },
        "gpt-4o-mini": {
            "model_id": "gpt-4o-mini",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1"
        },
        "deepseek-r1": {
            "model_id": "Pro/deepseek-ai/DeepSeek-R1",
            "provider": "siliconflow",
            "base_url": "https://api.siliconflow.cn/v1",
            "has_cot": True
        },
        "deepseek-v3": {
            "model_id": "deepseek-ai/DeepSeek-V3",
            "provider": "siliconflow",
            "base_url": "https://api.siliconflow.cn/v1"
        },
        "gemini-2.5-pro": {
            "model_id": "gemini-2.5-pro",
            "provider": "google",
            "has_cot": True
        }
    }
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the API model
        
        Args:
            model_name: Key from SUPPORTED_MODELS dict
            api_key: API key for authentication (if None, will try to get from environment)
            base_url: Custom base URL (if None, will use model's default)
            **kwargs: Additional parameters
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model name: {model_name}. "
                           f"Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.config = self.SUPPORTED_MODELS[model_name]
        self.has_cot = self.config.get("has_cot", False)
        
        # Initialize appropriate client based on provider
        if self.config["provider"] == "google":
            self.client = genai.Client(api_key=api_key)
        else:
            # Use provided base_url or model's default
            final_base_url = base_url or self.config.get("base_url")
            self.client = OpenAI(
                api_key=api_key,
                base_url=final_base_url
            )
        
        logger.info(f"Initialized {model_name} API model with {self.config['provider']} provider")
    
    def _make_request(self, messages: List[Dict[str, str]], **kwargs):
        """Make API request using appropriate client"""
        try:
            if self.config["provider"] == "google":
                return self._make_gemini_request(messages, **kwargs)
            else:
                return self._make_openai_request(messages, **kwargs)
        except Exception as e:
            logger.error(f"API request error: {e}")
            raise

    def _make_openai_request(self, messages: List[Dict[str, str]], **kwargs):
        """Make OpenAI-compatible API request"""
        params = {
            "model": self.config["model_id"],
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_new_tokens", 1024)
        }

        # Add seed if supported
        if kwargs.get("seed"):
            params["seed"] = kwargs["seed"]

        # Special handling for DeepSeek R1
        if "deepseek-r1" in self.model_name.lower():
            params["extra_body"] = {"thinking_budget": kwargs.get("thinking_budget", 256)}
            params["stream"] = False

        return self.client.chat.completions.create(**params)

    def _make_gemini_request(self, messages: List[Dict[str, str]], **kwargs):
        """Make Google Gemini API request"""
        # Create chat for Gemini
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=kwargs.get("thinking_budget", 128)
            ),
            max_output_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 0),
            seed=kwargs.get("seed", 1024)
        )

        chat = self.client.chats.create(model=self.config["model_id"], config=config)

        # Get the last user message
        user_message = messages[-1]["content"] if messages else ""

        return chat.send_message(
            user_message,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )
    
    def generate_response(
        self,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate a single response to a prompt
        
        Args:
            prompt: Input prompt
            generation_config: Dict with generation parameters
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Merge generation config
        params = kwargs.copy()
        if generation_config:
            params.update(generation_config)
        
        response = self._make_request(messages, **params)
        
        if self.config["provider"] == "google":
            # Gemini response handling - extract only the final answer
            answers = []
            for part in response.candidates[0].content.parts:
                if not part.thought:  # Only non-thinking parts
                    answers.append(part.text)
            return "".join(answers)
        elif self.has_cot and "deepseek" in self.model_name.lower():
            # DeepSeek-R1 CoT handling - return only the final content
            message = response.choices[0].message
            return message.content or ""
        else:
            # Standard response
            return response.choices[0].message.content
    
    def generate_multi_turn(
        self,
        conversation: List[str],
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multi-turn conversation
        
        Args:
            conversation: List of user messages
            generation_config: Dict with generation parameters
            **kwargs: Additional generation parameters
            
        Returns:
            List of model responses
        """
        responses = []
        conversation_history = []
        
        # Merge generation config
        params = kwargs.copy()
        if generation_config:
            params.update(generation_config)
        
        for user_message in conversation:
            conversation_history.append(user_message)
            
            # Build messages format for API
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": m} 
                       for i, m in enumerate(conversation_history)]
            
            response = self._make_request(messages, **params)
            
            if self.config["provider"] == "google":
                # Gemini response handling - extract only the final answer
                answers = []
                for part in response.candidates[0].content.parts:
                    if not part.thought:  # Only non-thinking parts
                        answers.append(part.text)

                reply = "".join(answers)
                conversation_history.append(reply)
                responses.append(reply)
            elif self.has_cot and "deepseek" in self.model_name.lower():
                # DeepSeek-R1 CoT handling - return only the final content
                message = response.choices[0].message
                reply = message.content or ""

                conversation_history.append(reply)
                responses.append(reply)
            else:
                # Standard response
                reply = response.choices[0].message.content
                conversation_history.append(reply)
                responses.append(reply)
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_id": self.config["model_id"],
            "provider": self.config["provider"],
            "has_cot": self.has_cot,
            "base_url": self.config["base_url"]
        }
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """List all supported model keys"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    def __repr__(self):
        return f"APIModel(model_name='{self.model_name}', provider='{self.config['provider']}')"
