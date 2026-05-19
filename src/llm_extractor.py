"""
LLM-based information extraction module for analyzing source code.
Uses Ollama for local LLM inference with models like Qwen3-Coder.
"""

import json
import logging
from typing import Dict, List, Any, Optional
import ollama
from src.llm_config import get_model_config, get_fallback_config, get_ollama_options, get_ollama_client_config

from src.llm_prompts import (
    SYSTEM_PROMPT,
    DOMAIN_ENTITIES_PROMPT,
    ALGORITHMS_PROMPT,
    PARADIGMS_AND_PATTERNS_PROMPT,
    DATA_STRUCTURES_PROMPT
)


class LLMExtractor:
    """Extracts domain-specific information from source code using Ollama."""

    def __init__(self, model_name=None, use_config=True):
        """
        Initialize the LLM extractor with Ollama.

        Args:
            model_name (str, optional): Name of the Ollama model to use for extraction
            use_config (bool): Whether to use configuration from llm_config

        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if use_config:
            self.config = get_model_config()

        else:
            self.config = {
                "model_name": model_name or "qwen2.5-coder:7b", "host": "http://localhost:11434"}

        self.model_name = model_name or self.config["model_name"]

        self.logger.info(
            f"Initializing Ollama Extractor with model: {self.model_name}")

        # Initialize Ollama client
        self.client = None
        self.ollama_options = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Ollama client with error handling and fallback."""
        try:
            # Initialize Ollama client
            client_config = get_ollama_client_config(self.config)
            self.client = ollama.Client(**client_config)

            # Test connection by trying to pull/check the model
            self._check_model_availability(self.model_name)
            self.ollama_options = get_ollama_options(self.config)

        except Exception as e:
            self.logger.warning(
                f"Failed to initialize primary model {self.model_name}: {e}")

            # Try fallback if enabled
            if self.config.get("enable_fallback", True):
                self.logger.info("Attempting to use fallback model...")
                try:
                    fallback_config = get_fallback_config()
                    fallback_client_config = get_ollama_client_config(
                        fallback_config)
                    self.client = ollama.Client(**fallback_client_config)

                    fallback_model = fallback_config["model_name"]
                    self._check_model_availability(fallback_model)
                    self.model_name = fallback_model
                    self.ollama_options = get_ollama_options(fallback_config)
                    self.logger.info(
                        f"Successfully initialized fallback model: {fallback_model}")

                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback model also failed: {fallback_error}")
                    self.client = None
            else:
                self.client = None

    def _check_model_availability(self, model_name: str):
        """
        Check if the model is available in Ollama, and pull it if necessary.

        Args:
            model_name (str): Name of the model to check
        """
        try:
            # Try to get model information
            self.client.show(model_name)
            self.logger.info(f"Model {model_name} is available")
        except Exception:
            # Model not found, try to pull it
            self.logger.info(
                f"Model {model_name} not found locally, attempting to pull...")
            try:
                self.client.pull(model_name)
                self.logger.info(f"Successfully pulled model: {model_name}")
            except Exception as pull_error:
                self.logger.error(
                    f"Failed to pull model {model_name}: {pull_error}")
                raise pull_error

    def _generate_response(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate response from the LLM using Ollama.

        Args:
            prompt (str): The prompt to send to the model
            max_new_tokens (int, optional): Maximum number of tokens to generate

        Returns:
            str: The model's response
        """
        if self.client is None:
            self.logger.warning(
                "Ollama client not initialized, returning placeholder response")
            return "{}"

        try:
            # Prepare the conversation format
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            # Prepare generation options
            options = self.ollama_options.copy() if self.ollama_options else {}
            if max_new_tokens is not None:
                options["num_predict"] = max_new_tokens

            # Generate response using Ollama chat API
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options
            )

            # Extract the message content
            if response and 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            else:
                self.logger.warning("Unexpected response format from Ollama")
                return "{}"

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return "{}"

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from the model, handling potential formatting issues.

        Args:
            response (str): The raw response from the model

        Returns:
            dict: Parsed JSON response or empty dict if parsing fails
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, return empty dict
                return {}

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return {}

    def extract_entities(self, code_content: str) -> Dict[str, Any]:
        """
        Extract domain entities from source code.

        Args:
            code_content (str): The source code content to analyze

        Returns:
            dict: Extracted entities with their metadata
        """
        prompt = DOMAIN_ENTITIES_PROMPT.format(code=code_content)
        response = self._generate_response(prompt)
        parsed_response = self._parse_json_response(response)

        # Ensure the response has the expected structure
        if "entities" not in parsed_response:
            parsed_response["entities"] = []
        if "domain" not in parsed_response:
            parsed_response["domain"] = "unknown"

        return {
            "entities": parsed_response["entities"],
            "metadata": {"domain": parsed_response["domain"]}
        }

    def extract_algorithms(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Extract algorithms from source code.

        Args:
            code_content (str): The source code content to analyze

        Returns:
            list: List of identified algorithms with metadata
        """
        prompt = ALGORITHMS_PROMPT.format(code=code_content)
        response = self._generate_response(prompt)
        parsed_response = self._parse_json_response(response)

        return parsed_response.get("algorithms", [])

    def extract_paradigms_and_patterns(self, code_content: str) -> Dict[str, Any]:
        """
        Extract programming paradigms and design patterns from source code.

        Args:
            code_content (str): The source code content to analyze

        Returns:
            dict: Dictionary containing lists of paradigms and design_patterns
        """
        prompt = PARADIGMS_AND_PATTERNS_PROMPT.format(code=code_content)
        response = self._generate_response(prompt)
        parsed_response = self._parse_json_response(response)

        return {
            "paradigms": parsed_response.get("paradigms", []),
            "design_patterns": parsed_response.get("design_patterns", [])
        }

    def extract_data_structures(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Extract data structures from source code.

        Args:
            code_content (str): The source code content to analyze

        Returns:
            list: List of identified data structures with metadata
        """
        prompt = DATA_STRUCTURES_PROMPT.format(code=code_content)
        response = self._generate_response(prompt)
        parsed_response = self._parse_json_response(response)

        return parsed_response.get("data_structures", [])
