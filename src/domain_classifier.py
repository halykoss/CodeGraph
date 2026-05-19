"""
Domain classification module for identifying application domains in source code.
"""

import json
import logging
from typing import List, Dict, Any
from src.llm_extractor import LLMExtractor
from src.llm_prompts import (
    DOMAIN_CLASSIFICATION_PROMPT,
    PERFORMANCE_CHARACTERISTICS_PROMPT
)

class DomainClassifier:
    """Classifies source code into application domains and technical functions."""
    
    # Performance characteristics
    PERFORMANCE_CHARACTERISTICS = [
        "time_complexity", "space_complexity", "throughput", "latency",
        "scalability", "reliability", "availability", "efficiency"
    ]
    
    def __init__(self, llm_extractor=None):
        """
        Initialize the domain classifier.
        
        Args:
            llm_extractor (LLMExtractor, optional): Instance of LLMExtractor to use
        """
        self.logger = logging.getLogger(__name__)
        
        if llm_extractor is None:
            self.logger.info("Initializing new LLM extractor for domain classification")
            self.llm_extractor = LLMExtractor()
        else:
            self.llm_extractor = llm_extractor
    
    def classify_domain(self, code_content: str) -> List[str]:
        """
        Classify source code into application domains.
        
        Args:
            code_content (str): The source code content to classify
            
        Returns:
            list: List of identified domains
        """
        try:
            prompt = DOMAIN_CLASSIFICATION_PROMPT.format(code=code_content)
            response = self.llm_extractor._generate_response(prompt)
            parsed_response = self.llm_extractor._parse_json_response(response)
            
            domains = parsed_response.get("domains", [])
            confidence = parsed_response.get("confidence", 0.0)
            
            self.logger.info(f"Domain classification result: {domains} (confidence: {confidence})")
            
            # Filter out domains that are not in our predefined list
            valid_domains = domains if isinstance(domains, list) else ["unknown"]
            
            return valid_domains if valid_domains else ["unknown"]
            
        except Exception as e:
            self.logger.error(f"Error in domain classification: {e}")
            return ["unknown"]
    
    def extract_performance_characteristics(self, code_content: str) -> List[Dict[str, Any]]:
        """
        Extract performance characteristics from source code.
        
        Args:
            code_content (str): The source code content to analyze
            
        Returns:
            list: List of identified performance characteristics with metadata
        """
        try:
            prompt = PERFORMANCE_CHARACTERISTICS_PROMPT.format(code=code_content)
            response = self.llm_extractor._generate_response(prompt)
            parsed_response = self.llm_extractor._parse_json_response(response)
            
            characteristics = parsed_response.get("characteristics", [])
            
            self.logger.info(f"Performance characteristics extraction result: {len(characteristics)} characteristics found")
            
            # Filter characteristics to ensure they have valid names
            valid_characteristics = []
            for char in characteristics:
                if isinstance(char, dict) and "name" in char:
                    if char["name"] in self.PERFORMANCE_CHARACTERISTICS:
                        valid_characteristics.append(char)
            
            return valid_characteristics
            
        except Exception as e:
            self.logger.error(f"Error in performance characteristics extraction: {e}")
            return []
