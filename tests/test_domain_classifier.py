"""
Unit tests for the domain classifier module.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from src.domain_classifier import DomainClassifier

class FakeExtractor:
    def _generate_response(self, prompt):
        return '{"domains": ["machine learning"], "confidence": 0.9}'

    def _parse_json_response(self, response):
        return {"domains": ["machine learning"], "confidence": 0.9}

class TestDomainClassifier(unittest.TestCase):
    """Test cases for the DomainClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = DomainClassifier(llm_extractor=FakeExtractor())
    
    def test_classify_domain_returns_list(self):
        """Test that classify_domain returns a list."""
        result = self.classifier.classify_domain("sample code")
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()
