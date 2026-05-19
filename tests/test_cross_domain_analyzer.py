"""
Unit tests for the cross-domain analyzer module.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from src.cross_domain_analyzer import CrossDomainAnalyzer
from src.graph_builder import GraphBuilder

class TestCrossDomainAnalyzer(unittest.TestCase):
    """Test cases for the CrossDomainAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a graph builder and add some test data
        self.graph_builder = GraphBuilder()
        
        # Add domains
        self.graph_builder.add_domain_node("bioinformatics")
        self.graph_builder.add_domain_node("image_processing")
        self.graph_builder.add_domain_node("telecommunications")
        
        # Add algorithms
        self.graph_builder.add_algorithm_node("fourier_transform")
        self.graph_builder.add_algorithm_node("convolution")
        
        # Add data structures
        self.graph_builder.add_data_structure_node("array")
        self.graph_builder.add_data_structure_node("linked_list")
        
        # Add files
        self.graph_builder.add_file_node("bio/file1.py")
        self.graph_builder.add_file_node("img/file2.py")
        self.graph_builder.add_file_node("telecom/file3.py")
        
        # Add relationships
        self.graph_builder.add_belongs_to_domain_edge("bio/file1.py", "bioinformatics")
        self.graph_builder.add_belongs_to_domain_edge("img/file2.py", "image_processing")
        self.graph_builder.add_belongs_to_domain_edge("telecom/file3.py", "telecommunications")
        
        self.graph_builder.add_uses_algorithm_edge("bio/file1.py", "fourier_transform")
        self.graph_builder.add_uses_algorithm_edge("img/file2.py", "fourier_transform")
        self.graph_builder.add_uses_algorithm_edge("img/file2.py", "convolution")
        self.graph_builder.add_uses_algorithm_edge("telecom/file3.py", "fourier_transform")
        
        # Add data structure relationships
        self.graph_builder.add_uses_data_structure_edge("bio/file1.py", "array")
        self.graph_builder.add_uses_data_structure_edge("img/file2.py", "array")
        self.graph_builder.add_uses_data_structure_edge("telecom/file3.py", "linked_list")
        
        # Create analyzer with the graph
        self.analyzer = CrossDomainAnalyzer(self.graph_builder.get_graph())
    
    def test_initialization(self):
        """Test that the analyzer is initialized correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.graph)
    
    def test_find_algorithm_transfer(self):
        """Test finding algorithm transfers across domains."""
        result = self.analyzer.find_algorithm_transfer("fourier_transform")
        self.assertEqual(result["algorithm"], "fourier_transform")
        self.assertEqual(result["total_usage"], 3)
        self.assertIn("bioinformatics", result["domains"])
        self.assertIn("image_processing", result["domains"])
        self.assertIn("telecommunications", result["domains"])
    
    def test_find_shared_patterns(self):
        """Test finding shared patterns between domains."""
        result = self.analyzer.find_shared_patterns()
        # Should find that domains share algorithms and data structures
        # In our test case, all domains use fourier_transform, and bio+img share array
        
        # Check for shared patterns
        found_shared = False
        for domain_pair, patterns in result.items():
            if patterns.get('algorithms') and "fourier_transform" in patterns['algorithms']:
                found_shared = True
                break
        
        # Domains should share fourier_transform
        self.assertTrue(found_shared)
    
    def test_find_domain_similarities(self):
        """Test finding domain similarities."""
        result = self.analyzer.find_domain_similarities()
        # All domains share the fourier_transform algorithm, so they should have some similarity
        self.assertIsInstance(result, dict)
        # Should have similarities between all pairs
        self.assertGreater(len(result), 0)
    
    def test_get_cross_domain_insights(self):
        """Test getting comprehensive cross-domain insights."""
        result = self.analyzer.get_cross_domain_insights()
        self.assertIn("algorithm_transfers", result)
        self.assertIn("shared_patterns", result)
        self.assertIn("domain_similarities", result)
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()
