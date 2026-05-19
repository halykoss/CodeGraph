"""
Unit tests for the graph builder module.
"""

import sys
import os
import unittest
from src.graph_builder import GraphBuilder

class TestGraphBuilder(unittest.TestCase):
    """Test cases for the GraphBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = GraphBuilder()
    
    def test_graph_initialization(self):
        """Test that the graph is initialized correctly."""
        graph = self.builder.get_graph()
        self.assertIsNotNone(graph)
        self.assertEqual(len(graph.nodes()), 0)
        self.assertEqual(len(graph.edges()), 0)
    
    def test_add_file_node(self):
        """Test adding a file node to the graph."""
        self.builder.add_file_node("test.py", {"language": "python"})
        graph = self.builder.get_graph()
        self.assertIn("test.py", graph.nodes())
        self.assertEqual(graph.nodes["test.py"]["type"], "file")
        self.assertEqual(graph.nodes["test.py"]["language"], "python")
    
    def test_add_domain_node(self):
        """Test adding a domain node to the graph."""
        self.builder.add_domain_node("bioinformatics", {"description": "Biological data analysis"})
        graph = self.builder.get_graph()
        self.assertIn("bioinformatics", graph.nodes())
        self.assertEqual(graph.nodes["bioinformatics"]["type"], "domain")
        self.assertEqual(graph.nodes["bioinformatics"]["description"], "Biological data analysis")
    
    def test_add_function_node(self):
        """Test adding a function node to the graph."""
        self.builder.add_function_node("algorithm", {"complexity": "O(n log n)"})
        graph = self.builder.get_graph()
        self.assertIn("algorithm", graph.nodes())
        self.assertEqual(graph.nodes["algorithm"]["type"], "function")
        self.assertEqual(graph.nodes["algorithm"]["complexity"], "O(n log n)")
    
    def test_add_algorithm_node(self):
        """Test adding an algorithm node to the graph."""
        self.builder.add_algorithm_node("fourier_transform", {"type": "signal_processing"})
        graph = self.builder.get_graph()
        self.assertIn("fourier_transform", graph.nodes())
        self.assertEqual(graph.nodes["fourier_transform"]["type"], "algorithm")
        self.assertEqual(graph.nodes["fourier_transform"]["type"], "algorithm")
    
    def test_add_data_type_node(self):
        """Test adding a data type node to the graph."""
        self.builder.add_data_type_node("integer", {"category": "primitive"})
        graph = self.builder.get_graph()
        self.assertIn("integer", graph.nodes())
        self.assertEqual(graph.nodes["integer"]["type"], "data_type")
        self.assertEqual(graph.nodes["integer"]["category"], "primitive")
    
    def test_add_programming_paradigm_node(self):
        """Test adding a programming paradigm node to the graph."""
        self.builder.add_programming_paradigm_node("functional", {"confidence": 0.95})
        graph = self.builder.get_graph()
        self.assertIn("functional", graph.nodes())
        self.assertEqual(graph.nodes["functional"]["type"], "programming_paradigm")
        self.assertEqual(graph.nodes["functional"]["confidence"], 0.95)
    
    def test_add_design_pattern_node(self):
        """Test adding a design pattern node to the graph."""
        self.builder.add_design_pattern_node("factory", {"category": "creational"})
        graph = self.builder.get_graph()
        self.assertIn("factory", graph.nodes())
        self.assertEqual(graph.nodes["factory"]["type"], "design_pattern")
        self.assertEqual(graph.nodes["factory"]["category"], "creational")
    
    def test_add_data_structure_node(self):
        """Test adding a data structure node to the graph."""
        self.builder.add_data_structure_node("binary_tree", {"category": "tree"})
        graph = self.builder.get_graph()
        self.assertIn("binary_tree", graph.nodes())
        self.assertEqual(graph.nodes["binary_tree"]["type"], "data_structure")
        self.assertEqual(graph.nodes["binary_tree"]["category"], "tree")
    
    def test_add_edge(self):
        """Test adding an edge to the graph."""
        self.builder.add_file_node("test.py")
        self.builder.add_domain_node("bioinformatics")
        self.builder.add_edge("test.py", "bioinformatics", "belongs_to_domain")
        
        graph = self.builder.get_graph()
        self.assertIn(("test.py", "bioinformatics"), graph.edges())
        self.assertEqual(graph.edges[("test.py", "bioinformatics")]["type"], "belongs_to_domain")
    
    def test_add_belongs_to_domain_edge(self):
        """Test adding a belongs_to_domain edge."""
        self.builder.add_file_node("bio/file.py")
        self.builder.add_domain_node("bioinformatics")
        self.builder.add_belongs_to_domain_edge("bio/file.py", "bioinformatics")
        
        graph = self.builder.get_graph()
        self.assertIn(("bio/file.py", "bioinformatics"), graph.edges())
        self.assertEqual(graph.edges[("bio/file.py", "bioinformatics")]["type"], "belongs_to_domain")
    
    def test_add_implements_function_edge(self):
        """Test adding an implements_function edge."""
        self.builder.add_file_node("sort.py")
        self.builder.add_function_node("algorithm")
        self.builder.add_implements_function_edge("sort.py", "algorithm")
        
        graph = self.builder.get_graph()
        self.assertIn(("sort.py", "algorithm"), graph.edges())
        self.assertEqual(graph.edges[("sort.py", "algorithm")]["type"], "implements_function")
    
    def test_add_uses_algorithm_edge(self):
        """Test adding a uses_algorithm edge."""
        self.builder.add_file_node("signal.py")
        self.builder.add_algorithm_node("fourier_transform")
        self.builder.add_uses_algorithm_edge("signal.py", "fourier_transform")
        
        graph = self.builder.get_graph()
        self.assertIn(("signal.py", "fourier_transform"), graph.edges())
        self.assertEqual(graph.edges[("signal.py", "fourier_transform")]["type"], "uses_algorithm")
    
    def test_add_uses_data_type_edge(self):
        """Test adding a uses_data_type edge."""
        self.builder.add_file_node("program.py")
        self.builder.add_data_type_node("float")
        self.builder.add_uses_data_type_edge("program.py", "float")
        
        graph = self.builder.get_graph()
        self.assertIn(("program.py", "float"), graph.edges())
        self.assertEqual(graph.edges[("program.py", "float")]["type"], "uses_data_type")
    
    def test_add_uses_paradigm_edge(self):
        """Test adding a uses_paradigm edge."""
        self.builder.add_file_node("functional_code.py")
        self.builder.add_programming_paradigm_node("functional")
        self.builder.add_uses_paradigm_edge("functional_code.py", "functional")
        
        graph = self.builder.get_graph()
        self.assertIn(("functional_code.py", "functional"), graph.edges())
        self.assertEqual(graph.edges[("functional_code.py", "functional")]["type"], "uses_paradigm")
    
    def test_add_uses_design_pattern_edge(self):
        """Test adding a uses_design_pattern edge."""
        self.builder.add_file_node("singleton_impl.py")
        self.builder.add_design_pattern_node("singleton")
        self.builder.add_uses_design_pattern_edge("singleton_impl.py", "singleton")
        
        graph = self.builder.get_graph()
        self.assertIn(("singleton_impl.py", "singleton"), graph.edges())
        self.assertEqual(graph.edges[("singleton_impl.py", "singleton")]["type"], "uses_design_pattern")
    
    def test_add_uses_data_structure_edge(self):
        """Test adding a uses_data_structure edge."""
        self.builder.add_file_node("tree_sort.py")
        self.builder.add_data_structure_node("binary_tree")
        self.builder.add_uses_data_structure_edge("tree_sort.py", "binary_tree")
        
        graph = self.builder.get_graph()
        self.assertIn(("tree_sort.py", "binary_tree"), graph.edges())
        self.assertEqual(graph.edges[("tree_sort.py", "binary_tree")]["type"], "uses_data_structure")
    
    def test_add_similar_to_edge(self):
        """Test adding a similar_to edge."""
        self.builder.add_algorithm_node("fft")
        self.builder.add_algorithm_node("dft")
        self.builder.add_similar_to_edge("fft", "dft", 0.8)
        
        graph = self.builder.get_graph()
        self.assertIn(("fft", "dft"), graph.edges())
        self.assertEqual(graph.edges[("fft", "dft")]["type"], "similar_to")
        self.assertEqual(graph.edges[("fft", "dft")]["weight"], 0.8)

if __name__ == '__main__':
    unittest.main()
