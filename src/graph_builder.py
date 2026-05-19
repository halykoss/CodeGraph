"""
Graph construction module for building knowledge graphs from code analysis.
"""

import networkx as nx

class GraphBuilder:
    """Builds knowledge graphs from code analysis results."""
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph = nx.DiGraph()
    
    def add_file_node(self, file_path, metadata=None):
        """
        Add a file node to the graph.
        
        Args:
            file_path (str): Path to the file
            metadata (dict): Metadata about the file
        """
        # Convert lists and dicts to strings for GraphML compatibility
        node_metadata = metadata or {}
        for key, value in node_metadata.items():
            if isinstance(value, list):
                node_metadata[key] = ", ".join(map(str, value))
            elif isinstance(value, dict):
                # Convert dict to string representation
                node_metadata[key] = str(value)
        self.graph.add_node(file_path, type="file", **node_metadata)
    
    def add_domain_node(self, domain, metadata=None):
        """
        Add a domain node to the graph.
        
        Args:
            domain (str): Domain name
            metadata (dict): Metadata about the domain
        """
        self.graph.add_node(domain, type="domain", **(metadata or {}))
    
    def add_function_node(self, function, metadata=None):
        """
        Add a technical function node to the graph.
        
        Args:
            function (str): Function name
            metadata (dict): Metadata about the function
        """
        self.graph.add_node(function, type="function", **(metadata or {}))
    
    def add_component_node(self, component, metadata=None):
        """
        Add a component node to the graph.
        
        Args:
            component (str): Component name
            metadata (dict): Metadata about the component
        """
        self.graph.add_node(component, type="component", **(metadata or {}))
    
    def add_algorithm_node(self, algorithm, metadata=None):
        """
        Add an algorithm node to the graph.
        
        Args:
            algorithm (str): Algorithm name
            metadata (dict): Metadata about the algorithm
        """
        node_metadata = metadata or {}
        node_metadata["type"] = "algorithm"
        self.graph.add_node(algorithm, **node_metadata)
    
    def add_data_type_node(self, data_type, metadata=None):
        """
        Add a data type node to the graph.
        
        Args:
            data_type (str): Data type name
            metadata (dict): Metadata about the data type
        """
        node_metadata = metadata or {}
        node_metadata["type"] = "data_type"
        self.graph.add_node(data_type, **node_metadata)
    
    def add_programming_paradigm_node(self, paradigm, metadata=None):
        """
        Add a programming paradigm node to the graph.
        
        Args:
            paradigm (str): Programming paradigm name
            metadata (dict): Metadata about the paradigm
        """
        node_metadata = metadata or {}
        node_metadata["type"] = "programming_paradigm"
        self.graph.add_node(paradigm, **node_metadata)
    
    def add_design_pattern_node(self, pattern, metadata=None):
        """
        Add a design pattern node to the graph.
        
        Args:
            pattern (str): Design pattern name
            metadata (dict): Metadata about the pattern
        """
        node_metadata = metadata or {}
        node_metadata["type"] = "design_pattern"
        self.graph.add_node(pattern, **node_metadata)
    
    def add_data_structure_node(self, data_structure, metadata=None):
        """
        Add a data structure node to the graph.
        
        Args:
            data_structure (str): Data structure name
            metadata (dict): Metadata about the data structure
        """
        node_metadata = metadata or {}
        node_metadata["type"] = "data_structure"
        self.graph.add_node(data_structure, **node_metadata)
    
    def add_edge(self, source, target, relationship, weight=1.0):
        """
        Add an edge between two nodes.
        
        Args:
            source (str): Source node
            target (str): Target node
            relationship (str): Type of relationship
            weight (float): Weight of the relationship
        """
        self.graph.add_edge(source, target, type=relationship, weight=weight)
    
    def add_belongs_to_domain_edge(self, file_path, domain):
        """
        Add a belongs_to_domain edge.
        
        Args:
            file_path (str): Path to the file
            domain (str): Domain name
        """
        self.add_edge(file_path, domain, "belongs_to_domain")
    
    def add_implements_function_edge(self, file_path, function):
        """
        Add an implements_function edge.
        
        Args:
            file_path (str): Path to the file
            function (str): Function name
        """
        self.add_edge(file_path, function, "implements_function")
    
    def add_uses_algorithm_edge(self, file_path, algorithm):
        """
        Add a uses_algorithm edge.
        
        Args:
            file_path (str): Path to the file
            algorithm (str): Algorithm name
        """
        self.add_edge(file_path, algorithm, "uses_algorithm")
    
    def add_uses_data_type_edge(self, file_path, data_type):
        """
        Add a uses_data_type edge.
        
        Args:
            file_path (str): Path to the file
            data_type (str): Data type name
        """
        self.add_edge(file_path, data_type, "uses_data_type")
    
    def add_uses_paradigm_edge(self, file_path, paradigm):
        """
        Add a uses_paradigm edge.
        
        Args:
            file_path (str): Path to the file
            paradigm (str): Programming paradigm name
        """
        self.add_edge(file_path, paradigm, "uses_paradigm")
    
    def add_uses_design_pattern_edge(self, file_path, pattern):
        """
        Add a uses_design_pattern edge.
        
        Args:
            file_path (str): Path to the file
            pattern (str): Design pattern name
        """
        self.add_edge(file_path, pattern, "uses_design_pattern")
    
    def add_uses_data_structure_edge(self, file_path, data_structure):
        """
        Add a uses_data_structure edge.
        
        Args:
            file_path (str): Path to the file
            data_structure (str): Data structure name
        """
        self.add_edge(file_path, data_structure, "uses_data_structure")
    
    def add_similar_to_edge(self, node1, node2, weight=1.0):
        """
        Add a similar_to edge between two nodes.
        
        Args:
            node1 (str): First node
            node2 (str): Second node
            weight (float): Similarity weight
        """
        self.add_edge(node1, node2, "similar_to", weight)
    
    def get_graph(self):
        """
        Get the constructed graph.
        
        Returns:
            nx.DiGraph: The knowledge graph
        """
        return self.graph
    
    def save_graph(self, filepath):
        """
        Save the graph to a file.
        
        Args:
            filepath (str): Path to save the graph
        """
        nx.write_graphml(self.graph, filepath)
    
    def load_graph(self, filepath):
        """
        Load a graph from a file.
        
        Args:
            filepath (str): Path to load the graph from
        """
        self.graph = nx.read_graphml(filepath)
