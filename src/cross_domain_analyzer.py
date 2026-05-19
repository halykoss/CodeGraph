"""
Cross-domain analysis module for identifying algorithm transfer and shared patterns.
"""

class CrossDomainAnalyzer:
    """Analyzes cross-domain similarities and algorithm transfers."""
    
    def __init__(self, graph):
        """
        Initialize the cross-domain analyzer.
        
        Args:
            graph: The knowledge graph to analyze
        """
        self.graph = graph
    
    def find_algorithm_transfer(self, algorithm):
        """
        Find how an algorithm is used across different domains.
        
        Args:
            algorithm (str): Algorithm name to analyze
            
        Returns:
            dict: Domains where the algorithm is used and usage patterns
        """
        # Find all files that use this algorithm
        files_using_algorithm = [
            edge[0] for edge in self.graph.edges(data=True)
            if edge[2].get('type') == 'uses_algorithm' and edge[1] == algorithm
        ]
        
        # Find domains for these files
        domains = {}
        for file_path in files_using_algorithm:
            # Find the domain this file belongs to
            domain_edges = [
                edge for edge in self.graph.out_edges(file_path, data=True)
                if edge[2].get('type') == 'belongs_to_domain'
            ]
            
            for _, domain, _ in domain_edges:
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append(file_path)
        
        return {
            "algorithm": algorithm,
            "domains": domains,
            "total_usage": len(files_using_algorithm)
        }
    
    def find_shared_patterns(self):
        """
        Find common technical solutions across domains.
        
        Returns:
            dict: Shared patterns and where they appear
        """
        # Find all algorithms and data structures used in the graph
        algorithm_edges = [
            edge for edge in self.graph.edges(data=True)
            if edge[2].get('type') == 'uses_algorithm'
        ]
        
        data_structure_edges = [
            edge for edge in self.graph.edges(data=True)
            if edge[2].get('type') == 'uses_data_structure'
        ]
        
        # Group algorithms and data structures by domain
        patterns_by_domain = {}
        
        # Process algorithms
        for file_path, algorithm, _ in algorithm_edges:
            # Find the domain this file belongs to
            domain_edges = [
                edge for edge in self.graph.out_edges(file_path, data=True)
                if edge[2].get('type') == 'belongs_to_domain'
            ]
            
            for _, domain, _ in domain_edges:
                if domain not in patterns_by_domain:
                    patterns_by_domain[domain] = set()
                patterns_by_domain[domain].add(('algorithm', algorithm))
        
        # Process data structures
        for file_path, data_structure, _ in data_structure_edges:
            # Find the domain this file belongs to
            domain_edges = [
                edge for edge in self.graph.out_edges(file_path, data=True)
                if edge[2].get('type') == 'belongs_to_domain'
            ]
            
            for _, domain, _ in domain_edges:
                if domain not in patterns_by_domain:
                    patterns_by_domain[domain] = set()
                patterns_by_domain[domain].add(('data_structure', data_structure))
        
        # Find shared patterns between domains
        shared_patterns = {}
        domains = list(patterns_by_domain.keys())
        
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1 = domains[i]
                domain2 = domains[j]
                
                shared = patterns_by_domain[domain1] & patterns_by_domain[domain2]
                
                if shared:
                    key = f"{domain1} <-> {domain2}"
                    # Separate algorithms and data structures
                    shared_algorithms = [item[1] for item in shared if item[0] == 'algorithm']
                    shared_data_structures = [item[1] for item in shared if item[0] == 'data_structure']
                    shared_patterns[key] = {
                        'algorithms': shared_algorithms,
                        'data_structures': shared_data_structures
                    }
        
        return shared_patterns
    
    def find_domain_similarities(self):
        """
        Detect similarities between domains based on shared algorithms and data structures.
        
        Returns:
            dict: Similarity scores between domains
        """
        # Find all domains
        domains = [
            node for node, attrs in self.graph.nodes(data=True)
            if attrs.get('type') == 'domain'
        ]
        
        # Calculate similarity scores
        similarities = {}
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1 = domains[i]
                domain2 = domains[j]
                
                # Find algorithms and data structures used in both domains
                patterns_domain1 = self._get_patterns_for_domain(domain1)
                patterns_domain2 = self._get_patterns_for_domain(domain2)
                
                # Calculate Jaccard similarity
                intersection = len(patterns_domain1 & patterns_domain2)
                union = len(patterns_domain1 | patterns_domain2)
                
                if union > 0:
                    similarity = intersection / union
                    key = f"{domain1} <-> {domain2}"
                    similarities[key] = similarity
        
        return similarities
    
    def _get_patterns_for_domain(self, domain):
        """
        Get all patterns (algorithms and data structures) used in a specific domain.
        
        Args:
            domain (str): Domain name
            
        Returns:
            set: Set of patterns used in the domain
        """
        patterns = set()
        
        # Find all files in this domain
        file_edges = [
            edge for edge in self.graph.in_edges(domain, data=True)
            if edge[2].get('type') == 'belongs_to_domain'
        ]
        
        # Find algorithms and data structures used by these files
        for file_path, domain_node, _ in file_edges:
            # Find algorithms
            algorithm_edges = [
                edge for edge in self.graph.out_edges(file_path, data=True)
                if edge[2].get('type') == 'uses_algorithm'
            ]
            for _, algorithm, _ in algorithm_edges:
                patterns.add(('algorithm', algorithm))
            
            # Find data structures
            data_structure_edges = [
                edge for edge in self.graph.out_edges(file_path, data=True)
                if edge[2].get('type') == 'uses_data_structure'
            ]
            for _, data_structure, _ in data_structure_edges:
                patterns.add(('data_structure', data_structure))
        
        return patterns
    
    def _get_algorithms_for_domain(self, domain):
        """
        Get all algorithms used in a specific domain.
        
        Args:
            domain (str): Domain name
            
        Returns:
            set: Set of algorithms used in the domain
        """
        algorithms = set()
        
        # Find all files in this domain
        file_edges = [
            edge for edge in self.graph.in_edges(domain, data=True)
            if edge[2].get('type') == 'belongs_to_domain'
        ]
        
        # Find algorithms used by these files
        for file_path, domain_node, _ in file_edges:
            algorithm_edges = [
                edge for edge in self.graph.out_edges(file_path, data=True)
                if edge[2].get('type') == 'uses_algorithm'
            ]
            
            for _, algorithm, _ in algorithm_edges:
                algorithms.add(algorithm)
        
        return algorithms
    
    def get_cross_domain_insights(self):
        """
        Get comprehensive cross-domain insights.
        
        Returns:
            dict: Comprehensive analysis of cross-domain relationships
        """
        return {
            "algorithm_transfers": self._get_algorithm_transfers(),
            "shared_patterns": self.find_shared_patterns(),
            "domain_similarities": self.find_domain_similarities()
        }
    
    def _get_algorithm_transfers(self):
        """
        Get all algorithm transfers across domains.
        
        Returns:
            dict: Algorithms and their usage across domains
        """
        # Find all algorithms in the graph
        algorithms = [
            node for node, attrs in self.graph.nodes(data=True)
            if attrs.get('type') == 'algorithm'
        ]
        
        # Get transfer information for each algorithm
        transfers = {}
        for algorithm in algorithms:
            transfer_info = self.find_algorithm_transfer(algorithm)
            if transfer_info["total_usage"] > 0:
                transfers[algorithm] = transfer_info["domains"]
        
        return transfers
