"""
Main orchestrator module for the LLM-Based Code Graph Construction framework.
Uses HuggingFaceTB/stack-edu dataset for code analysis.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import json
from src.domain_classifier import DomainClassifier
from src.llm_extractor import LLMExtractor
from src.graph_builder import GraphBuilder
from src.cross_domain_analyzer import CrossDomainAnalyzer
from utils.utils import (
    download_stack_edu_dataset,
    save_samples_to_file, 
    load_samples_from_file
)

class StackEduCodeGraphOrchestrator:
    """
    Orchestrates code graph construction using the HuggingFaceTB/stack-edu dataset.
    """
    
    def __init__(self, cache_dir: str = "cache", enable_cache: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            cache_dir: Directory to cache downloaded samples
            enable_cache: Whether to enable caching of samples
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("🚀 Initializing LLM-based Code Graph Orchestrator...")
        self.llm_extractor = LLMExtractor()
        self.domain_classifier = DomainClassifier(self.llm_extractor)
        self.graph_builder = GraphBuilder()
        self.cross_domain_analyzer = None
        
        # Cache settings
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        os.makedirs(cache_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'domains_found': set(),
            'functions_found': set(),
            'algorithms_found': set(),
            'paradigms_found': set(),
            'design_patterns_found': set(),
            'data_structures_found': set(),
            'errors': 0
        }

    def download_dataset(self, num_samples: int = 1000, programming_language: str = 'Python') -> List[Dict]:
        """
        Download samples from HuggingFaceTB/stack-edu dataset.
        
        Args:
            num_samples: Number of samples to download
            programming_language: Programming language to filter (e.g., 'Python', 'Java', 'JavaScript')
            
        Returns:
            List of code samples
        """
        cache_file = os.path.join(self.cache_dir, f"stack_edu_{num_samples}_{programming_language}_samples.json")
        
        # Try to load from cache first
        if self.enable_cache and os.path.exists(cache_file):
            self.logger.info(f"📂 Loading samples from cache: {cache_file}")
            samples = load_samples_from_file(cache_file)
            if samples:
                self.stats['total_samples'] = len(samples)
                return samples
        
        # Download fresh samples
        self.logger.info(f"📥 Downloading {num_samples} {programming_language} samples from HuggingFaceTB/stack-edu...")
        samples = download_stack_edu_dataset(num_samples, programming_language)
        
        # Cache the results
        if self.enable_cache and samples:
            save_samples_to_file(samples, cache_file)
        
        self.stats['total_samples'] = len(samples)
        return samples
    
    def process_code_sample(self, sample: Dict) -> Dict[str, Any]:
        """
        Process a single code sample from the dataset.
        
        Args:
            sample: Code sample from the dataset
            
        Returns:
            Dictionary with extracted information
        """
        try:
            code_content = sample.get('content', '')
            sample_id = sample.get('blob_id', 'unknown')
            language = sample.get('language', 'unknown')
            
            if not code_content.strip():
                return None
            
            self.logger.debug(f"🔍 Processing sample {sample_id} ({language})")
            
            # Extract information using LLM
            domains = self.domain_classifier.classify_domain(code_content)
            functions = [] # Removed technical functions extraction
            #performance_chars = self.domain_classifier.extract_performance_characteristics(code_content)
            
            #entities = self.llm_extractor.extract_entities(code_content)
            algorithms = self.llm_extractor.extract_algorithms(code_content)
            paradigms_and_patterns = self.llm_extractor.extract_paradigms_and_patterns(code_content)
            paradigms = paradigms_and_patterns.get("paradigms", [])
            design_patterns = paradigms_and_patterns.get("design_patterns", [])
            #data_structures = self.llm_extractor.extract_data_structures(code_content)
            #protocols = self.llm_extractor.extract_protocols(code_content)
            #relationships = self.llm_extractor.extract_relationships(code_content)
            
            # Update statistics
            self.stats['domains_found'].update(domains)
            self.stats['functions_found'].update(functions)
            if isinstance(algorithms, list):
                for algo in algorithms:
                    if isinstance(algo, dict) and 'name' in algo:
                        self.stats['algorithms_found'].add(algo['name'])
                    elif isinstance(algo, str):
                        self.stats['algorithms_found'].add(algo)
            
            if isinstance(paradigms, list):
                for para in paradigms:
                    if isinstance(para, dict) and 'name' in para:
                        self.stats['paradigms_found'].add(para['name'])
                    elif isinstance(para, str):
                        self.stats['paradigms_found'].add(para)
            
            if isinstance(design_patterns, list):
                for dp in design_patterns:
                    if isinstance(dp, dict) and 'name' in dp:
                        self.stats['design_patterns_found'].add(dp['name'])
                    elif isinstance(dp, str):
                        self.stats['design_patterns_found'].add(dp)
            
            
            # Build the graph
            file_node_id = f"{sample_id}_{language}"
            
            # Add file node with metadata
            file_metadata = {
                'language': language,
                'size': sample.get('size', 0),
                'repository': sample.get('repository', 'unknown'),
                'file_path': sample.get('file_path', ''),
                #'performance_characteristics': performance_chars,
                #'entities': entities,
                'paradigms': paradigms,
                'design_patterns': design_patterns,
                #'data_structures': data_structures,
                #'protocols': protocols,
                #'relationships': relationships
            }
            
            self.graph_builder.add_file_node(file_node_id, file_metadata)
            
            # Add domain nodes and relationships
            for domain in domains:
                self.graph_builder.add_domain_node(domain)
                self.graph_builder.add_belongs_to_domain_edge(file_node_id, domain)
            
            # Add function nodes and relationships
            for function in functions:
                self.graph_builder.add_function_node(function)
                self.graph_builder.add_implements_function_edge(file_node_id, function)
            
            # Add algorithm nodes and relationships
            if isinstance(algorithms, list):
                for algorithm in algorithms:
                    if isinstance(algorithm, dict):
                        algo_name = algorithm.get('name', 'unknown_algorithm')
                        algo_metadata = {
                            'category': algorithm.get('category', ''),
                            'complexity': algorithm.get('complexity', ''),
                            'description': algorithm.get('description', '')
                        }
                        self.graph_builder.add_algorithm_node(algo_name, algo_metadata)
                        self.graph_builder.add_uses_algorithm_edge(file_node_id, algo_name)
                    elif isinstance(algorithm, str):
                        self.graph_builder.add_algorithm_node(algorithm)
                        self.graph_builder.add_uses_algorithm_edge(file_node_id, algorithm)
            
            # Add programming paradigm nodes and relationships
            if isinstance(paradigms, list):
                for paradigm in paradigms:
                    if isinstance(paradigm, dict):
                        para_name = paradigm.get('name', 'unknown_paradigm')
                        para_metadata = {
                            'confidence': paradigm.get('confidence', ''),
                            'evidence': paradigm.get('evidence', '')
                        }
                        self.graph_builder.add_programming_paradigm_node(para_name, para_metadata)
                        self.graph_builder.add_uses_paradigm_edge(file_node_id, para_name)
                    elif isinstance(paradigm, str):
                        self.graph_builder.add_programming_paradigm_node(paradigm)
                        self.graph_builder.add_uses_paradigm_edge(file_node_id, paradigm)
            
            # Add design pattern nodes and relationships
            if isinstance(design_patterns, list):
                for design_pattern in design_patterns:
                    if isinstance(design_pattern, dict):
                        dp_name = design_pattern.get('name', 'unknown_design_pattern')
                        dp_metadata = {
                            'category': design_pattern.get('category', ''),
                            'description': design_pattern.get('description', '')
                        }
                        self.graph_builder.add_design_pattern_node(dp_name, dp_metadata)
                        self.graph_builder.add_uses_design_pattern_edge(file_node_id, dp_name)
                    elif isinstance(design_pattern, str):
                        self.graph_builder.add_design_pattern_node(design_pattern)
                        self.graph_builder.add_uses_design_pattern_edge(file_node_id, design_pattern)
            
            self.stats['processed_samples'] += 1
            
            return {
                'sample_id': sample_id,
                'language': language,
                'domains': domains,
                'functions': functions,
                'algorithms': algorithms,
                'paradigms': paradigms,
                'design_patterns': design_patterns,
                'file_node_id': file_node_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing sample {sample.get('id', 'unknown')}: {e}")
            self.stats['errors'] += 1
            return None
    
    def process_dataset(self, samples: List[Dict], max_samples: int = None) -> List[Dict]:
        """
        Process multiple code samples from the dataset.
        
        Args:
            samples: List of code samples
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            List of processing results
        """
        if max_samples:
            samples = samples[:max_samples]
        
        results = []
        
        self.logger.info(f"🔄 Processing {len(samples)} code samples...")
        
        for sample in tqdm(samples, desc="Processing samples"):
            result = self.process_code_sample(sample)
            if result:
                results.append(result)
        
        self.logger.info(f"✅ Processed {len(results)} samples successfully")
        return results
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*50)
        print("📊 PROCESSING STATISTICS")
        print("="*50)
        print(f"Total samples downloaded: {self.stats['total_samples']}")
        print(f"Successfully processed: {self.stats['processed_samples']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Success rate: {(self.stats['processed_samples']/self.stats['total_samples']*100):.1f}%")
        print(f"\nUnique domains found: {len(self.stats['domains_found'])}")
        print(f"Domain list: {sorted(list(self.stats['domains_found']))[:10]}{'...' if len(self.stats['domains_found']) > 10 else ''}")
        print(f"\nUnique functions found: {len(self.stats['functions_found'])}")
        print(f"Function list: {sorted(list(self.stats['functions_found']))[:10]}{'...' if len(self.stats['functions_found']) > 10 else ''}")
        print(f"\nUnique algorithms found: {len(self.stats['algorithms_found'])}")
        print(f"Algorithm list: {sorted(list(self.stats['algorithms_found']))[:10]}{'...' if len(self.stats['algorithms_found']) > 10 else ''}")
        print(f"\nUnique paradigms found: {len(self.stats['paradigms_found'])}")
        print(f"Paradigm list: {sorted(list(self.stats['paradigms_found']))[:10]}{'...' if len(self.stats['paradigms_found']) > 10 else ''}")
        print(f"\nUnique design patterns found: {len(self.stats['design_patterns_found'])}")
        print(f"Design pattern list: {sorted(list(self.stats['design_patterns_found']))[:10]}{'...' if len(self.stats['design_patterns_found']) > 10 else ''}")
        print(f"\nUnique data structures found: {len(self.stats['data_structures_found'])}")
        print(f"Data structure list: {sorted(list(self.stats['data_structures_found']))[:10]}{'...' if len(self.stats['data_structures_found']) > 10 else ''}")
    
    def save_results(self, results: List[Dict], filepath: str):
        """
        Save processing results to a file.
        
        Args:
            results: List of processing results
            filepath: Path to save the results
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': {
                    'total_samples': self.stats['total_samples'],
                    'processed_samples': self.stats['processed_samples'],
                    'errors': self.stats['errors'],
                    'domains_found': list(self.stats['domains_found']),
                    #'functions_found': list(self.stats['functions_found']),
                    'algorithms_found': list(self.stats['algorithms_found']),
                    'paradigms_found': list(self.stats['paradigms_found']),
                    'design_patterns_found': list(self.stats['design_patterns_found']),
                    #'data_structures_found': list(self.stats['data_structures_found'])
                },
                'results': results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Results saved to {filepath}")
    
    def analyze_cross_domain(self):
        """
        Perform cross-domain analysis on the constructed graph.
        
        Returns:
            CrossDomainAnalyzer: Analyzer with insights
        """
        self.logger.info("🔍 Performing cross-domain analysis...")
        self.cross_domain_analyzer = CrossDomainAnalyzer(self.graph_builder.get_graph())
        return self.cross_domain_analyzer
    
    def get_graph(self):
        """
        Get the constructed knowledge graph.
        
        Returns:
            nx.DiGraph: The knowledge graph
        """
        return self.graph_builder.get_graph()
    
    def save_graph(self, filepath: str):
        """
        Save the knowledge graph to a file.
        
        Args:
            filepath: Path to save the graph
        """
        self.graph_builder.save_graph(filepath)
        self.logger.info(f"💾 Knowledge graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """
        Load a knowledge graph from a file.
        
        Args:
            filepath: Path to load the graph from
        """
        self.graph_builder.load_graph(filepath)
        self.logger.info(f"📂 Knowledge graph loaded from {filepath}")

    def run_full_pipeline(self, num_samples: int = 1000, programming_language: str = 'Python', output_dir: str = "output", resume: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline: download, process, analyze, and save results.
        
        Args:
            num_samples: Number of samples to process from the dataset
            programming_language: Programming language to filter (e.g., 'Python', 'Java', 'JavaScript')
            output_dir: Directory to save outputs
            resume: Whether to resume from previous results
            
        Returns:
            Dictionary with pipeline results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Download dataset
        self.logger.info("🎯 Step 1: Downloading dataset...")
        samples = self.download_dataset(num_samples, programming_language)
        
        if not samples:
            self.logger.error("❌ No samples downloaded. Exiting.")
            return {'success': False, 'error': 'No samples downloaded'}
            
        processed_ids = set()
        
        # Resume logic
        if resume:
            # Look for processing_results_part_*.json files
            import glob
            part_files = glob.glob(os.path.join(output_dir, "processing_results_part_*.json"))
            part_files.sort()
            
            for p_file in part_files:
                try:
                    self.logger.info(f"📂 Checking processed samples in {p_file}")
                    with open(p_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        part_results = data.get('results', [])
                        
                        # Identify processed samples
                        for res in part_results:
                             if 'sample_id' in res:
                                 processed_ids.add(res['sample_id'])
                except Exception as e:
                     self.logger.warning(f"⚠️ Failed to load existing results from {p_file}: {e}")
            
            self.logger.info(f"⏩ Found {len(processed_ids)} already processed samples across {len(part_files)} files.")

        
        # Filter samples to process
        # Use blob_id as identifier which matches sample_id in results
        samples_to_process = [s for s in samples if s.get('blob_id') not in processed_ids]
        
        if len(samples_to_process) < len(samples):
             self.logger.info(f"⏭️ Skipping {len(samples) - len(samples_to_process)} already processed samples.")
        
        # Step 2: Process samples in chunks
        CHUNK_SIZE = 20000
        total_to_process = len(samples_to_process)
        self.logger.info(f"🎯 Step 2: Processing {total_to_process} remaining samples in chunks of {CHUNK_SIZE}...")
        
        # Calculate starting chunk index for filename
        # Start user from 0 or max existing index + 1
        existing_parts = [f for f in os.listdir(output_dir) if f.startswith("processing_results_part_") and f.endswith(".json")]
        start_chunk_idx = 0
        if existing_parts:
             try:
                 indices = [int(f.replace("processing_results_part_", "").replace(".json", "")) for f in existing_parts]
                 if indices:
                     start_chunk_idx = max(indices) + 1
             except ValueError:
                 pass

        processed_count = 0
        
        for i in range(0, total_to_process, CHUNK_SIZE):
            chunk = samples_to_process[i:i + CHUNK_SIZE]
            chunk_idx = start_chunk_idx + (i // CHUNK_SIZE)
            
            self.logger.info(f"🔄 Processing chunk {chunk_idx} ({len(chunk)} samples)...")
            
            # Process chunk
            chunk_results = self.process_dataset(chunk)
            
            # Save chunk results
            chunk_file = os.path.join(output_dir, f"processing_results_part_{chunk_idx}.json")
            self.save_results(chunk_results, chunk_file)
            
            processed_count += len(chunk_results)
            
            # Clear memory if needed (though local vars should be collected)
            del chunk_results
            
            self.logger.info(f"✅ Chunk {chunk_idx} saved. Progress: {min(i + CHUNK_SIZE, total_to_process)}/{total_to_process}")

        
        # Step 4: Save knowledge graph - REMOVED as per request to improve scalability
        # graph_file = os.path.join(output_dir, "knowledge_graph.graphml")
        # self.save_graph(graph_file)
        
        # Step 5: Cross-domain analysis - REMOVED as per request.
        
        # Step 6: Print statistics
        self.print_statistics()
        
        return {
            'success': True,
            'samples_processed': processed_count,
            'graph_nodes': len(self.get_graph().nodes()),
            'graph_edges': len(self.get_graph().edges()),
            'output_files': {
                'results_dir': output_dir, 
                'insights': None
            }
        }

def main():
    """
    Main function to run the Stack-Edu code analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="LLM-Based Code Graph Construction using HuggingFaceTB/stack-edu dataset"
    )
    
    parser.add_argument(
        '--samples', '-n', 
        type=int, 
        default=1000,
        help='Number of samples to download and process (default: 1000)'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        default='Python',
        help='Programming language to filter (default: Python). Options: Python, Java, JavaScript, C, C++, Go, Ruby, etc.'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output/full_pipeline',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--cache-dir', '-c',
        type=str,
        default='cache',
        help='Cache directory for downloaded samples (default: cache)'
    )
    
    parser.add_argument(
        '--disable-cache',
        action='store_true',
        help='Disable caching of downloaded samples'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing from existing results if available'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 LLM-Based Code Graph Construction")
    print(f"📊 Dataset: HuggingFaceTB/stack-edu")
    print(f"🔢 Samples: {args.samples}")
    print(f" Language: {args.language}")
    print(f"📁 Output: {args.output_dir}")
    print("="*50)
    
    try:
        # Initialize orchestrator
        orchestrator = StackEduCodeGraphOrchestrator(
            cache_dir=args.cache_dir,
            enable_cache=not args.disable_cache
        )
        
        # Run the full pipeline
        results = orchestrator.run_full_pipeline(
            num_samples=args.samples,
            programming_language=args.language,
            output_dir=args.output_dir,
            resume=args.resume
        )
        
        if results['success']:
            print("\n🎉 Pipeline completed successfully!")
            print(f"✅ Processed {results['samples_processed']} samples")
            print(f"📈 Graph: {results['graph_nodes']} nodes, {results['graph_edges']} edges")
            print(f"📁 Output files:")
            for file_type, filepath in results['output_files'].items():
                if filepath:
                    print(f"   - {file_type}: {filepath}")
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def quick_demo():
    """
    Run a quick demo with a small number of samples.
    """
    print("🎯 Quick Demo Mode - Processing 50 Python samples")
    
    orchestrator = StackEduCodeGraphOrchestrator()
    results = orchestrator.run_full_pipeline(
        num_samples=50,
        programming_language='Python',
        output_dir='demo_output'
    )
    
    if results['success']:
        print("\n✅ Demo completed! Check the 'demo_output' directory for results.")
    else:
        print(f"\n❌ Demo failed: {results.get('error')}")


if __name__ == "__main__":
    # Check if we should run demo mode
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        quick_demo()
    else:
        exit(main())
