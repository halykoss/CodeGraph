"""
LLM prompts for extracting domain-specific information from source code.
"""

# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert software engineer and domain specialist. 
Your task is to analyze source code and extract domain-specific information that will be used to build a knowledge graph.
Be precise and accurate in your analysis."""

# Prompt for extracting domain entities
DOMAIN_ENTITIES_PROMPT = """Analyze the following source code and identify domain-specific entities:

{code}

Please provide your response in JSON format with the following structure:
{{
  "entities": [
    {{
      "name": "entity_name",
      "type": "entity_type"
    }}
  ],
  "domain": "primary domain of this code"
}}

If no domain entities are found, return an empty array for entities.
"""

# Prompt for extracting algorithms
ALGORITHMS_PROMPT = """Analyze the following source code and identify algorithms implemented or used:

{code}

Please provide your response in JSON format with the following structure:
{{
  "algorithms": [
    {{
      "name": "algorithm_name",
      "category": "algorithm_category",
      "complexity": "time/space complexity if known"
    }}
  ]
}}

If no algorithms are found, return an empty array.
"""

# Prompt for extracting programming paradigms and design patterns
PARADIGMS_AND_PATTERNS_PROMPT = """Analyze the following source code and identify both programming paradigms and design patterns used:

{code}

Please provide your response in JSON format with the following structure:
{{
  "paradigms": [
    {{
      "name": "paradigm_name",
      "confidence": 0.95,
      "evidence": "brief description of evidence"
    }}
  ],
  "design_patterns": [
    {{
      "name": "pattern_name",
      "category": "creational, structural, behavioral, etc.",
      "description": "brief description of how it's used"
    }}
  ]
}}

Common paradigms include: object-oriented, functional, procedural, declarative, 
imperative, event-driven, concurrent, reactive, etc.

Common patterns include: Singleton, Factory, Observer, Strategy, Decorator, Adapter, 
Repository, MVC, Dependency Injection, etc.

If no paradigms or patterns are found, return empty arrays.
"""

# Prompt for extracting data structures
DATA_STRUCTURES_PROMPT = """Analyze the following source code and identify data structures used or implemented:

{code}

Please provide your response in JSON format with the following structure:
{{
  "data_structures": [
    {{
      "name": "data_structure_name",
      "category": "linear, tree, graph, hash-based, etc.",
      "operations": "operations supported (e.g., insert, delete, search)"
    }}
  ]
}}

If no data structures are found, return an empty array.
"""

# Prompt for domain classification
DOMAIN_CLASSIFICATION_PROMPT = """Analyze the following source code and classify it into application domains:

{code}

Please provide your response in JSON format with the following structure:
{{
  "domains": ["domain1", "domain2", ...],
  "confidence": 0.95
}}

Common domains include: web development, machine learning, data science, bioinformatics, 
image processing, networking, databases, finance, IoT, robotics, gaming, etc.

If the domain is unclear, return ["unknown"].
"""

# Prompt for extracting performance characteristics
PERFORMANCE_CHARACTERISTICS_PROMPT = """Analyze the following source code and identify performance characteristics:

{code}

Please provide your response in JSON format with the following structure:
{{
  "characteristics": [
    {{
      "name": "characteristic_name",
      "value": "estimated or mentioned value",
      "description": "brief description"
    }}
  ]
}}

Performance characteristics include: time_complexity, space_complexity, throughput, latency,
scalability, reliability, availability, efficiency.

If no performance characteristics can be determined, return an empty array.
"""
