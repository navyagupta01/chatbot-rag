
from dataclasses import dataclass
from typing import List, Dict, Any
import re
from config import QueryTypes, AgentConfig

@dataclass
class QueryPlan:
    """
    Represents an execution plan for answering a user query
    query_type:str
    original_query:str
    This is the output of the agentic planning process - a structured
    approach to answering the user's question effectively.
    """
    query_type: str                    # Type of query (summary, search, etc.)
    original_query: str                # User's original question
    steps: List[str]                   # Planned execution steps
    chunks_needed: int                 # Number of document chunks to retrieve
    similarity_threshold: float        # Minimum relevance threshold
    max_tokens: int                    # Maximum response length
    reasoning_approach: str            # High-level reasoning strategy
    confidence: float                  # Confidence in plan quality (0-1)

class QueryPlanner:
    """
    Intelligent Query Analysis and Planning System

    This class embodies the "agentic" behavior of our system:
    1. Analyzes user intent from natural language
    2. Creates strategic plans for different query types
    3. Optimizes retrieval and generation parameters
    4. Provides reasoning transparency
    """

    def __init__(self):
        """Initialize the query planner with analysis patterns"""
        self.query_patterns = self._initialize_patterns()
        self.planning_strategies = self._initialize_strategies()

    def analyze_query(self, query: str) -> QueryPlan:
        """
        Analyze user query and create an intelligent execution plan

        This is the core agentic function that decides HOW to answer
        a question before actually answering it.

        Args:
            query: User's natural language query

        Returns:
            QueryPlan with strategic approach to answering
        """

        # Normalize query for analysis
        normalized_query = self._normalize_query(query)

        # Step 1: Detect query type using multiple methods
        query_type = self._detect_query_type(normalized_query)

        # Step 2: Extract key entities and concepts
        entities = self._extract_entities(normalized_query)

        # Step 3: Assess query complexity
        complexity = self._assess_complexity(normalized_query)

        # Step 4: Create strategic plan
        plan = self._create_strategic_plan(query_type, normalized_query, entities, complexity)

        # Step 5: Optimize parameters based on query characteristics
        plan = self._optimize_parameters(plan, complexity)

        return plan

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text for better analysis

        Args:
            query: Raw user query

        Returns:
            Normalized query string
        """
        # Convert to lowercase for pattern matching
        normalized = query.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove common filler words that don't affect intent
        filler_words = ['please', 'could you', 'can you', 'would you']
        for filler in filler_words:
            normalized = normalized.replace(filler, '').strip()

        return normalized

    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query using keyword analysis and patterns

        This is crucial for agentic behavior - understanding WHAT
        the user wants before deciding HOW to provide it.

        Args:
            query: Normalized query string

        Returns:
            Query type identifier
        """

        # Calculate scores for each query type
        type_scores = {}

        for query_type, keywords in AgentConfig.QUERY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
                    # Boost score if keyword appears at the beginning
                    if query.startswith(keyword):
                        score += 0.5

            # Normalize score by number of keywords
            type_scores[query_type] = score / len(keywords) if keywords else 0

        # Additional pattern-based detection
        type_scores.update(self._pattern_based_detection(query))

        # Return the type with highest score
        if type_scores:
            best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            if type_scores[best_type] > 0:
                return best_type

        # Default to search if no clear type detected
        return QueryTypes.SEARCH

    def _pattern_based_detection(self, query: str) -> Dict[str, float]:
        """
        Use regex patterns to detect query types

        Args:
            query: Normalized query string

        Returns:
            Dictionary of query type scores
        """
        scores = {}

        # Summary patterns
        if re.search(r'\b(summarize|summary|overview|main points|key points)\b', query):
            scores[QueryTypes.SUMMARY] = 1.0

        # Comparison patterns
        if re.search(r'\b(compare|comparison|versus|vs\.?|difference|similarities)\b', query):
            scores[QueryTypes.COMPARISON] = 1.0

        # Analysis patterns
        if re.search(r'\b(analyze|analysis|why|because|implications?|significance)\b', query):
            scores[QueryTypes.ANALYSIS] = 1.0

        # Search patterns (questions)
        if re.search(r'^(what|where|when|who|how|which|is|are|does|do|did)\b', query):
            scores[QueryTypes.SEARCH] = 0.8

        return scores

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract key entities and concepts from the query

        This helps the planner understand what specific information
        the user is interested in.

        Args:
            query: Normalized query string

        Returns:
            List of identified entities/concepts
        """
        entities = []

        # Look for quoted phrases (explicit entities)
        quoted_entities = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_entities)

        # Look for capitalized words (likely proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized_words)

        # Look for numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities.extend(numbers)

        return list(set(entities))  # Remove duplicates

    def _assess_complexity(self, query: str) -> str:
        """
        Assess the complexity of the query

        More complex queries need different handling strategies.

        Args:
            query: Normalized query string

        Returns:
            Complexity level: 'simple', 'moderate', or 'complex'
        """
        complexity_indicators = {
            'simple': ['what is', 'who is', 'when', 'where'],
            'moderate': ['how', 'why', 'explain', 'describe'],
            'complex': ['analyze', 'compare', 'evaluate', 'implications', 'relationship']
        }

        # Check for complexity indicators
        for level, indicators in complexity_indicators.items():
            if any(indicator in query for indicator in indicators):
                return level

        # Length-based complexity assessment
        word_count = len(query.split())
        if word_count <= 5:
            return 'simple'
        elif word_count <= 15:
            return 'moderate'
        else:
            return 'complex'

    def _create_strategic_plan(self, query_type: str, query: str, entities: List[str], complexity: str) -> QueryPlan:
        """
        Create a strategic plan for answering the query

        This is where the agentic magic happens - creating a thoughtful
        approach to answering based on the analysis.

        Args:
            query_type: Detected query type
            query: Original query
            entities: Extracted entities
            complexity: Assessed complexity

        Returns:
            Initial QueryPlan
        """

        # Get base configuration for this query type
        config = AgentConfig.QUERY_TYPE_THRESHOLDS.get(query_type, {})

        # Create type-specific execution steps
        steps = self._create_execution_steps(query_type, complexity, entities)

        # Determine reasoning approach
        reasoning_approach = self._determine_reasoning_approach(query_type, complexity)

        # Calculate confidence based on various factors
        confidence = self._calculate_plan_confidence(query_type, entities, complexity)

        return QueryPlan(
            query_type=query_type,
            original_query=query,
            steps=steps,
            chunks_needed=config.get('chunks_needed', 5),
            similarity_threshold=config.get('similarity_threshold', 0.5),
            max_tokens=config.get('max_tokens', 1000),
            reasoning_approach=reasoning_approach,
            confidence=confidence
        )

    def _create_execution_steps(self, query_type: str, complexity: str, entities: List[str]) -> List[str]:
        """
        Create specific execution steps for different query types

        Args:
            query_type: Type of query
            complexity: Complexity level
            entities: Extracted entities

        Returns:
            List of execution steps
        """

        base_steps = {
            QueryTypes.SUMMARY: [
                "Scan document for main topics and themes",
                "Identify key points from each major section",
                "Extract important details and supporting information",
                "Synthesize information into coherent summary",
                "Structure summary with clear organization"
            ],

            QueryTypes.SEARCH: [
                "Understand the specific question being asked",
                "Locate relevant information in document",
                "Verify accuracy and completeness of information",
                "Provide direct answer with supporting context"
            ],

            QueryTypes.COMPARISON: [
                "Identify entities or concepts to compare",
                "Gather information about each entity",
                "Analyze similarities between entities",
                "Analyze differences between entities",
                "Present structured comparison"
            ],

            QueryTypes.ANALYSIS: [
                "Gather relevant background information",
                "Identify key factors and relationships",
                "Apply analytical reasoning framework",
                "Draw insights and conclusions",
                "Consider broader implications"
            ]
        }

        steps = base_steps.get(query_type, base_steps[QueryTypes.SEARCH])

        # Modify steps based on complexity
        if complexity == 'complex':
            steps.insert(-1, "Cross-reference information across multiple sources")
            steps.append("Validate conclusions against available evidence")

        # Add entity-specific steps if entities were found
        if entities:
            steps.insert(1, f"Focus on information related to: {', '.join(entities[:3])}")

        return steps

    def _determine_reasoning_approach(self, query_type: str, complexity: str) -> str:
        """
        Determine the high-level reasoning approach

        Args:
            query_type: Type of query
            complexity: Complexity level

        Returns:
            Reasoning approach description
        """

        approaches = {
            QueryTypes.SUMMARY: "Hierarchical information synthesis",
            QueryTypes.SEARCH: "Direct information retrieval and validation",
            QueryTypes.COMPARISON: "Parallel analysis with structured comparison",
            QueryTypes.ANALYSIS: "Deep analytical reasoning with inference"
        }

        base_approach = approaches.get(query_type, "General information processing")

        if complexity == 'complex':
            return f"Multi-step {base_approach.lower()} with cross-validation"
        else:
            return base_approach

    def _calculate_plan_confidence(self, query_type: str, entities: List[str], complexity: str) -> float:
        """
        Calculate confidence in the execution plan

        Args:
            query_type: Detected query type
            entities: Extracted entities
            complexity: Query complexity

        Returns:
            Confidence score (0.0 to 1.0)
        """

        confidence = 0.7  # Base confidence

        # Boost confidence if query type was clearly detected
        if query_type != QueryTypes.SEARCH:  # Search is default, so less confident
            confidence += 0.2

        # Boost confidence if entities were extracted
        if entities:
            confidence += min(0.1 * len(entities), 0.2)

        # Adjust for complexity
        complexity_adjustments = {
            'simple': 0.1,
            'moderate': 0.0,
            'complex': -0.1
        }
        confidence += complexity_adjustments.get(complexity, 0)

        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, confidence))

    def _optimize_parameters(self, plan: QueryPlan, complexity: str) -> QueryPlan:
        """
        Optimize retrieval and generation parameters based on plan

        Args:
            plan: Initial query plan
            complexity: Query complexity

        Returns:
            Optimized QueryPlan
        """

        # Adjust parameters based on complexity
        if complexity == 'complex':
            plan.chunks_needed = min(plan.chunks_needed + 3, 12)  # More context for complex queries
            plan.max_tokens = min(plan.max_tokens + 500, 2000)   # Longer responses
            plan.similarity_threshold = max(plan.similarity_threshold - 0.1, 0.2)  # Cast wider net

        elif complexity == 'simple':
            plan.chunks_needed = max(plan.chunks_needed - 2, 3)   # Less context needed
            plan.max_tokens = max(plan.max_tokens - 200, 500)     # Shorter responses
            plan.similarity_threshold = min(plan.similarity_threshold + 0.1, 0.8)  # Higher precision

        return plan

    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for query analysis"""
        return {
            'question_words': [r'\b(what|where|when|who|how|why|which)\b'],
            'comparison_words': [r'\b(compare|vs|versus|difference|similar)\b'],
            'summary_words': [r'\b(summarize|summary|overview|outline)\b'],
            'analysis_words': [r'\b(analyze|analysis|explain|evaluate)\b']
        }

    def _initialize_strategies(self) -> Dict[str, Dict]:
        """Initialize planning strategies for different scenarios"""
        return {
            'high_confidence': {'boost_threshold': True, 'increase_context': False},
            'low_confidence': {'boost_threshold': False, 'increase_context': True},
            'entity_rich': {'focus_retrieval': True, 'targeted_search': True},
            'entity_poor': {'broad_retrieval': True, 'comprehensive_search': True}
        }
