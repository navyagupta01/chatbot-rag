
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
from enum import Enum

from .query_planner import QueryPlan

class ReasoningState(Enum):
    """States of the reasoning process"""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ReasoningStep:
    """
    Represents a single step in the reasoning process

    Each step captures what the agent is doing and why,
    providing transparency into the decision-making process.
    """
    step_number: int              # Order in the reasoning chain
    description: str              # What is being done
    reasoning: str                # Why it's being done
    timestamp: float              # When the step occurred
    inputs: Dict[str, Any]        # Input data for this step
    outputs: Dict[str, Any]       # Output data from this step
    confidence: float             # Confidence in this step (0-1)
    duration: Optional[float] = None    # Time taken for this step

class ReasoningSession:
    """
    Manages a complete reasoning session for answering a query

    This class tracks the entire reasoning process from start to finish,
    enabling transparency and debugging of agent behavior.
    """

    def __init__(self, query: str, plan: QueryPlan):
        """
        Initialize a new reasoning session

        Args:
            query: The user query being reasoned about
            plan: The query plan to execute
        """
        self.query = query
        self.plan = plan
        self.steps: List[ReasoningStep] = []
        self.state = ReasoningState.INITIALIZED
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.session_id = f"session_{int(self.start_time)}"

    def add_step(self, description: str, reasoning: str = "", inputs: Dict = None, outputs: Dict = None, confidence: float = 1.0):
        """
        Add a reasoning step to the session

        Args:
            description: What is being done
            reasoning: Why it's being done
            inputs: Input data for this step
            outputs: Output data from this step
            confidence: Confidence in this step
        """
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            description=description,
            reasoning=reasoning or f"Following planned step {len(self.steps) + 1}",
            timestamp=time.time(),
            inputs=inputs or {},
            outputs=outputs or {},
            confidence=confidence
        )

        self.steps.append(step)

        if self.state == ReasoningState.INITIALIZED:
            self.state = ReasoningState.IN_PROGRESS

    def complete(self, success: bool = True):
        """
        Mark the reasoning session as complete

        Args:
            success: Whether the reasoning was successful
        """
        self.end_time = time.time()
        self.state = ReasoningState.COMPLETED if success else ReasoningState.FAILED

        # Calculate duration for all steps
        for i, step in enumerate(self.steps):
            if i < len(self.steps) - 1:
                step.duration = self.steps[i + 1].timestamp - step.timestamp
            else:
                step.duration = self.end_time - step.timestamp

    def get_duration(self) -> float:
        """Get total session duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the reasoning session"""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "query_type": self.plan.query_type,
            "total_steps": len(self.steps),
            "duration": self.get_duration(),
            "state": self.state.value,
            "avg_confidence": sum(step.confidence for step in self.steps) / len(self.steps) if self.steps else 0
        }

class ReasoningEngine:
    """
    Main Reasoning Engine for Agentic Behavior

    This engine coordinates multi-step reasoning processes,
    ensuring that the agent's decision-making is transparent
    and follows logical steps.
    """

    def __init__(self):
        """Initialize the reasoning engine"""
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.completed_sessions: List[ReasoningSession] = []
        self.reasoning_strategies = self._initialize_strategies()

    def start_reasoning_session(self, query: str, plan: QueryPlan) -> ReasoningSession:
        """
        Start a new reasoning session

        Args:
            query: User query to reason about
            plan: Query execution plan

        Returns:
            New ReasoningSession instance
        """
        session = ReasoningSession(query, plan)
        self.active_sessions[session.session_id] = session

        # Add initial reasoning step
        session.add_step(
            description=f"Initialized reasoning session for {plan.query_type} query",
            reasoning=f"Query analysis indicated {plan.query_type} approach with {plan.confidence:.2f} confidence",
            inputs={"query": query, "plan": plan.__dict__},
            confidence=plan.confidence
        )

        return session

    def execute_reasoning_step(
            self,
            session: ReasoningSession,
            step_description: str,
            reasoning_logic: str,
            execution_function: callable,
            inputs: Dict[str, Any] = None
    ) -> Any:
        """
        Execute a single reasoning step with full tracking

        Args:
            session: Active reasoning session
            step_description: Description of what's being done
            reasoning_logic: Explanation of why it's being done
            execution_function: Function to execute for this step
            inputs: Input parameters for the execution

        Returns:
            Result of the execution function
        """
        start_time = time.time()

        try:
            # Execute the step
            result = execution_function(**(inputs or {}))

            # Calculate confidence based on result
            confidence = self._assess_step_confidence(result, step_description)

            # Add step to session
            session.add_step(
                description=step_description,
                reasoning=reasoning_logic,
                inputs=inputs or {},
                outputs={"result": str(result)[:200]},  # Truncate long outputs
                confidence=confidence
            )

            return result

        except Exception as e:
            # Handle step failure
            session.add_step(
                description=f"Failed: {step_description}",
                reasoning=f"Error during execution: {str(e)}",
                inputs=inputs or {},
                outputs={"error": str(e)},
                confidence=0.0
            )
            raise

    def _assess_step_confidence(self, result: Any, step_description: str) -> float:
        """
        Assess confidence in a reasoning step based on its result

        Args:
            result: Result of the step execution
            step_description: Description of the step

        Returns:
            Confidence score (0.0 to 1.0)
        """

        # Basic confidence assessment
        base_confidence = 0.8

        # Adjust based on result type and content
        if result is None or result == "":
            return 0.3  # Low confidence for empty results

        if isinstance(result, (list, tuple)) and len(result) == 0:
            return 0.4  # Low confidence for empty collections

        if isinstance(result, str) and len(result) < 10:
            return 0.6  # Medium confidence for very short text

        # Boost confidence for successful retrieval steps
        if "retriev" in step_description.lower() and result:
            base_confidence += 0.1

        # Boost confidence for successful analysis steps
        if "analyz" in step_description.lower() and result:
            base_confidence += 0.05

        return min(1.0, base_confidence)

    def complete_session(self, session: ReasoningSession, success: bool = True):
        """
        Complete a reasoning session and move it to completed sessions

        Args:
            session: Session to complete
            success: Whether the session was successful
        """
        session.complete(success)

        # Move from active to completed
        if session.session_id in self.active_sessions:
            del self.active_sessions[session.session_id]

        self.completed_sessions.append(session)

        # Keep only recent completed sessions (memory management)
        if len(self.completed_sessions) > 50:
            self.completed_sessions = self.completed_sessions[-50:]

    def get_reasoning_trace(self, session: ReasoningSession) -> List[str]:
        """
        Get a human-readable reasoning trace

        Args:
            session: Reasoning session

        Returns:
            List of reasoning steps as strings
        """
        trace = []

        for step in session.steps:
            trace.append(f"Step {step.step_number}: {step.description}")
            if step.reasoning and step.reasoning != step.description:
                trace.append(f"  Reasoning: {step.reasoning}")
            if step.confidence < 0.7:
                trace.append(f"  Confidence: {step.confidence:.2f} (low)")

        return trace

    def get_session_metrics(self, session: ReasoningSession) -> Dict[str, Any]:
        """
        Get detailed metrics for a reasoning session

        Args:
            session: Reasoning session

        Returns:
            Dictionary of session metrics
        """
        if not session.steps:
            return {"error": "No steps recorded"}

        step_durations = [step.duration for step in session.steps if step.duration]
        step_confidences = [step.confidence for step in session.steps]

        return {
            "total_steps": len(session.steps),
            "total_duration": session.get_duration(),
            "avg_step_duration": sum(step_durations) / len(step_durations) if step_durations else 0,
            "avg_confidence": sum(step_confidences) / len(step_confidences),
            "min_confidence": min(step_confidences),
            "max_confidence": max(step_confidences),
            "session_state": session.state.value,
            "query_type": session.plan.query_type
        }

    def _initialize_strategies(self) -> Dict[str, Dict]:
        """
        Initialize reasoning strategies for different scenarios

        Returns:
            Dictionary of reasoning strategies
        """
        return {
            "summary": {
                "approach": "hierarchical_synthesis",
                "focus": "broad_coverage",
                "depth": "moderate"
            },
            "search": {
                "approach": "targeted_retrieval",
                "focus": "precision",
                "depth": "specific"
            },
            "comparison": {
                "approach": "parallel_analysis",
                "focus": "structural_comparison",
                "depth": "detailed"
            },
            "analysis": {
                "approach": "deep_reasoning",
                "focus": "insight_generation",
                "depth": "comprehensive"
            }
        }

    def get_reasoning_insights(self, session: ReasoningSession) -> Dict[str, Any]:
        """
        Generate insights about the reasoning process

        Args:
            session: Completed reasoning session

        Returns:
            Dictionary of reasoning insights
        """
        if session.state != ReasoningState.COMPLETED:
            return {"warning": "Session not completed"}

        insights = {
            "reasoning_quality": "high" if session.steps and all(s.confidence > 0.7 for s in session.steps) else "moderate",
            "bottlenecks": [],
            "strengths": [],
            "recommendations": []
        }

        # Identify bottlenecks (slow steps)
        if session.steps:
            avg_duration = sum(s.duration for s in session.steps if s.duration) / len([s for s in session.steps if s.duration])
            bottlenecks = [s for s in session.steps if s.duration and s.duration > avg_duration * 2]
            insights["bottlenecks"] = [f"Step {s.step_number}: {s.description}" for s in bottlenecks]

        # Identify strengths (high confidence steps)
        high_confidence_steps = [s for s in session.steps if s.confidence > 0.9]
        insights["strengths"] = [f"Step {s.step_number}: {s.description}" for s in high_confidence_steps]

        # Generate recommendations
        if insights["bottlenecks"]:
            insights["recommendations"].append("Consider optimizing slow reasoning steps")

        if session.steps and min(s.confidence for s in session.steps) < 0.5:
            insights["recommendations"].append("Some steps had low confidence - consider additional validation")

        return insights
