import asyncio
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import AutonomousConfig, create_autonomous_config
from utils.dspy import configure_dspy
from utils.logger import setup_logger
from utils.user_profile import UserProfile, create_sample_user_profile
from config import estimate_cost, load_config


logger = setup_logger()

from agents import EvaluationAgent, PromptOptimizerAgent, RecommendationAgent
from agents.evaluation import EvaluationResult
from utils.metrics_callback import MetricsCallbackHandler

# Global agent instances
prompt_optimizer = None
recommendation_agent = None
evaluation_agent = None


class SystemState(TypedDict):
    """
    Global state for the multi-agent system

    Attributes:
        current_user_profile: User profile being processed
        recommendations: Story recommendations from recommendation agent
        evaluation_result: Results from evaluation
        prompt_optimization_feedback: Feedback for prompt improvements
        iteration_count: Number of optimization iterations
        error_state: Error information if any
        workflow_stage: Current stage in the workflow
        autonomous_config: Configuration for autonomous loop
        start_time: When the optimization loop started
        score_history: History of evaluation scores
        last_improvement_iteration: Last iteration where we saw improvement
    """

    current_user_profile: Optional[UserProfile]
    recommendations: List[Dict[str, Any]]
    evaluation_result: Optional[EvaluationResult]
    prompt_optimization_feedback: str
    iteration_count: int
    error_state: Optional[str]
    workflow_stage: str
    autonomous_config: AutonomousConfig
    start_time: datetime
    score_history: List[float]
    last_improvement_iteration: int

def initialize_agents():
    """initialize agents"""
    global prompt_optimizer, recommendation_agent, evaluation_agent

    prompt_optimizer = PromptOptimizerAgent()
    recommendation_agent = RecommendationAgent()
    evaluation_agent = EvaluationAgent(recommendation_agent)

    logger.important("✅ all agents initialized")


def prompt_optimization_node(state: SystemState) -> SystemState:
    """Node for prompt optimization"""
    try:
        logger.important("🔧 start prompt optimization, may take 2-3 minutes")
        result = prompt_optimizer.optimize_prompts(state)
        logger.important("✅ prompt optimization completed")
        return {**state, **result}
    except Exception as e:
        return {**state, "error_state": f"Prompt optimization error: {str(e)}"}


def recommendation_node(state: SystemState) -> SystemState:
    """Node for generating recommendations with partial user information"""
    try:
        user_profile = state.get("current_user_profile")
        if not user_profile:
            logger.error("recommendation node error: no user profile provided")
            return {**state, "error_state": "No user profile provided"}

        # Use partial tags to generate recommendations (simulate incomplete information)
        user_tags = user_profile.partial_tags

        # Get recommendations (using partial information)
        recommendations = recommendation_agent.recommend_story(
            user_tags=user_tags,
            count=10,  # recommend 10 stories
            candidate_count=50,
        )

        return {
            **state,
            "recommendations": recommendations,
            "workflow_stage": "recommendation_complete",
        }
    except Exception as e:
        logger.error(f"❌ recommendation node error: {str(e)}")
        return {**state, "error_state": f"Recommendation error: {str(e)}"}


def evaluation_node(state: SystemState) -> SystemState:
    """Node for evaluation comparing partial vs full information recommendations"""
    try:
        result = evaluation_agent.evaluate_recommendations(state)
        iteration_count = state.get("iteration_count", 0) + 1
        result["iteration_count"] = iteration_count

        # Track score history and improvement
        eval_result = result.get("evaluation_result")
        if eval_result:
            current_score = eval_result.eval_score
            score_history = state.get("score_history", [])
            score_history.append(current_score)
            result["score_history"] = score_history

            # Show important evaluation metrics (brief but useful)
            logger.important(
                f" {iteration_count}th evaluation - total score: {eval_result.eval_score:.3f}"
            )
            logger.important(
                f"   Precision: {eval_result.precision:.3f}"
            )

            # Show full recommended IDs (for comparison)
            if (
                hasattr(eval_result, "ground_truth_ids")
                and eval_result.ground_truth_ids
            ):
                logger.important(
                    f"   full information recommended IDs: {eval_result.ground_truth_ids[:10]}"
                )

            # Detailed information recorded in log file
            logger.info("detailed evaluation results:")
            logger.info(f"   Precision: {eval_result.precision:.3f}")

            # Check if this is an improvement
            if len(score_history) > 1:
                previous_score = score_history[-2]
                min_improvement = state.get(
                    "autonomous_config", AutonomousConfig()
                ).min_improvement_threshold
                if current_score > previous_score + min_improvement:
                    result["last_improvement_iteration"] = iteration_count
                    logger.important(
                        f"score improved: {previous_score:.3f} → {current_score:.3f} (+{current_score - previous_score:.3f})"
                    )
                else:
                    result["last_improvement_iteration"] = state.get(
                        "last_improvement_iteration", 0
                    )
                    logger.info(
                        f"score changed: {previous_score:.3f} → {current_score:.3f} ({current_score - previous_score:+.3f})"
                    )
            else:
                result["last_improvement_iteration"] = iteration_count
                logger.important(f"initial score: {current_score:.3f}")

        return {**state, **result}
    except Exception as e:
        logger.error(f"❌ evaluation node error: {str(e)}")
        return {**state, "error_state": f"Evaluation error: {str(e)}"}


def autonomous_router(
    state: SystemState,
) -> Literal["continue_optimization", "complete"]:
    """
    Autonomous routing decision based on multiple criteria:
    1. Target score achievement
    2. Time budget constraints
    3. Maximum iterations
    4. Early stopping (no improvement)
    5. Adaptive target adjustment
    """

    # Check for errors
    if state.get("error_state"):
        logger.error("❌ Stopping due to error")
        return "complete"

    # Get configuration and current state
    config = state.get("autonomous_config", AutonomousConfig())
    evaluation_result = state.get("evaluation_result")
    iteration_count = state.get("iteration_count", 0)
    start_time = state.get("start_time", datetime.now())
    score_history = state.get("score_history", [])
    last_improvement = state.get("last_improvement_iteration", 0)

    # Check time budget
    elapsed_time = datetime.now() - start_time
    if elapsed_time.total_seconds() / 60 > config.time_budget_minutes:
        logger.important(
            f"⏰ Time budget exceeded ({config.time_budget_minutes} minutes)"
        )
        return "complete"

    # Check maximum iterations
    if iteration_count >= config.max_iterations:
        logger.important(f"Maximum iterations reached ({config.max_iterations})")
        return "complete"

    if not evaluation_result:
        return "continue_optimization"

    current_score = evaluation_result.eval_score

    # Check if target score achieved
    target_score = config.target_score

    # Adaptive target adjustment based on progress
    if config.adaptive_target and len(score_history) > 2:
        # If we're making steady progress but target seems too high, adjust it
        recent_trend = score_history[-3:]
        if all(
            recent_trend[i] < recent_trend[i + 1] for i in range(len(recent_trend) - 1)
        ):
            # Positive trend, but check if target is realistic
            max_observed = max(score_history)
            if target_score > max_observed + 0.1:  # Target seems too high
                adjusted_target = max_observed + 0.05
                logger.important(
                    f"adapting target score: {target_score:.3f} → {adjusted_target:.3f}"
                )
                target_score = adjusted_target

    if current_score >= target_score:
        logger.important(
            f"target score achieved! {current_score:.3f} >= {target_score:.3f}"
        )
        return "complete"

    # Early stopping check
    iterations_without_improvement = iteration_count - last_improvement
    if iterations_without_improvement >= config.early_stopping_patience:
        logger.important(
            f"early stopping: no improvement for {iterations_without_improvement} iterations"
        )
        return "complete"

    # Continue optimization - only record detailed information to log file
    remaining_time = config.time_budget_minutes - (elapsed_time.total_seconds() / 60)
    logger.info(
        f"Continue optimization (Score: {current_score:.3f}/{target_score:.3f}, "
        f"Iteration: {iteration_count}/{config.max_iterations}, "
        f"Time left: {remaining_time:.1f}min)"
    )

    return "continue_optimization"


def build_sekai_workflow() -> StateGraph:
    """
    Build the complete Sekai recommendation system workflow
    """
    # Create the state graph
    workflow = StateGraph(SystemState)

    # Add nodes for each agent (remove tag simulation)
    workflow.add_node("recommendation_generation", recommendation_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("prompt_optimization", prompt_optimization_node)

    # Define the workflow edges (start from recommendation)
    workflow.add_edge(START, "recommendation_generation")
    workflow.add_edge("recommendation_generation", "evaluation")

    # Conditional edge for autonomous optimization loop
    workflow.add_conditional_edges(
        "evaluation",
        autonomous_router,
        {"continue_optimization": "prompt_optimization", "complete": END},
    )

    # Loop back from optimization to recommendation (skip tag simulation)
    workflow.add_edge("prompt_optimization", "recommendation_generation")

    return workflow


async def run_autonomous_recommendation(
    autonomous_config: Optional[AutonomousConfig] = None,
):
    """
    Main function - run the Sekai Self Evolving System (SSES)

    Args:
        autonomous_config: autonomous optimization configuration, if None then use default configuration
    """
    logger.important("Starting Sekai Self Evolving System (SSES)")

    # 1. Configure DSPy
    configure_dspy()

    # 2. Initialize agents
    initialize_agents()

    # 3. Prepare configuration
    if autonomous_config is None:
        autonomous_config = create_autonomous_config()

    logger.important(
        f"🎯 Configuration: target {autonomous_config.target_score}, max {autonomous_config.max_iterations} rounds, {autonomous_config.time_budget_minutes} minutes"
    )

    # 4. Build workflow and callbacks
    workflow = build_sekai_workflow()
    compiled_workflow = workflow.compile()
    metrics_callback = MetricsCallbackHandler(logger=logger)

    # 5. Prepare initial state
    sample_user = create_sample_user_profile()
    initial_state: SystemState = {
        "current_user_profile": sample_user,
        "recommendations": [],
        "evaluation_result": None,
        "prompt_optimization_feedback": "",
        "iteration_count": 0,
        "error_state": None,
        "workflow_stage": "initialized",
        "autonomous_config": autonomous_config,
        "start_time": datetime.now(),
        "score_history": [],
        "last_improvement_iteration": 0,
    }

    logger.info(f"test user: {sample_user.username}")
    logger.info(f"full profile: {sample_user.full_profile}")
    user_info = sample_user.get_info()
    logger.important(
        f"user settings: {user_info['partial_tags']}/{user_info['total_tags']} tags ({user_info['partial_tag_ratio']:.1%})"
    )

    logger.important("start autonomous optimization loop")

    try:
        score_history = []
        config = {"callbacks": [metrics_callback]}
        final_state = initial_state.copy()  # save final state

        # 7. Execute the workflow and display progress
        async for output in compiled_workflow.astream(initial_state, config=config):
            for node_name, node_output in output.items():
                logger.info(f"execute node: {node_name}")
                
                final_state.update(node_output)

                # Show evaluation result summary (important information is marked in evaluation_node)
                if (
                    "evaluation_result" in node_output
                    and node_output["evaluation_result"]
                ):
                    # Detailed score change recorded in log file
                    current_history = node_output.get("score_history", [])
                    if len(current_history) > 1:
                        history_str = ", ".join(
                            f"{score:.3f}"
                            + (" ← latest" if i == len(current_history) - 1 else "")
                            for i, score in enumerate(current_history)
                        )
                        logger.important(f"score change: [{history_str}]")

                    score_history = current_history

                # Show error
                if node_output.get("error_state"):
                    logger.error(f"❌ error: {node_output['error_state']}")

        _print_final_summary(
            score_history, initial_state["start_time"], metrics_callback
        )
        
        logger.important("=" * 60)
        logger.important("final recommendations:")
        final_recommendations = final_state.get("recommendations", [])
        if final_recommendations:
            final_story_ids = [rec.get("story_id") for rec in final_recommendations if rec.get("story_id")]
            logger.important(f"📚 final recommended story IDs: {final_story_ids}")

    except Exception as e:
        logger.error(f"❌ execution failed: {str(e)}")


def _print_final_summary(
    score_history: List[float], start_time: datetime, metrics: MetricsCallbackHandler
):
    """print final summary"""
    # print performance summary
    if score_history:
        initial_score = score_history[0]
        final_score = score_history[-1]
        total_improvement = final_score - initial_score
        elapsed_time = datetime.now() - start_time

        logger.important("=" * 60)
        logger.important(f"optimization summary:")
        logger.important(
            f"   initial → final: {initial_score:.3f} → {final_score:.3f} (Δ{total_improvement:+.3f})"
        )
        logger.important(
            f"   iterations: {len(score_history)}, time: {elapsed_time.total_seconds() / 60:.1f} minutes"
        )
        logger.important("✅ autonomous optimization completed!")
    else:
        logger.warning("⚠️ no score history, cannot generate optimization summary.")

    # get metrics summary
    metrics_summary = metrics.get_summary()

    # calculate cost estimation
    total_cost = _calculate_total_cost(metrics_summary)

    # print token consumption and cost analysis
    logger.important("=" * 60)
    logger.important("💰 token consumption and cost analysis:")
    logger.important(f"   🔢 LLM call total: {metrics_summary['successful_requests']}")
    logger.important(f"   📊 total token consumption: {metrics_summary['total_tokens']:,}")
    logger.important(f"      - input token: {metrics_summary['prompt_tokens']:,}")
    logger.important(f"      - output token: {metrics_summary['completion_tokens']:,}")
    logger.important(f"   💵 estimated total cost: ${total_cost:.4f}")
    logger.important(
        f"   ⏱️ average call duration: {metrics_summary['average_step_duration_ms']:.0f} ms"
    )

    # calculate efficiency metrics
    if score_history and len(score_history) > 1:
        total_improvement = score_history[-1] - score_history[0]
        if total_improvement > 0:
            cost_per_improvement = total_cost / total_improvement
            tokens_per_improvement = metrics_summary["total_tokens"] / total_improvement


def _calculate_total_cost(metrics_summary: Dict[str, Any]) -> float:
    """
    calculate total API call cost

    Args:
        metrics_summary: metrics summary dictionary

    Returns:
        estimated total cost (USD)
    """
    try:
        config = load_config()
        total_cost = 0.0

        # get input and output token count
        input_tokens = metrics_summary.get("prompt_tokens", 0)
        output_tokens = metrics_summary.get("completion_tokens", 0)
        total_calls = metrics_summary.get("successful_requests", 0)

        if total_calls == 0:
            return 0.0

        agents_count = len(config.get("agents", {}))
        if agents_count == 0:
            return 0.0

        avg_input_per_agent = input_tokens // agents_count
        avg_output_per_agent = output_tokens // agents_count

        for agent_name, agent_config in config.get("agents", {}).items():
            provider = agent_config.get("provider", "")
            model = agent_config.get("model", "")

            if provider and model:
                try:
                    agent_cost = estimate_cost(
                        provider, model, avg_input_per_agent, avg_output_per_agent
                    )
                    total_cost += agent_cost
                    logger.info(
                        f"   {agent_name}: ${agent_cost:.4f} ({provider}/{model})"
                    )
                except Exception as e:
                    logger.warning(f"    cannot calculate {agent_name} cost: {e}")

        return total_cost

    except Exception as e:
        logger.warning(f"cost calculation failed: {e}")


def main():
    """Main function - start the Sekai Self Evolving System (SSES)"""
    try:
        print("Sekai Self Evolving System (SSES)")

        config = AutonomousConfig(
            max_iterations=3, time_budget_minutes=3, target_score=0.7
        )

        # Run the autonomous recommendation system
        asyncio.run(run_autonomous_recommendation(autonomous_config=config))

    except KeyboardInterrupt:
        print("User interrupted")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
    finally:
        logger.important("SSES closed")


if __name__ == "__main__":
    main()
