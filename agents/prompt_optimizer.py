import re
from typing import Any, Dict

import dspy
from dspy.teleprompt import MIPROv2

import utils.dspy as dspy_utils
from config import validate_environment
from utils.dspy import get_dspy_recommendation_module
from utils.evaluation_metrics import calculate_all_metrics
from utils.logger import setup_logger
from utils.vector_store import get_vector_store
from agents.recommendation import _format_candidates
import traceback
from agents.recommendation import _build_dense_query_sentence


logger = setup_logger()
dspy.disable_logging()
dspy.settings.configure(cache_turn_on=True)


class PromptOptimizerAgent:
    """
    DSPy automatic prompt optimizer
    Using MIPROv2 to automatically optimize the prompt of the recommendation module
    """

    def __init__(self):
        # Import here to avoid circular imports
        # Check if environment is available
        env_status = validate_environment()
        if not env_status.get("prompt_optimizer", False):
            logger.warning("⚠️ Prompt Optimizer: API key not set, will fail at runtime")

        self.dspy_optimizer = None  # MIPROv2 optimizer
        self.trainset = []  # training data set

        self._init_dspy_optimizer()
        logger.important("✅ Prompt Optimizer Agent initialization completed")

    def _init_dspy_optimizer(self):
        """Initialize DSPy optimizer"""
        try:
            # Check if DSPy is already configured (by main.py)
            if dspy.settings.lm is None:
                raise RuntimeError(
                    "DSPy LM not configured, please call _configure_dspy() in main.py first"
                )

            logger.info(f"🔧 Using configured DSPy LM: {type(dspy.settings.lm)}")

            self.dspy_optimizer = MIPROv2(
                metric=self._evaluation_metric,
                auto="light",
                verbose=False,
                track_stats=False,
                log_dir="logs/dspy_logs",
            )

            logger.info("✅ DSPy MIPROv2 optimizer initialization successful")

        except ImportError as e:
            logger.error(f"❌ DSPy import failed: {e}")
            self.use_dspy = False
        except Exception as e:
            logger.error(f"❌ DSPy optimizer initialization failed: {e}")
            self.use_dspy = False

    def _evaluation_metric(self, example, prediction, trace=None):
        """
        DSPy evaluation metric - using unified evaluation standard
        """
        try:
            if not hasattr(prediction, "recommended_ids") or not hasattr(
                example, "expected_ids"
            ):
                logger.warning("⚠️ Evaluation skipped: prediction or example missing required attributes")
                return 0.0

            predicted_text = prediction.recommended_ids
            if not predicted_text or not isinstance(predicted_text, str):
                logger.warning("📊 Evaluation metric: LLM has no valid output, returning 0.0")
                return 0.0

            predicted_ids_str = [num for num in re.findall(r"\d+", predicted_text)]

            expected_ids_int = example.expected_ids
            expected_ids_str = [str(i) for i in expected_ids_int]

            if not predicted_ids_str or not expected_ids_str:
                logger.warning(
                    f"📊 Evaluation metric: parsed IDs are empty, returning 0.0 (Predicted: {len(predicted_ids_str)}, Expected: {len(expected_ids_str)})"
                )
                return 0.0

            metrics = calculate_all_metrics(predicted_ids_str, expected_ids_str)
            score = metrics["eval_score"]

            return score

        except Exception as e:
            logger.error(f"❌ DSPy evaluation metric calculation failed: {e}")
            return 0.0

    def optimize_prompts(self, state) -> Dict[str, Any]:
        """
        Use DSPy MIPROv2 to automatically optimize the recommendation module
        """
        return self._dspy_optimize(state)

    def _dspy_optimize(self, state) -> Dict[str, Any]:
        """Use DSPy to automatically optimize the prompt"""
        try:
            # Collect training data
            self._collect_training_data(state)

            # Check if training data is enough (MIPROv2 requires at least 2 samples)
            if len(self.trainset) < 2:
                logger.warning(
                    f"⚠️ DSPy training data insufficient ({len(self.trainset)} samples), at least 2 samples are required"
                )
                return {
                    "prompt_optimization_feedback": f"Waiting for more evaluation results, currently {len(self.trainset)} samples, at least 2 samples are required",
                    "workflow_stage": "waiting_for_training_data",
                }

            if len(self.trainset) < 10:
                training_samples = self.trainset
            else:
                training_samples = self.trainset[-10:]

            dspy_module = get_dspy_recommendation_module()

            optimized_module = self.dspy_optimizer.compile(
                student=dspy_module,
                trainset=training_samples,
                requires_permission_to_run=False,
                provide_traceback=False,
            )
            iteration = state.get("iteration_count", 0)
            optimized_module.save(f"logs/optimized_module_{iteration}.json")
            # Update global module
            dspy_utils._global_dspy_module = optimized_module

            logger.important("✅ DSPy prompt optimization completed")

            return {
                "prompt_optimization_feedback": "DSPy MIPROv2 automatic prompt optimization completed",
                "workflow_stage": "dspy_optimization_complete",
            }

        except Exception as e:
            logger.error(f"❌ DSPy optimization failed: {e}")
            return {
                "prompt_optimization_feedback": f"DSPy optimization failed: {str(e)}",
                "workflow_stage": "optimization_failed",
            }

    def _collect_training_data(self, state):
        """
        Collect DSPy training data - stable version
        Key improvement: the input and output of the training samples are based on the recommendation process with full user information, ensuring consistency
        """
        try:
            evaluation_result = state.get("evaluation_result")
            user_profile = state.get("current_user_profile")

            # Only collect training data when there is an evaluation result
            if evaluation_result and user_profile:

                ground_truth_stories = evaluation_result.ground_truth_recommendations
                available_story_ids = [
                    str(story.get("story_id", story))
                    for story in ground_truth_stories
                ]
                candidates_info_text = [
                    story.get("story_info", story) for story in ground_truth_stories
                ]
                ground_truth_int_ids = [story.get("story_id", story) for story in ground_truth_stories]

                if len(self.trainset) < 2:
                    for i in range(6):
                        temp_example = dspy.Example(
                            user_tags=", ".join(user_profile.partial_tags),
                            available_story_ids=", ".join(map(str, available_story_ids)),
                            candidates_info=candidates_info_text,
                            recommendation_count=str(i+5),
                            expected_ids=ground_truth_int_ids[:i+5],  # expected output is ground truth
                        ).with_inputs(
                            "user_tags",
                            "available_story_ids",
                            "candidates_info",
                            "recommendation_count",
                        )
                        self.trainset.append(temp_example)
            else:
                logger.info("📋 Waiting for evaluation results to collect training data...")

        except Exception as e:
            logger.error(f"❌ Failed to collect training data: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
