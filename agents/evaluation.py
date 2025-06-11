import random
from dataclasses import dataclass
from typing import Any, Dict, List

from agents.recommendation import RecommendationAgent
from config import create_llm, validate_environment
from utils.evaluation_metrics import calculate_all_metrics
from utils.logger import setup_logger

logger = setup_logger()


@dataclass
class EvaluationResult:
    """simplified evaluation result - using precision as the core metric"""

    eval_score: float  # comprehensive score (precision)
    predicted_tags: List[str]  # tags used under partial information
    predicted_recommendations: List[str]  # recommendations under partial information
    ground_truth_recommendations: List[str]  # recommendations under full information
    detailed_analysis: str

    # core recommendation metrics
    precision: float  # precision/accuracy

    # partial information
    partial_info_ratio: float  # partial information ratio
    recommendation_confidence: float  # recommendation confidence


class EvaluationAgent:
    """
    Enhanced evaluation agent comparing partial vs full information recommendations
    """

    def __init__(self, recommendation_agent: RecommendationAgent):
        # check if environment is available
        env_status = validate_environment()
        if not env_status.get("evaluation", False):
            logger.warning("⚠️ Evaluation Agent: API key not set, will fail at runtime")

        self.llm = create_llm("evaluation")
        self.recommendation_agent = recommendation_agent
        logger.important("✅ Evaluation Agent initialized")

    def evaluate_recommendations(self, state) -> Dict[str, Any]:
        """
        Compare partial-information recommendations against full-information ground truth
        """
        logger.important("📊 Evaluation Agent: start evaluating recommendations...")

        user_profile = state.get("current_user_profile")
        partial_recommendations = state.get("recommendations", [])

        if not user_profile or not partial_recommendations:
            logger.error("❌ missing user profile or recommendations")
            return {"error_state": "Missing user profile or recommendations"}

        try:
            # 1. get story IDs based on partial information recommendations
            partial_rec_ids = [
                str(rec.get("story_id", rec)) if isinstance(rec, dict) else str(rec)
                for rec in partial_recommendations
            ]

            # 2. get ground truth recommendations using full user information
            ground_truth_recommendations = self._get_ground_truth_recommendations(
                user_profile
            )

            ground_truth_ids = [
                str(rec.get("story_id", rec)) if isinstance(rec, dict) else str(rec)
                for rec in ground_truth_recommendations
            ]

            logger.info(f"📊 partial information recommendations: {partial_rec_ids[:5]}...")
            logger.info(f"📊 ground truth recommendations (GT): {ground_truth_ids[:5]}...")

            # 3. use unified evaluation library to calculate all metrics
            metrics = calculate_all_metrics(partial_rec_ids, ground_truth_ids)

            # 4. get user information ratio and calculate confidence
            user_info = user_profile.get_info()
            partial_info_ratio = user_info["partial_tag_ratio"]
            recommendation_confidence = self._calculate_confidence(
                metrics["precision"], partial_info_ratio
            )

            # 5. generate detailed analysis
            detailed_analysis = f"""
            === partial information vs full information recommendation comparison ===
            user: {user_info["username"]}
            available information: {user_info["partial_tags"]}/{user_info["total_tags"]} tags ({partial_info_ratio:.1%})
            
            === evaluation metrics ===
            • precision (Precision): {metrics["precision"]:.3f} - how many are truly relevant
            
            === comprehensive score: {metrics["eval_score"]:.3f} ===
            """

            evaluation_result = EvaluationResult(
                eval_score=metrics["eval_score"],
                predicted_tags=user_profile.partial_tags,
                predicted_recommendations=partial_rec_ids,
                ground_truth_recommendations=ground_truth_recommendations,
                detailed_analysis=detailed_analysis,
                precision=metrics["precision"],
                partial_info_ratio=partial_info_ratio,
                recommendation_confidence=recommendation_confidence,
            )

            return {
                "evaluation_result": evaluation_result,
                "workflow_stage": "evaluation_complete",
            }

        except Exception as e:
            return {"error_state": f"Evaluation error: {str(e)}"}

    def _get_ground_truth_recommendations(self, user_profile) -> List[Dict[str, Any]]:
        """use full user information to get ground truth recommendations"""
        try:
            # calculate ground truth every time, not using cache
            # this can reflect the impact of DSPy optimization on the recommendation system
            ground_truth_recommendations = self.recommendation_agent.recommend_story(
                user_tags=user_profile.all_tags,  # use full tags
                count=10,
                candidate_count=50,
            )

            logger.info(
                f"✅ got {len(ground_truth_recommendations)} ground truth recommendations"
            )
            return ground_truth_recommendations

        except Exception as e:
            logger.error(f"❌ failed to get ground truth recommendations: {e}")
            raise e

    def _calculate_confidence(
        self, overlap_score: float, partial_info_ratio: float
    ) -> float:
        """calculate recommendation confidence"""
        # confidence based on overlap and available information ratio
        base_confidence = overlap_score * partial_info_ratio

        # add some randomness to simulate real-world scenarios
        noise = random.uniform(-0.05, 0.05)
        return max(0, min(1, base_confidence + noise))
