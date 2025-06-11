from typing import Any, Dict, List

from config import validate_environment
from utils.dspy import get_dspy_recommendation_module
from utils.logger import setup_logger
from utils.vector_store import get_vector_store

logger = setup_logger()


class RecommendationAgent:
    """Story recommendation agent: vector search + DSPy intelligent selection"""

    def __init__(self):
        """Initialize recommendation agent"""

        env_status = validate_environment()
        if not env_status.get("recommendation", False):
            logger.warning("⚠️ Recommendation Agent: API key not set, will fail at runtime")

        self._init_vector_store()

        self._init_dspy_module()
        logger.important("✅ Recommendation Agent initialization completed")

    def _init_vector_store(self):
        """Initialize vector store"""
        try:
            self.vector_store = get_vector_store()
            logger.info("Vector store initialization successful")
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            raise e

    def _init_dspy_module(self):
        """
        Initialize DSPy recommendation module placeholder.
        Actual module retrieval will be done dynamically at each recommendation to ensure using the latest optimized version.
        """
        try:
            get_dspy_recommendation_module()
            logger.info("DSPy recommendation module activated (latest version will be retrieved dynamically)")
        except ImportError as e:
            logger.error(f"❌ DSPy module import failed: {e}")
            raise RuntimeError("DSPy module is required, please ensure it is properly installed and configured") from e

    def recommend_story_id(
        self,
        user_tags: List[str] = None,
        count: int = 5,
        candidate_count: int = 30,
    ) -> List[int]:
        """
        Recommend multiple story IDs

        Args:
            user_tags: List of user preference tags
            count: Number of stories to recommend
            candidate_count: Number of candidate stories

        Returns:
            List of recommended story IDs
        """
        try:
            # Build search query (based on user tags)
            search_query = _build_dense_query_sentence(user_tags)

            # 1. Use vector search to get candidate stories
            vector_store = self.vector_store
            search_results = vector_store.similarity_search_by_query(
                query=search_query, user_tags=user_tags, k=candidate_count
            )

            if not search_results:
                logger.warning("❌ Vector search found no candidate stories")
                return []

            # 2. Cache search results for subsequent use
            self._cached_search_results = search_results
            self._cached_search_query = search_query

            # 3. Prepare candidate story data
            candidates_text = _format_candidates(search_results)

            # 4. Use DSPy to select final recommendations
            story_ids = self._dspy_select_stories(
                user_tags=user_tags or [], candidates_text=candidates_text, count=count
            )

            if story_ids:
                logger.important(f"✅ Recommendation completed, selected story IDs: {story_ids}")
            else:
                logger.warning("❌ LLM selection failed")

            return story_ids

        except Exception as e:
            logger.error(f"❌ Recommendation process failed: {e}")
            return []

    def recommend_story(
        self,
        user_tags: List[str] = None,
        count: int = 10,
        candidate_count: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Recommend multiple complete story information

        Args:
            user_tags: List of user preference tags
            count: Number of stories to recommend
            candidate_count: Number of candidate stories

        Returns:
            List of recommended complete story information
        """
        try:
            logger.important(f"🎯 Starting complete story recommendation - target story count: {count}")

            # 1. Call recommend_story_id to get recommended story IDs
            story_ids = self.recommend_story_id(
                user_tags=user_tags,
                count=count,
                candidate_count=candidate_count,
            )

            if not story_ids:
                logger.warning("❌ No recommendation IDs obtained")
                return []

            # 2. Extract complete story information from cached search results
            if not hasattr(self, "_cached_search_results"):
                logger.error("❌ Missing cached search results")
                return []

            search_results = self._cached_search_results
            recommended_stories = []
            search_dict = {
                doc.metadata["story_id"]: (doc, similarity)
                for doc, similarity in search_results
            }

            for story_id in story_ids:
                if story_id in search_dict:
                    doc, similarity = search_dict[story_id]
                    story_info = {
                        "story_id": story_id,
                        "title": doc.metadata["title"],
                        "intro": doc.metadata["intro"],
                        "tags": doc.metadata["tags"],
                        "created_at": doc.metadata.get("created_at"),
                        "similarity": similarity,
                        "source": doc.metadata.get("source", "milvus"),
                    }
                    recommended_stories.append(story_info)
                else:
                    logger.warning(f"⚠️ Story ID {story_id} not found in search results")

            logger.important(f"✅ Complete recommendation finished, returning {len(recommended_stories)} stories")
            return recommended_stories

        except Exception as e:
            logger.error(f"❌ Complete recommendation process failed: {e}")
            return []

    def _parse_dspy_output(
        self, output_text: str, available_ids: List[int], target_count: int
    ) -> List[int]:
        """Parse story IDs from DSPy output"""
        try:
            # Extract numeric IDs
            id_strings = [s.strip() for s in output_text.split(",")]
            parsed_ids = []

            for id_str in id_strings:
                try:
                    story_id = int(id_str)
                    if story_id in available_ids and story_id not in parsed_ids:
                        parsed_ids.append(story_id)
                except ValueError:
                    continue

            # Ensure sufficient quantity
            if len(parsed_ids) < target_count:
                # Supplement from available IDs
                remaining_count = target_count - len(parsed_ids)
                backup_ids = [
                    id
                    for id in available_ids[: remaining_count * 2]
                    if id not in parsed_ids
                ]
                parsed_ids.extend(backup_ids[:remaining_count])

            return parsed_ids[:target_count]

        except Exception as e:
            logger.error(f"❌ ID parsing failed: {e}")
            return available_ids[:target_count]

    def _dspy_select_stories(
        self, user_tags: List[str], candidates_text: str, count: int
    ) -> List[int]:
        """Use DSPy module to select stories"""
        try:
            # Extract available IDs from cached search results
            available_story_ids = []
            story_similarity = []
            if hasattr(self, "_cached_search_results"):
                for doc, similarity in self._cached_search_results:
                    story_id = doc.metadata.get("story_id")
                    if story_id is not None:
                        available_story_ids.append(story_id)
                        story_similarity.append(similarity)

            if not available_story_ids:
                logger.error("❌ DSPy selection failed: no available story IDs")
                return []

            # ⚠️ Dynamically get the latest DSPy module to ensure using the optimized version
            dspy_module = get_dspy_recommendation_module()

            # Call DSPy module, which now returns a Prediction object
            prediction = dspy_module.forward(
                user_tags=user_tags,
                available_story_ids=available_story_ids,
                candidates_info=candidates_text,
                recommendation_count=count,
            )

            # Parse IDs from prediction
            raw_output = ""
            if hasattr(prediction, "recommended_ids") and prediction.recommended_ids:
                raw_output = prediction.recommended_ids
            else:
                logger.warning(
                    "DSPy prediction response missing 'recommended_ids' attribute or its value is empty"
                )
                # Trigger fallback below
                raise ValueError("Empty or invalid DSPy output")

            recommended_ids = self._parse_dspy_output(
                raw_output, available_story_ids, count
            )

            return recommended_ids

        except Exception as e:
            logger.error(f"❌ DSPy selection failed: {e}")
            # Simple fallback: return first few IDs
            if hasattr(self, "_cached_search_results") and self._cached_search_results:
                fallback_ids = []
                for doc, similarity in self._cached_search_results[:count]:
                    story_id = doc.metadata.get("story_id")
                    if story_id is not None:
                        fallback_ids.append(story_id)
                logger.warning(
                    f"⚠️ Using fallback approach, returning first {len(fallback_ids)} IDs: {fallback_ids}"
                )
                return fallback_ids
            return []


def _build_dense_query_sentence(user_tags: list[str]) -> str:
    if not user_tags:
        return "Looking for a meaningful, immersive story with emotional depth and strong character arcs."

    base = "Looking for a story that includes"

    # Standardize tags, replace underscores with spaces (embedding works better)
    tags_nl = [tag.replace("-", " ") for tag in user_tags]

    # Construct natural language query
    if len(tags_nl) == 1:
        return f"{base} {tags_nl[0]}."

    elif len(tags_nl) == 2:
        return f"{base} {tags_nl[0]} and {tags_nl[1]}."

    else:
        # Last one uses "and", others use comma
        joined = ", ".join(tags_nl[:-1]) + f", and {tags_nl[-1]}"
        return f"{base} {joined}."


def _format_candidates(search_results: List) -> str:
    """Format candidate stories as text, highlighting story IDs"""
    candidates = []

    for i, (doc, similarity) in enumerate(search_results, 1):
        metadata = doc.metadata
        candidate_text = f"""Story {i}:
**ID: {metadata["story_id"]}**
Title: {metadata["title"]}
Intro: {metadata["intro"][:200]}{"..." if len(metadata["intro"]) > 200 else ""}
Tags: {", ".join(metadata["tags"]) if metadata["tags"] else "None"}
Similarity: {similarity:.3f}
"""
        candidates.append(candidate_text)

    return "\n".join(candidates)
