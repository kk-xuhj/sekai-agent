from typing import List

import dspy

from config import create_llm
from utils.logger import setup_logger

logger = setup_logger()
dspy.disable_logging()
dspy.settings.configure(cache_turn_on=True)


class DSPyLMAdapter(dspy.LM):
    """Adapt LangChain LLM to DSPy LM"""

    def __init__(self, langchain_llm):
        super().__init__(model="langchain_adapter")
        self.langchain_llm = langchain_llm

        # DSPy needs the basic attributes, set default values to avoid attribute access errors
        self.kwargs = {
            "temperature": 0.1,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        self.history = []
        self.provider = "custom"

        # add other attributes that may be needed
        self._cached_kwargs = {}
        self._request_counter = 0

    def __call__(self, prompt=None, messages=None, **kwargs):
        try:
            self._request_counter += 1

            # save kwargs
            self.kwargs.update(kwargs)
            self._cached_kwargs = kwargs.copy()

            # handle input from different dspy modules
            prompt_text = ""
            input_type = "unknown"

            if messages is not None:
                input_type = "messages"
                if isinstance(messages, list):
                    # try to extract 'content' from dictionary, otherwise convert to string
                    content_parts = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            content_parts.append(msg.get("content", str(msg)))
                        else:
                            content_parts.append(str(msg))
                    prompt_text = "\n".join(content_parts)
                else:
                    prompt_text = str(messages)
            elif prompt is not None:
                input_type = "prompt"
                prompt_text = prompt
            else:
                prompt_text = ""

            # call LangChain LLM
            response = self.langchain_llm.invoke(prompt_text)
            result = [response.content]

            # save to history
            self.history.append(
                {
                    "prompt": prompt_text,
                    "response": response.content,
                    "kwargs": kwargs.copy(),
                    "request_id": self._request_counter,
                    "input_type": input_type,
                }
            )

            return result

        except Exception as e:
            logger.error(f"❌ DSPy LM call failed: {e}")
            return [""]

    def basic_request(self, messages=None, **kwargs):
        """DSPy standard method"""
        # compatible with old call
        return self.__call__(messages=messages, **kwargs)

    def generate(self, messages=None, **kwargs):
        """generate method"""
        return self.__call__(messages=messages, **kwargs)

    def request(self, messages=None, **kwargs):
        """request method"""
        return self.__call__(messages=messages, **kwargs)

    def __getitem__(self, key):
        """support dictionary access"""
        return self.kwargs.get(key)

    def get(self, key, default=None):
        """support get method"""
        return self.kwargs.get(key, default)


def configure_dspy():
    """configure DSPy LM"""
    try:
        # use recommendation configured LLM as DSPy's default LM
        llm = create_llm("recommendation")

        # use DSPy LM adapter
        dspy_lm = DSPyLMAdapter(llm)
        dspy.configure(lm=dspy_lm)

        logger.important("✅ DSPy LM configured")

    except Exception as e:
        logger.error(f"❌ DSPy LM configuration failed: {e}")
        raise e


def is_dspy_configured() -> bool:
    """check if DSPy is configured"""
    return dspy.settings.lm is not None


def ensure_dspy_configured():
    """ensure DSPy is configured, if not, configure it automatically"""
    if not is_dspy_configured():
        configure_dspy()


class StoryRecommendationSignature(dspy.Signature):
    """story recommendation input and output signature"""

    # input fields
    user_tags = dspy.InputField(desc="User preferences, comma separated")
    available_story_ids = dspy.InputField(desc="Available story IDs, comma separated")
    candidates_info = dspy.InputField(desc="Candidate story details")
    recommendation_count = dspy.InputField(desc="Number of stories to recommend")

    # output fields
    recommended_ids = dspy.OutputField(
        desc="Recommended story IDs, comma separated, only from available IDs"
    )


class DSPyRecommendationModule(dspy.Module):
    """DSPy recommendation module - an optimized recommendation system"""

    def __init__(self):
        super().__init__()

        # ensure DSPy is configured
        ensure_dspy_configured()

        # directly initialize DSPy component
        self.recommend = dspy.ChainOfThought(StoryRecommendationSignature, 
                                             rationale_type="Let me analyze the user's partial preferences and inference his more preferred tags, then recommend the stories with the highest tag overlap and thematic similarity.")

        logger.important("✅ DSPy recommendation module initialized")

    def forward(
        self,
        user_tags: List[str],
        available_story_ids: List[int],
        candidates_info: str,
        recommendation_count: int = 5,
    ) -> dspy.Prediction:
        """
        forward inference, generate recommendation results.
        this method returns the original DSPy Prediction object, so the optimizer can access the original LLM output.
        the caller is responsible for parsing `prediction.recommended_ids`.

        Args:
            user_tags: user preferences tags
            available_story_ids: available story IDs
            candidates_info: candidate story details
            recommendation_count: number of stories to recommend

        Returns:
            dspy.Prediction object, containing `recommended_ids` field
        """
        try:
            # --- input formatting ---
            safe_count = str(recommendation_count)

            if isinstance(available_story_ids, str):
                available_ids_str = available_story_ids
            else:
                available_ids_str = ", ".join(map(str, available_story_ids))

            if isinstance(user_tags, str):
                user_tags_str = user_tags
            else:
                user_tags_str = ", ".join(user_tags) if user_tags else "no specific preferences"

            # call DSPy inference
            prediction = self.recommend(
                user_tags=user_tags_str,
                available_story_ids=available_ids_str,
                candidates_info=candidates_info,
                recommendation_count=safe_count,
            )

            # check if prediction has recommended_ids attribute
            if not hasattr(prediction, "recommended_ids"):
                logger.warning("❌ DSPy prediction missing recommended_ids attribute")

            return prediction

        except Exception as e:
            logger.error(f"❌ DSPy recommendation failed: {e}")
            return dspy.Prediction(recommended_ids="")


# global instance and utility functions
_global_dspy_module = None


def get_dspy_recommendation_module() -> DSPyRecommendationModule:
    """get global DSPy recommendation module"""
    global _global_dspy_module

    if _global_dspy_module is None:
        _global_dspy_module = DSPyRecommendationModule()

    return _global_dspy_module


def reset_dspy_module():
    """reset DSPy module (for testing or reinitialization)"""
    global _global_dspy_module
    _global_dspy_module = None
