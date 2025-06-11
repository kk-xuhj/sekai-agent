import time
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class MetricsCallbackHandler(BaseCallbackHandler):
    """Callback handler to collect token usage and timing metrics"""

    def __init__(self, logger=None) -> None:
        super().__init__()
        self.logger = logger
        self.successful_requests = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.steps = []
        self._step_starts: Dict[str, float] = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Record time when LLM call starts"""
        run_id = kwargs["run_id"]
        self._step_starts[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Aggregate token usage and duration when LLM call ends"""
        run_id = kwargs["run_id"]
        start_time = self._step_starts.pop(run_id, time.time())
        duration_ms = (time.time() - start_time) * 1000

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            usage_metadata = response.generations[0][0].message.usage_metadata
            prompt_tokens = usage_metadata.get("input_tokens", 0)
            completion_tokens = usage_metadata.get("output_tokens", 0)
            total_tokens = usage_metadata.get("total_tokens", 0)
        except (IndexError, AttributeError, KeyError) as e:
            pass

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.successful_requests += 1

        self.steps.append(
            {
                "duration_ms": duration_ms,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            "successful_requests": self.successful_requests,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "steps": self.steps,
            "average_step_duration_ms": sum(s["duration_ms"] for s in self.steps)
            / len(self.steps)
            if self.steps
            else 0,
        }
