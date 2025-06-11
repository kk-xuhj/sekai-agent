from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

from utils.llm_factory import LLMFactory

_config_data: Optional[Dict[str, Any]] = None
_llm_factory: Optional[LLMFactory] = None


@dataclass
class AutonomousConfig:
    """autonomous optimization loop configuration"""

    target_score: float = 0.85
    max_iterations: int = 5
    time_budget_minutes: float = 10.0
    min_improvement_threshold: float = 0.02
    adaptive_target: bool = True
    early_stopping_patience: int = 2


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: configuration file path

    Returns:
        configuration data dictionary
    """
    global _config_data

    if _config_data is None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML configuration file format error: {e}")

    return _config_data


def get_llm_factory() -> LLMFactory:
    """
    Get LLM factory instance

    Returns:
        LLM factory instance
    """
    global _llm_factory

    if _llm_factory is None:
        config_data = load_config()
        _llm_factory = LLMFactory(config_data)

    return _llm_factory


def create_llm(agent_type: str, **override_params):
    """
    Create LLM instance

    Args:
        agent_type: agent type (prompt_optimizer, recommendation, evaluation)
        **override_params: override default configuration parameters

    Returns:
        configured LLM instance
    """
    factory = get_llm_factory()
    return factory.create_llm(agent_type, **override_params)


def create_autonomous_config(**kwargs) -> AutonomousConfig:
    """
    Create autonomous optimization configuration

    Args:
        **kwargs: override default configuration parameters

    Returns:
        autonomous optimization configuration instance
    """
    config_data = load_config()
    autonomous_config = config_data.get("autonomous", {})
    autonomous_config.update(kwargs)

    return AutonomousConfig(**autonomous_config)


def switch_to_preset(preset_name: str):
    """
    Switch to preset configuration

    Args:
        preset_name: preset name (openai, anthropic)
    """
    global _config_data, _llm_factory

    config_data = load_config()

    if preset_name not in config_data.get("presets", {}):
        raise ValueError(f"unknown preset configuration: {preset_name}")

    # update agent configuration
    preset_config = config_data["presets"][preset_name]
    config_data["agents"].update(preset_config)

    # reset factory instance to use new configuration
    _llm_factory = None

    print(f"🔄 switched to {preset_name.upper()} preset configuration")


def get_config_summary() -> str:
    """get configuration summary"""
    factory = get_llm_factory()
    return factory.get_config_summary()


def validate_environment() -> Dict[str, bool]:
    """validate environment variables"""
    factory = get_llm_factory()
    return factory.validate_environment()


def get_model_pricing(provider: str, model: str) -> Dict[str, float]:
    """get model pricing information"""
    factory = get_llm_factory()
    return factory.get_model_pricing(provider, model)


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float:
    """estimate usage cost"""
    factory = get_llm_factory()
    return factory.estimate_cost(provider, model, input_tokens, output_tokens)


# backward compatibility
class ModelProvider:
    """backward compatible model provider enumeration"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class AgentType:
    """backward compatible agent type enumeration"""

    PROMPT_OPTIMIZER = "prompt_optimizer"
    RECOMMENDATION = "recommendation"
    EVALUATION = "evaluation"


__all__ = [
    "load_config",
    "get_llm_factory",
    "create_llm",
    "create_autonomous_config",
    "switch_to_preset",
    "get_config_summary",
    "validate_environment",
    "get_model_pricing",
    "estimate_cost",
    "AutonomousConfig",
    "ModelProvider",
    "AgentType",
]
