import os
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from utils.logger import setup_logger

logger = setup_logger()


class LLMFactory:
    """LLM creation factory class"""

    def __init__(self, config_data: Dict[str, Any]):
        """
        initialize LLM factory

        Args:
            config_data: configuration data loaded from YAML
        """
        self.providers = config_data["providers"]
        self.agents = config_data["agents"]

    def create_llm(self, agent_type: str, **override_params):
        """
        create LLM instance based on agent type

        Args:
            agent_type: agent type (prompt_optimizer, recommendation, evaluation)
            **override_params: override default configuration parameters

        Returns:
            configured LLM instance
        """
        if agent_type not in self.agents:
            raise ValueError(f"unknown agent type: {agent_type}")

        # get agent configuration
        agent_config = self.agents[agent_type].copy()
        agent_config.update(override_params)

        provider_name = agent_config["provider"]
        model_name = agent_config["model"]

        # get provider and model configuration
        provider_config = self.providers[provider_name]
        model_config = provider_config["models"][model_name]

        # record pricing information
        pricing = model_config.get("pricing", {})

        # create corresponding LLM based on provider
        if provider_name == "openai":
            return self._create_openai_llm(provider_config, model_config, agent_config)
        elif provider_name == "anthropic":
            return self._create_anthropic_llm(
                provider_config, model_config, agent_config
            )
        elif provider_name == "google":
            return self._create_google_llm(provider_config, model_config, agent_config)
        else:
            raise ValueError(f"unsupported provider: {provider_name}")

    def _create_openai_llm(
        self, provider_config: Dict, model_config: Dict, agent_config: Dict
    ):
        """create OpenAI LLM instance"""
        try:
            api_key = os.getenv(provider_config["api_key_env"])
            if not api_key:
                raise ValueError(f"{provider_config['api_key_env']} environment variable not set")

            return ChatOpenAI(
                model=model_config["name"],
                temperature=agent_config["temperature"],
                max_tokens=min(agent_config["max_tokens"], model_config["max_tokens"]),
                api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "langchain_openai not installed. Please run: pip install langchain_openai"
            )

    def _create_anthropic_llm(
        self, provider_config: Dict, model_config: Dict, agent_config: Dict
    ):
        """create Anthropic LLM instance"""
        try:
            api_key = os.getenv(provider_config["api_key_env"])
            if not api_key:
                raise ValueError(f"{provider_config['api_key_env']} environment variable not set")

            return ChatAnthropic(
                model=model_config["name"],
                temperature=agent_config["temperature"],
                max_tokens=min(agent_config["max_tokens"], model_config["max_tokens"]),
                api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "langchain_anthropic not installed. Please run: pip install langchain_anthropic"
            )

    def _create_google_llm(
        self, provider_config: Dict, model_config: Dict, agent_config: Dict
    ):
        """create Google Gemini LLM instance"""
        try:
            api_key = os.getenv(provider_config["api_key_env"])
            if not api_key:
                raise ValueError(f"{provider_config['api_key_env']} environment variable not set")

            return ChatGoogleGenerativeAI(
                model=model_config["name"],
                temperature=agent_config["temperature"],
                max_output_tokens=min(
                    agent_config["max_tokens"], model_config["max_tokens"]
                ),
                google_api_key=api_key,
            )
        except ImportError:
            raise ImportError(
                "langchain_google_genai not installed. Please run: pip install langchain_google_genai"
            )

    def get_model_pricing(self, provider: str, model: str) -> Dict[str, float]:
        """
        get model pricing information

        Args:
            provider: provider name
            model: model name

        Returns:
            dictionary containing input_per_1k and output_per_1k prices
        """
        try:
            return self.providers[provider]["models"][model]["pricing"]
        except KeyError:
            return {"input_per_1k": 0.0, "output_per_1k": 0.0}

    def estimate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        estimate usage cost

        Args:
            provider: provider name
            model: model name
            input_tokens: input token count
            output_tokens: output token count

        Returns:
            estimated cost (USD)
        """
        pricing = self.get_model_pricing(provider, model)

        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]

        return input_cost + output_cost

    def get_config_summary(self) -> str:
        """get configuration summary"""
        summary = "🤖 current LLM configuration:\n"
        summary += "=" * 50 + "\n"

        for agent_type, config in self.agents.items():
            provider_name = config["provider"]
            model_name = config["model"]
            model_config = self.providers[provider_name]["models"][model_name]
            pricing = model_config.get("pricing", {})

            summary += f"\n📋 {agent_type.replace('_', ' ').title()}:\n"
            summary += f"    provider: {provider_name.title()}\n"
            summary += f"    model: {model_config['name']}\n"
            summary += f"    temperature: {config['temperature']}\n"
            summary += f"    max tokens: {config['max_tokens']}\n"
            summary += f"    price: ${pricing.get('input_per_1k', 0):.6f}/${pricing.get('output_per_1k', 0):.6f} (input/output per 1K)\n"
            summary += f"    description: {config['description']}\n"

        return summary

    def validate_environment(self) -> Dict[str, bool]:
        """
        validate environment variables

        Returns:
            API keys availability for each agent type
        """
        results = {}

        for agent_type, config in self.agents.items():
            provider_name = config["provider"]
            api_key_env = self.providers[provider_name]["api_key_env"]
            api_key = os.getenv(api_key_env)
            results[agent_type] = api_key is not None

        return results
