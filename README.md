# Sekai Agent: A Self-Evolving Story Recommendation System

This project implements a multi-agent system designed to autonomously optimize and refine story recommendations.


## How to Run

**1. Setup Environment**
```bash
# Create a virtual environment
uv sync -p 3.11
source .venv/bin/activate
```

**2. Export Your GOOGLE API KEY**
```bash
export GOOGLE_API_KEY==<your_api_key>
```

**3. Run All in One Script**
This project uses a standalone Milvus instance managed via Docker.
```bash
# This will start Milvus and etcd services.
# Ensure you have Docker installed and running.
bash start.sh
```

## Agent Loop
![](/static/agent_loop.png)

## Technical Architecture

### Tools
- **Language**: Python 3.11
- **Package Management**: `uv` for fast, modern dependency resolution and virtual environment management.

### Frameworks
- **Workflow Orchestration (LangGraph)**: The entire multi-agent workflow is defined and executed as a state machine using LangGraph. LangChain is also used for token usage tracing.

- **Automated Prompt Engineering (DSPy)**: DSPy is the core framework for prompt optimization. It abstracts away the complexities of traditional prompt engineering by treating prompts as modules that can be programmatically refined. The `PromptOptimizerAgent` uses DSPy to automatically improve recommendation prompts based on quantitative evaluation feedback.

- **Vector Store (Milvus)**: A highly scalable, open-source vector database used for efficient similarity searches. The `RecommendationAgent` queries Milvus to retrieve a set of relevant candidate stories based on a user's preferences before the final selection by LLM.

## Agent Roles

The system is composed of three specialized agents. Note that you can easily change the model in the [agents llm config](https://github.com/kk-xuhj/sekai-agent/blob/7ef56551a18cd4d6fdcd7dc131745d2dd4ed5a97/config/config.yaml#L70).

1.  **`RecommendationAgent`**
    - **Purpose**: To provide 10 story recommendations for a user given a small set of their preference tags.
    - **Model**: Designed to be fast and efficient, using a model like `Gemini 2.0 Flash Lite`.
    - **Process**: It converts the user's tags into a semantic search query, retrieves candidate stories from Milvus, and then uses an LLM with the currently optimized prompt to select the final 10 stories.

2.  **`EvaluationAgent`**
    - **Purpose**: To score the quality of the `RecommendationAgent`'s output.
    - **Model**: Uses a powerful reasoning model like `Gemini 2.0 Flash Lite` to ensure high-quality evaluation.
    - **Process**:
        1.  It first generates a "ground-truth" list of ideal recommendations using the user's *full* profile.
        2.  It then compares the `RecommendationAgent`'s list (generated from *partial* tags) against this ground-truth list.
        3.  It computes a `precision@10` score and generates qualitative feedback for the optimization agent.

3.  **`PromptOptimizerAgent`**
    - **Purpose**: To improve the master recommendation prompt.
    - **Model**: Any powerful model (`Gemini 2.0 Flash` for cost considerations since I am using my own API KEY).
    - **Process**: It takes the evaluation score and feedback from the `EvaluationAgent` and uses DSPy's optimization capabilities to generate and compile a new, improved prompt for the `RecommendationAgent` to use in the next cycle.

## Evaluation & Scoring

- **Metric**: **Precision@10**. This metric was chosen because it directly measures the accuracy of the top 10 recommended stories. F1 and Recall are equal to this metric when the lenght of predition is the same as the groundtruth.

- **Ground Truth Generation**: During the evaluation phase, the full user profile is fed into the Recommendation Agent to obtain the ground truth. This ground truth is not only used for evaluation, but also serves as training data for DSPy to optimize the prompt.

- **Stopping Rule**: The autonomous loop terminates when one of the following conditions is met:
    1.  **Target Score Reached**: The `precision@10` score meets or exceeds a predefined target (e.g., `0.7`, you can also tweak it if you like).
    2.  **Time Budget Exceeded**: The total runtime exceeds the configured limit (e.g., 3 minutes).
    3.  **Max Iterations Reached**: The loop completes the maximum number of configured iterations.
    4.  **Early Stopping**: The evaluation score fails to improve for a set number of "patience" iterations.

## Sample Optimization Log

Please Refer to the [example_log](/out.log)


## Caching Strategy

- **Embedding Cache**: Story embeddings are pre-calculated using the `import_data.py` and stored permanently in Milvus. This avoids the expensive process of re-calculating embeddings for the entire story corpus on every run.
- **LLM Caching**: The DSPy framework has built-in caching. By enabling it (`dspy.settings.configure(cache_turn_on=True)`), all LLM calls are cached to disk. This dramatically speeds up development and repeated runs by avoiding redundant API calls for the same inputs, saving both time and money.
- **In-Memory Search Results**: Within a single optimization cycle, the candidate stories retrieved from Milvus are cached in memory. This allows different agents within the same loop to access the identical candidate set without performing redundant vector searches.

## Scaling to Production

- **Decoupled Services**: Each agent would be deployed as a separate microservice (e.g., using FastAPI on a serverless platform). Communication would be handled via a message queue to manage the asynchronous workflow.
- **Managed Vector DB**: The standalone Milvus container would be replaced with a production-grade, managed vector database like Zilliz Cloud or Milvus on Kubernetes. This provides high availability, automatic scaling, data backups, and multi-tenancy.
- **Horizontal Scaling**: The `RecommendationAgent` service would be scaled horizontally to handle high volumes of real-time user requests. The `Optimization` and `Evaluation` tasks would run as less frequent, asynchronous background jobs on a separate, more powerful compute instance.

## Further Discussion & Future Work

1.  **Model Exploration**: Due to time constraints, I focused primarily on testing and debugging with Google's Gemini models. The framework is model-agnostic, and performance could likely be enhanced by experimenting with other state-of-the-art models from providers like OpenAI or Anthropic (To Be Set Up).

2.  **Framework Choice (DSPy)**: I selected the DSPy framework for its programmatic approach to prompt optimization. Its comprehensive documentation and active community enabled rapid development for this take-home challenge. Its seamless integration with LangChain was also a major plus for tracking token consumption. Most importantly, DSPy automates the traditionally manual and time-consuming process of prompt tuning. While other excellent frameworks exist, I am keen to explore alternatives in future work.

3.  **Evaluation and Stability**: I experimented with various metrics but found their performance to be similar within this specific framework. It is crucial to acknowledge that the inherent non-determinism in LLMs means the optimization loop does not guarantee improvement on every single run. However, after significant tuning—including dataset construction, input signature design, and metric combinations. The system now could achieve better performance than the initial baseline prompt in the most of runs.

4.  **The Challenge of Ground-Truth Data**: I found that the most formidable challenge in this project was defining the evaluation dataset. The process where an agent must autonomously generate its own ground truth for training and evaluation is non-trivial. Designing a robust method for defining, generating, and expanding this dataset in a fully automated way is a significant hurdle and a fascinating area for future exploration.

