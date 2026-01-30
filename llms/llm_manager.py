import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '../.env')):
    load_dotenv()


class LlmManager:
    """
    Manages LLM instances with production-grade model selection and configuration.
    """
    def __init__(self):
        """
        Initialize the LLM manager with model configurations.

        All models default to temperature=0 for deterministic behavior.
        Override temperature in get_llm() for creative tasks.
        """
        self.llm_configs = {
            # Ultra-fast local model: Best for simple tasks, high throughput, zero cost
            'gemma3:270m': {
                "class": ChatOllama,
                "params": {"temperature": 0, "model": "gemma3:270m"}
            },

            # Advanced local model: Requires 32GB+ RAM, best local quality
            # Only use if you have sufficient RAM available
            'gpt-oss:20b': {
                "class": ChatOllama,
                "params": {"temperature": 0, "model": "gpt-oss:20b"}
            },

            # Balanced local model: Good quality/speed ratio for medium tasks
            'gemma3:4b': {
                "class": ChatOllama,
                "params": {"temperature": 0, "model": "gemma3:4b"}
            },

            # Production cloud model: Best quality, function calling, large context
            # Cost: ~$0.0015 per request (1K tokens avg)
            'gpt-4.1-mini': {
                "class": ChatOpenAI,
                "params": {"temperature": 0, "model": "gpt-4.1-mini"}
            }
        }

    def get_llm(self, llm_name: str, temperature: int, callbacks: Optional[List[BaseCallbackHandler]] = None, bind_stop: bool = False) -> BaseChatModel:
        """
        Factory method to instantiate configured LLM instances.

        This method provides a centralized way to create LLM instances with
        consistent configuration, monitoring, and safety controls.

        ═══════════════════════════════════════════════════════════════════════════
        PARAMETER GUIDE
        ═══════════════════════════════════════════════════════════════════════════

        Args:
            llm_name (str): Model identifier from self.llm_configs
                Available options:
                • 'gemma3:270m'  - Ultra-fast local (latency-critical tasks)
                • 'gemma3:4b'    - Balanced local (cost-optimized quality)
                • 'gpt-oss:20b'  - Advanced local (requires 32GB RAM)
                • 'gpt-4.1-mini' - Production cloud (best quality)

                See class docstring for detailed comparison and selection guide.

            temperature (int or float): Controls output randomness (0.0 to 2.0)
                • 0.0   - Deterministic (same input → same output)
                          Use for: Classification, extraction, testing

                • 0.3-0.5 - Slightly creative (balanced variation)
                            Use for: Q&A, summarization, chat

                • 0.7-1.0 - Creative (diverse outputs)
                            Use for: Content generation, brainstorming

                • 1.5-2.0 - Highly creative (unpredictable)
                            Use for: Experimental/artistic tasks

                RECOMMENDATION: Start with 0.0 for reliability, increase only
                if you need variation in outputs.

            callbacks (Optional[List[BaseCallbackHandler]]):
                Callback handlers for monitoring and logging.

                Common use cases:
                • Logging: Track token usage and costs
                • Monitoring: Measure latency and errors
                • Debugging: Inspect prompts and responses
                • Streaming: Stream responses to UI in real-time

                Example:
                    from support.callback_handler import CustomCallbackHandler
                    callbacks = [CustomCallbackHandler()]
                    llm = manager.get_llm("gpt-4.1-mini", 0, callbacks=callbacks)

                Set to None to disable callbacks (faster, less overhead).

            bind_stop (bool): Whether to bind stop sequences to the model.

                Stop sequences tell the model to stop generating when it
                encounters specific strings. This is a safety mechanism.

                Currently only implemented for OpenAI models with sequences:
                • "\nObservation:" - Stops ReAct agent loops
                • "Observation"   - Alternative stop trigger

                WHY THIS MATTERS:
                In ReAct agent patterns, the model should stop after generating
                an action and let the tool execute. Without stop sequences, the
                model might hallucinate tool outputs instead of waiting for real
                tool execution.

                WHEN TO USE:
                • Set to True for ReAct agents using gpt-4.1-mini
                • Set to False for general Q&A, chat, summarization
                • Currently ignored for Ollama models (different mechanism)

                Example:
                    # For ReAct agent
                    llm = manager.get_llm("gpt-4.1-mini", 0, bind_stop=True)

                    # For general use
                    llm = manager.get_llm("gpt-4.1-mini", 0, bind_stop=False)

        Returns:
            BaseChatModel: Configured LLM instance ready for inference.
                Can be used with LangChain chains, agents, or direct invocation.

        Raises:
            ValueError: If llm_name is not in self.llm_configs.

        ═══════════════════════════════════════════════════════════════════════════
        USAGE EXAMPLES
        ═══════════════════════════════════════════════════════════════════════════

        Example 1: Simple deterministic Q&A
        ────────────────────────────────────
            manager = LlmManager()
            llm = manager.get_llm("gpt-4.1-mini", temperature=0)
            response = llm.invoke("What is 2+2?")
            # Always returns "4" (deterministic)

        Example 2: Creative content generation
        ───────────────────────────────────────
            llm = manager.get_llm("gemma3:4b", temperature=0.8)
            response = llm.invoke("Write a product tagline for eco-friendly shoes")
            # Returns varied creative outputs each time

        Example 3: Cost-optimized classification
        ─────────────────────────────────────────
            llm = manager.get_llm("gemma3:270m", temperature=0)
            response = llm.invoke("Classify: 'Great product!' → [positive/negative]")
            # Fast, free, deterministic

        Example 4: ReAct agent with monitoring
        ───────────────────────────────────────
            from support.callback_handler import CustomCallbackHandler

            llm = manager.get_llm(
                "gpt-4.1-mini",
                temperature=0,
                callbacks=[CustomCallbackHandler()],
                bind_stop=True  # Prevents hallucinated observations
            )
            # Use with LangChain ReAct agent

        Example 5: High-quality analysis (when cost isn't a concern)
        ─────────────────────────────────────────────────────────────
            llm = manager.get_llm("gpt-4.1-mini", temperature=0)
            response = llm.invoke(
                "Analyze this financial report and identify the top 3 risks..."
            )
            # Best quality, worth the ~$0.002 cost

        Example 6: High-throughput batch processing
        ────────────────────────────────────────────
            llm = manager.get_llm("gemma3:270m", temperature=0)
            for document in documents:  # Process 1000s of documents
                summary = llm.invoke(f"Summarize: {document}")
            # Fast (50-100ms each), zero cost

        ═══════════════════════════════════════════════════════════════════════════
        IMPLEMENTATION DETAILS
        ═══════════════════════════════════════════════════════════════════════════

        The method:
        1. Validates llm_name exists in configurations
        2. Deep copies base params to avoid mutation
        3. Overrides temperature with provided value
        4. Adds callbacks if provided
        5. Binds stop sequences if requested (OpenAI only)
        6. Returns instantiated model

        Why deep copy?
        Without deepcopy, modifying params would affect the base config for
        subsequent calls, causing unexpected behavior.

        Why bind_stop only for OpenAI?
        Ollama models use a different mechanism for stop sequences that's
        configured at the model level, not via bind().
        """
        # Validate model exists
        if llm_name not in self.llm_configs:
            raise ValueError(f"LLM '{llm_name}' is not supported. Available: {list(self.llm_configs.keys())}")

        config = self.llm_configs[llm_name]
        llm_class = config["class"]

        # Use deepcopy to create a safe copy of parameters
        params = deepcopy(config["params"])
        params["temperature"] = temperature

        # Add callbacks for monitoring/logging
        if callbacks:
            params["callbacks"] = callbacks

        # Instantiate + Bind stop sequences only if requested AND for OpenAI models
        if bind_stop and llm_name == 'gpt-4.1-mini':
            llm = llm_class(**params).bind(stop=["\nObservation:", "Observation"])

        # for other models, just instantiate
        else:
            llm = llm_class(**params)

        return llm

    #region Model Information and Recommendation
    @staticmethod
    def get_model_info(llm_name: str) -> dict:
        """
        Get detailed information about a specific model.

        Useful for making informed decisions about model selection, cost estimation,
        and performance expectations.

        Args:
            llm_name: Model identifier

        Returns:
            dict: Model specifications and characteristics

        Example:
            manager = LlmManager()
            info = manager.get_model_info("gpt-4.1-mini")
            print(f"Cost per 1K tokens: ${info['cost_per_1k_tokens']}")
            print(f"Estimated latency: {info['avg_latency_ms']}ms")
        """
        model_specs = {
            'gemma3:270m': {
                'provider': 'ollama',
                'type': 'local',
                'parameters': '270M',
                'context_window': 8192,
                'avg_latency_ms': 85,
                'tokens_per_second': 200,
                'cost_per_1k_tokens': 0.0,
                'ram_required_gb': 4,
                'supports_function_calling': False,
                'quality_score': 6.5,
                'recommended_for': ['classification', 'extraction', 'high-throughput'],
                'max_concurrent_requests': 10
            },
            'gemma3:4b': {
                'provider': 'ollama',
                'type': 'local',
                'parameters': '4B',
                'context_window': 8192,
                'avg_latency_ms': 420,
                'tokens_per_second': 100,
                'cost_per_1k_tokens': 0.0,
                'ram_required_gb': 16,
                'supports_function_calling': False,
                'quality_score': 7.8,
                'recommended_for': ['summarization', 'qa', 'content-generation'],
                'max_concurrent_requests': 5
            },
            'gpt-oss:20b': {
                'provider': 'ollama',
                'type': 'local',
                'parameters': '20B',
                'context_window': 8192,
                'avg_latency_ms': 2100,
                'tokens_per_second': 30,
                'cost_per_1k_tokens': 0.0,
                'ram_required_gb': 32,
                'supports_function_calling': False,
                'quality_score': 8.5,
                'recommended_for': ['analysis', 'reasoning', 'research'],
                'max_concurrent_requests': 1,
                'warning': 'Requires 32GB+ RAM'
            },
            'gpt-4.1-mini': {
                'provider': 'openai',
                'type': 'cloud',
                'context_window': 128000,
                'avg_latency_ms': 650,
                'tokens_per_second': 150,
                'cost_per_1k_tokens': 0.00038,  # Average of input ($0.15) and output ($0.60)
                'cost_per_request_estimate': 0.0015,  # Assuming 1K tokens average
                'ram_required_gb': 0,  # Cloud-based
                'supports_function_calling': True,
                'supports_structured_output': True,
                'quality_score': 9.2,
                'recommended_for': ['production', 'function-calling', 'complex-reasoning'],
                'max_concurrent_requests': 50,
                'uptime_sla': '99.9%'
            }
        }

        if llm_name not in model_specs:
            raise ValueError(f"Unknown model: {llm_name}")

        return model_specs[llm_name]

    @staticmethod
    def recommend_model(
            task_type: str = None,
        max_latency_ms: int = None,
        max_cost_per_request: float = None,
        requires_function_calling: bool = False,
        requires_privacy: bool = False
    ) -> str:
        """
        Get model recommendation based on requirements.

        This is a simplified decision helper. For production use, consider
        implementing the full ModelSelector from models/model_selector.py
        with comprehensive trade-off analysis.

        Args:
            task_type: Type of task ('classification', 'qa', 'summarization',
                      'reasoning', 'generation')
            max_latency_ms: Maximum acceptable latency in milliseconds
            max_cost_per_request: Maximum cost in USD per request
            requires_function_calling: Whether function calling is required
            requires_privacy: Whether data must stay local (no cloud APIs)

        Returns:
            str: Recommended model name

        Example:
            manager = LlmManager()

            # Fast classification task
            model = manager.recommend_model(
                task_type='classification',
                max_latency_ms=100
            )
            # Returns: 'gemma3:270m'

            # Production quality with function calling
            model = manager.recommend_model(
                task_type='reasoning',
                requires_function_calling=True
            )
            # Returns: 'gpt-4.1-mini'

            # Privacy-sensitive summarization
            model = manager.recommend_model(
                task_type='summarization',
                requires_privacy=True
            )
            # Returns: 'gemma3:4b'
        """
        # Hard constraints
        if requires_function_calling:
            return 'gpt-4.1-mini'  # Only model with function calling

        if requires_privacy and max_cost_per_request == 0:
            # Must be local and free
            if max_latency_ms and max_latency_ms < 200:
                return 'gemma3:270m'
            return 'gemma3:4b'

        # Latency-based
        if max_latency_ms:
            if max_latency_ms < 200:
                return 'gemma3:270m'
            elif max_latency_ms < 500:
                return 'gemma3:4b' if max_cost_per_request == 0 else 'gpt-4.1-mini'

        # Cost-based
        if max_cost_per_request is not None:
            if max_cost_per_request == 0:
                # Free local models only
                if task_type in ['classification', 'extraction']:
                    return 'gemma3:270m'
                return 'gemma3:4b'
            elif max_cost_per_request < 0.001:
                return 'gemma3:4b'  # Cloud would exceed budget

        # Task-based recommendations
        task_recommendations = {
            'classification': 'gemma3:270m',
            'extraction': 'gemma3:270m',
            'qa': 'gemma3:4b',
            'summarization': 'gemma3:4b',
            'generation': 'gemma3:4b',
            'reasoning': 'gpt-4.1-mini',
            'analysis': 'gpt-4.1-mini',
            'function-calling': 'gpt-4.1-mini'
        }

        if task_type in task_recommendations:
            return task_recommendations[task_type]

        # Default: balanced option
        return 'gemma3:4b'

    def list_models(self) -> list:
        """
        List all available models.

        Returns:
            list: Model names
        """
        return list(self.llm_configs.keys())

    def get_cost_estimate(self, llm_name: str, requests_per_month: int, avg_tokens_per_request: int = 1000) -> dict:
        """
        Estimate monthly costs for a given usage pattern.

        Args:
            llm_name: Model to estimate costs for
            requests_per_month: Expected number of requests per month
            avg_tokens_per_request: Average tokens per request (default 1000)

        Returns:
            dict: Cost breakdown

        Example:
            manager = LlmManager()
            estimate = manager.get_cost_estimate('gpt-4.1-mini', 10000, 1000)
            print(f"Monthly cost: ${estimate['monthly_cost_usd']:.2f}")
        """
        info = self.get_model_info(llm_name)

        cost_per_request = (avg_tokens_per_request / 1000) * info['cost_per_1k_tokens']
        monthly_cost = cost_per_request * requests_per_month

        return {
            'model': llm_name,
            'requests_per_month': requests_per_month,
            'avg_tokens_per_request': avg_tokens_per_request,
            'cost_per_request_usd': cost_per_request,
            'monthly_cost_usd': monthly_cost,
            'cost_breakdown': {
                'api_costs': monthly_cost if info['type'] == 'cloud' else 0,
                'infrastructure_costs': 0 if info['type'] == 'cloud' else 'See hosting costs',
                'note': 'Local models have zero API cost but require compute infrastructure'
            }
        }
    #endregion
