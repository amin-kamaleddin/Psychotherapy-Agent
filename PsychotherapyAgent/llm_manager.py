"""
LLM Manager for PsychotherapyAgent.

Handles initialization and management of all ChatOpenAI instances.
"""

import logging
import os
import httpx
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class LLMManager:
    """Manages all LLM instances for PsychotherapyAgent."""

    # OpenRouter base URL
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: SecretStr):
        """
        Initialize all LLM instances.

        Args:
            api_key: OpenAI API key (as SecretStr)
        """
        self.api_key = api_key
        
        # Load environment variables
        load_dotenv()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Therapist model selection.
        #   - Default ("gpt-5-mini" or unset) -> OpenAI, using the OpenAI key.
        #   - An OpenRouter-formatted id (contains "/", e.g. "qwen/qwen3.8-27b") ->
        #     routed through OpenRouter using OPENROUTER_API_KEY.
        therapist_model = os.getenv("PSYCH_THERAPIST_MODEL") or "gpt-5-mini"
        self.therapist_model = therapist_model

        use_openrouter = ("/" in therapist_model) or therapist_model.lower().startswith(("google/", "qwen/", "deepseek/", "meta-llama/"))

        if use_openrouter and not openrouter_api_key:
            logging.warning("PSYCH_THERAPIST_MODEL is an OpenRouter model but OPENROUTER_API_KEY missing. Using OpenAI gpt-5-mini.")
            use_openrouter = False
            therapist_model = "gpt-5-mini"

        def _chat_openai(model: str, api_key_val, base_url=None, temperature=None, max_tokens=None, extra_body=None):
            kwargs = {"model": model, "api_key": api_key_val}
            # Strict, bounded timeouts so a hung request RAISES instead of blocking a worker thread
            # forever. httpx.Timeout applies to connect/write/read/pool. A single read timeout is
            # reset per packet on streams, so we also set an overall deadline via CAPPED retries
            # and a connect+read bound. max_retries=0 prevents langchain from re-sending a hung
            # request indefinitely (a common cause of the multi-minute stalls seen in the runs).
            kwargs["timeout"] = httpx.Timeout(
                connect=30.0,
                write=30.0,
                read=120.0,
                pool=30.0,
            )
            kwargs["max_retries"] = 0
            if base_url:
                kwargs["base_url"] = base_url
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if extra_body:
                kwargs["extra_body"] = extra_body
            return ChatOpenAI(**kwargs)

        # For OpenRouter reasoning models, cap chain-of-thought effort so it doesn't eat the
        # entire output budget (the structured answer must still fit within max_tokens).
        reasoning_body = None
        if use_openrouter:
            reasoning_body = {"reasoning": {"effort": "low"}}

        # Domain-focused generation. The configured model (e.g. qwen) is often a REASONING model:
        # it spends a large chunk of output tokens on chain-of-thought BEFORE the actual answer.
        # From the logs, structured-feedback calls used ~3700-4000 reasoning tokens.
        # The cap must leave room for BOTH the reasoning and the (short) structured JSON answer,
        # otherwise the answer is truncated -> LengthFinishReasonError -> unstructured fallback.
        # 8000 leaves headroom for reasoning (~4000) + answer (~1-2k) + margin.
        FEEDBACK_MAX = 8000    # structured MI evaluation JSON (reasoning + answer)
        THERAPIST_MAX = 8000   # proposer (3 candidates / self-select) + judge (reasoning + answer)

        # Core LLMs (summarization, patient simulation) — use the therapist model so the whole
        # pipeline runs on the configured model. Default: gpt-5-mini.
        self.llm_mini_summary = _chat_openai(
            "gpt-5-mini" if not use_openrouter else therapist_model,
            self.api_key if not use_openrouter else SecretStr(openrouter_api_key),
            self.OPENROUTER_BASE_URL if use_openrouter else None,
            max_tokens=THERAPIST_MAX,
            extra_body=reasoning_body if use_openrouter else None
        )
        self.llm_mini_virtualPatient = _chat_openai(
            "gpt-5-mini" if not use_openrouter else therapist_model,
            self.api_key if not use_openrouter else SecretStr(openrouter_api_key),
            self.OPENROUTER_BASE_URL if use_openrouter else None,
            max_tokens=THERAPIST_MAX,
            extra_body=reasoning_body if use_openrouter else None
        )
        self.llm_feedback = _chat_openai(
            "gpt-5-mini" if not use_openrouter else therapist_model,
            self.api_key if not use_openrouter else SecretStr(openrouter_api_key),
            self.OPENROUTER_BASE_URL if use_openrouter else None,
            max_tokens=FEEDBACK_MAX,
            extra_body=reasoning_body if use_openrouter else None
        )
        
        # Evaluator should use OpenRouter Gemini when available (independent of therapist model)
        if openrouter_api_key:
            self.llm_mini_virtualEvaluator = _chat_openai(
                "google/gemini-2.5-flash", SecretStr(openrouter_api_key), self.OPENROUTER_BASE_URL, temperature=0.1,
                max_tokens=FEEDBACK_MAX,
                extra_body=reasoning_body
            )
            logging.info("Virtual evaluator initialized with OpenRouter Gemini 2.5 Flash.")
        else:
            logging.warning("OPENROUTER_API_KEY not found in environment. Using OpenAI for virtual evaluator.")
            self.llm_mini_virtualEvaluator = _chat_openai("gpt-5-mini", self.api_key, max_tokens=FEEDBACK_MAX)
        
        # Therapist agents (for multi-agent mode) - driven by the configured model.
        self.llm_therapist_proposer = _chat_openai(
            therapist_model,
            SecretStr(openrouter_api_key) if use_openrouter else self.api_key,
            self.OPENROUTER_BASE_URL if use_openrouter else None,
            max_tokens=THERAPIST_MAX,
            extra_body=reasoning_body if use_openrouter else None
        )
        self.llm_therapist_judge = _chat_openai(
            therapist_model,
            SecretStr(openrouter_api_key) if use_openrouter else self.api_key,
            self.OPENROUTER_BASE_URL if use_openrouter else None,
            max_tokens=THERAPIST_MAX,
            extra_body=reasoning_body if use_openrouter else None
        )
        
        logging.info(f"LLM Manager initialized with therapist model: {therapist_model} "
                     f"(OpenRouter={use_openrouter}).")

