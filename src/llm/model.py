from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

from src.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> BaseLLM | BaseChatModel:
    
    cfg = settings or get_settings()

    if cfg.use_hf_inference_api:
        endpoint = HuggingFaceEndpoint(
            repo_id=cfg.llm_model,
            huggingfacehub_api_token=cfg.hf_token,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        return ChatHuggingFace(llm=endpoint)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, token=cfg.hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model,
            token=cfg.hf_token,
            device_map="auto",
            torch_dtype="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load local LLM. Set USE_HF_INFERENCE_API=true and "
            "provide HF_TOKEN, or ensure GPU resources are available."
        ) from exc
