# Copyright 2025 AI21 Labs
_pp_tokenizer = None


def init_process_tokenizer(tokenizer_name_or_path):
    """Initializer for ProcessPool workers to create a tokenizer per process."""
    global _pp_tokenizer
    from verl.utils import hf_tokenizer

    try:
        _pp_tokenizer = hf_tokenizer(tokenizer_name_or_path)
    except Exception:
        pass


def compute_valid_mask_for_chunk(conversation_chunk, tools_chunk, force_thinking, max_prompt_length):
    global _pp_tokenizer
    """Compute validity mask for a chunk of conversations using the per-process tokenizer."""
    has_tools = any(len(t) > 0 for t in tools_chunk)
    if has_tools:
        # Unfortunately, apply_chat_template() doesn't support batching + per-example tools
        rendered = [
            _pp_tokenizer.apply_chat_template(
                single_conversation,
                add_generation_prompt=True,
                force_thinking=force_thinking,
                tools=single_tool,
            )
            for single_conversation, single_tool in zip(conversation_chunk, tools_chunk, strict=False)
        ]
    else:
        rendered = _pp_tokenizer.apply_chat_template(
            conversation_chunk,
            add_generation_prompt=True,
            force_thinking=force_thinking,
        )
    return [len(r) <= max_prompt_length for r in rendered]
