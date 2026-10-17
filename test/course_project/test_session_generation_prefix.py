"""Session reconstruction must retain the actual model generation prefix."""

from types import SimpleNamespace

import pytest
from test_session_kv import _load_method


@pytest.mark.parametrize("prefix", ["<think>\n", "<think>\n\n</think>\n\n"])
def test_next_suffix_preserves_model_generation_prefix(prefix):
    base = [11, 12]
    header = "<|im_start|>assistant\n"
    generation = [ord(c) for c in header + prefix]
    output = [ord(c) for c in "42"] + [0]
    history = base + generation + output
    tokenizer = SimpleNamespace(
        eos_token_id=0, decode=lambda ids, **kwargs: "".join(map(chr, ids))
    )

    def token_ids(tokenizer, messages, *, add_generation_prompt):
        if len(messages) == 2:
            return base + generation if add_generation_prompt else base
        return base + [ord(c) for c in header + messages[2]["content"]] + [0, 3, 4]

    next_suffix = _load_method(
        "scripts/run_session_kv.py",
        None,
        "_next_suffix",
        SYSTEM_PROMPT="system",
        _token_ids=token_ids,
        _initial_suffix=lambda tokenizer, text: base + generation,
    )
    assert next_suffix(
        tokenizer,
        "next",
        prior_user_text="previous",
        previous_output_ids=output,
        history_ids=history,
    ) == [3, 4]


@pytest.mark.parametrize("eos", [False, True])
@pytest.mark.parametrize(
    ("prefix", "output_text"),
    [
        ("<think>\n\n</think>\n\n", "42"),
        ("<think>\n", "Thinking Process:\n   "),
        ("<think>\n", "reasoning\n</think>\n42"),
    ],
)
def test_qwen_history_remains_append_only_when_template_rewrites_assistant_content(
    eos, prefix, output_text
):

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, *, add_special_tokens=False):
            return [
                0 if char == "\0" else ord(char)
                for char in text.replace("<|im_end|>", "\0")
            ]

        def decode(self, ids, *, skip_special_tokens=False):
            return "".join("<|im_end|>" if value == 0 else chr(value) for value in ids)

    tokenizer = Tokenizer()

    def token_ids(tokenizer, messages, *, add_generation_prompt):
        text = ""
        for message in messages:
            content = message["content"].strip()
            if message["role"] == "assistant" and "</think>" in content:
                content = content.split("</think>")[-1].lstrip("\n")
            text += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + prefix
        return tokenizer.encode(text)

    def initial(tokenizer, text):
        return token_ids(
            tokenizer,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": text},
            ],
            add_generation_prompt=True,
        )

    output = tokenizer.encode(output_text) + ([0] if eos else [])
    history = initial(tokenizer, "previous") + output
    original_history = history.copy()
    next_suffix = _load_method(
        "scripts/run_session_kv.py",
        None,
        "_next_suffix",
        SYSTEM_PROMPT="system",
        _token_ids=token_ids,
        _initial_suffix=initial,
    )
    suffix = next_suffix(
        tokenizer,
        "next",
        prior_user_text="previous",
        previous_output_ids=output,
        history_ids=history,
    )
    boundary = "\n" if eos else "<|im_end|>\n"
    expected = (
        boundary + "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n" + prefix
    )
    assert suffix == tokenizer.encode(expected)
    assert history == original_history
    with pytest.raises(ValueError, match="does not preserve the cached Qwen"):
        next_suffix(
            tokenizer,
            "next",
            prior_user_text="previous",
            previous_output_ids=output,
            history_ids=history + [999],
        )
