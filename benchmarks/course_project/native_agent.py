# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any

from benchmarks.course_project.core import chat_template_token_ids

ASSISTANT_HEADER = "<|im_start|>assistant\n"


def initial_qwen_context(
    tokenizer: Any, *, system_prompt: str, user_text: str
) -> tuple[list[int], str]:
    """Capture the initial native IDs and this tokenizer's real generation prefix."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    initial_ids = chat_template_token_ids(tokenizer, messages)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    without_generation = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not isinstance(full_text, str) or not isinstance(without_generation, str):
        raise ValueError("Qwen validation requires a text chat template")
    if not full_text.startswith(without_generation):
        raise ValueError("Qwen generation template rewrites the initial messages")
    prefix = full_text[len(without_generation) :]
    if not prefix.startswith(ASSISTANT_HEADER):
        raise ValueError("unsupported Qwen generation prefix")
    if list(tokenizer.encode(full_text, add_special_tokens=False)) != initial_ids:
        raise ValueError("Qwen template text does not reproduce native input IDs")
    return initial_ids, prefix


def append_qwen_user_suffix(
    tokenizer: Any,
    *,
    generation_prefix: str,
    previous_output_ids: list[int],
    user_text: str,
) -> list[int]:
    """Append one turn without decoding, stripping, or re-encoding old output IDs."""
    if not previous_output_ids:
        raise ValueError("previous output is required to append a user turn")
    if not generation_prefix.startswith(ASSISTANT_HEADER):
        raise ValueError("unsupported Qwen generation prefix")
    boundary = (
        "\n" if previous_output_ids[-1] == tokenizer.eos_token_id else "<|im_end|>\n"
    )
    text = (
        boundary
        + "<|im_start|>user\n"
        + user_text.strip()
        + "<|im_end|>\n"
        + generation_prefix
    )
    return list(tokenizer.encode(text, add_special_tokens=False))
