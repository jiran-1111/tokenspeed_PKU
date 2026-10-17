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

import torch

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.logits_processor import should_apply_lm_head_quant_method
from tokenspeed.runtime.model_loader import get_model
from tokenspeed.runtime.utils import (
    get_available_gpu_memory,
    get_colorful_logger,
    set_cuda_arch,
)
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


def _configure_npu_lm_head(model, *, requested_dtype: str | None, device: str) -> None:
    """Expand an explicitly selected dense BF16 head before cache planning."""
    if requested_dtype is None:
        return
    if requested_dtype != "float32":
        raise ValueError("NPU LM head override supports only float32")
    if device != "npu":
        raise ValueError("LM head precision override requires NPU")
    if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
        raise ValueError("LM head precision override requires dense Qwen3.5")
    if model.encoder_only or model.lm_head is None:
        raise ValueError("LM head precision override requires a text LM head")
    if model.config.tie_word_embeddings:
        raise ValueError("LM head precision override requires untied embeddings")
    head = model.lm_head
    if should_apply_lm_head_quant_method(head, head.quant_method):
        raise ValueError("LM head precision override requires unquantized weights")
    if head.weight.dtype != torch.bfloat16:
        raise ValueError("LM head precision override requires BF16 projected weights")
    # FP32 exactly represents every loaded BF16 value; retain Parameter metadata.
    head.weight.data = head.weight.data.float()


class WeightLoader:
    """Handles model weight loading from disk.

    This class is stateless and does not modify external state.
    It returns LoadedModel with all necessary information.
    """

    @staticmethod
    def load_model(
        model_config: ModelConfig,
        server_args: ServerArgs,
        device: str,
        gpu_id: int,
        memory_saver_adapter: TorchMemorySaverAdapter,
    ):
        """Load model from disk.

        Args:
            model_config: Model configuration
            server_args: Server arguments
            device: Device type ("cuda", "cpu")
            gpu_id: GPU ID
            memory_saver_adapter: Memory saver adapter

        Returns:
            LoadedModel with model and dtype
        """
        logger.info(
            "Load weight begin. avail mem="
            f"{get_available_gpu_memory(device, gpu_id):.2f} GB",
        )

        # Reduce thread conflicts during weight loading
        if device != "cpu":
            torch.set_num_threads(1)

        set_cuda_arch()

        # Create load config
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            ext_yaml=server_args.ext_yaml,
            weight_loader_prefetch_checkpoints=server_args.weight_loader_prefetch_checkpoints,
            weight_loader_prefetch_num_threads=server_args.weight_loader_prefetch_num_threads,
        )

        # Load model with memory saver context. Tag as "weights" with CPU backup
        # so release_memory_occupation offloads (and restores) them byte-exact.
        with memory_saver_adapter.region(tag="weights", enable_cpu_backup=True):
            model = get_model(
                model_config=model_config,
                load_config=load_config,
                device_config=DeviceConfig(device),
            )
            _configure_npu_lm_head(
                model,
                requested_dtype=model_config.model_override_args.get(
                    "npu_lm_head_dtype"
                ),
                device=device,
            )
            initialize_engram = getattr(model, "initialize_engram", None)
            if callable(initialize_engram):
                # Immutable tokenizer/hash tensors belong to the weight lifetime,
                # before cache budgeting or the first warmup/captured forward.
                tokenizer = get_tokenizer(
                    server_args.tokenizer,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    architectures=model_config.hf_config.architectures,
                    **model_config.tokenizer_kwargs,
                )
                initialize_engram(tokenizer)

        # Load KV cache scaling factors if using FP8
        if server_args.kv_cache_dtype == "fp8_e4m3":
            if server_args.quantization_param_path is not None:
                if callable(getattr(model, "load_kv_cache_scales", None)):
                    model.load_kv_cache_scales(server_args.quantization_param_path)
                    logger.info(
                        "Loaded KV cache scaling factors from "
                        f"{server_args.quantization_param_path!s}",
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        f"model {model.__class__} does not support loading scaling factors."
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors provided. "
                    "Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        dtype = model_config.dtype

        logger.info(
            f"Load weight end. type={type(model).__name__!s}, dtype={dtype!s}, avail "
            f"mem={get_available_gpu_memory(device, gpu_id):.2f} GB",
        )

        return model
