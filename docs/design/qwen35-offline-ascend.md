# Qwen3.5 offline Ascend correctness harness

The course experiment in `tokenspeed.experiments.qwen35` is a single-request,
text-only correctness and performance harness. It executes every language-model
layer, loads all 320 text tensors strictly, and uses the tied embedding for the
language-model head. The vision tower and speculative MTP are outside text-only
autoregressive inference and are explicitly excluded.

This is **not a registered production serving backend**. It does not change the
one-path scheduler invariant, the LCM cache layout, or production Qwen3.5 model
selection. It provides no paged batching, prefix reuse, retraction, speculative
decoding, multimodal input, or ACL Graph claim. Its measurements must not be
presented as `tokenspeed serve` throughput.

Runtime experiment code imports operators through `tokenspeed-kernel`.
`ops.experimental_qwen35` contains portable ATen compositions whose tensors
remain on the input device; torch_npu dispatches the NPU operations. It does not
silently move unsupported arithmetic to the CPU. Algorithm and precision choices
are explicit in the CLI.

`RequestState` is caller-owned and is never keyed by scheduler slot. It holds
the absolute text position, full-attention KV tensors, the last three causal
convolution inputs, and FP32 GDN state. Fresh requests construct fresh state;
functional GDN and convolution updates preserve earlier tensor snapshots.
Production integration must instead use the existing LCM state groups and
checkpoint/retraction contracts described in `cache-concepts.md`; this request
object must not be transplanted into the scheduler as a private slot cache.

Both multi-token input and decode call the same model forward and update the
same state representation. Chunk GDN uses stable recursive block substitution
for its unit lower-triangular system. Explicit high powers of the lower triangle
were rejected: nearly parallel keys and beta close to one caused catastrophic
FP32 cancellation despite passing independent random-input tests. Regression
tests include this adversarial distribution and lengths across chunk boundaries.

Validation uses unmodified Transformers 5.17.0 on CPU in FP32 with the exact
checkpoint values, plus independent SDPA/Conv1D references and complete recurrent
GDN comparisons. Performance runs use one visible chip and synchronized wall
time. Optional stage timing is intrusive and includes host dispatch and device
execution; it is not a hardware kernel occupancy or bandwidth profiler.

The kernel package initializer only loads optional profiling when explicitly
requested by its existing environment switches. Pure ATen kernels can therefore
be imported without installing unrelated GPU compiler packages. Tests cover
both disabled and enabled bootstrap behavior.
