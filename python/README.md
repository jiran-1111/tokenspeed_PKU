# TokenSpeed Python runtime

TokenSpeed provides an LLM inference engine with a C++ scheduler and a portable
kernel boundary. This fork adds Ascend course experiments and reproducible
Qwen3.5 configurations while preserving the upstream runtime structure.

Installation depends on the selected hardware and its compatible software stack.
See the [repository README](https://github.com/jiran-1111/tokenspeed_PKU/blob/fix/oss-project-20261018/README.md)
for project-local setup, supported experiment scope and validation instructions.
The top-level project is a multi-package source repository; install the runtime
from `python/` and follow the repository's device setup instructions.

The software is MIT licensed. Model weights and third-party libraries retain
their respective licenses.
