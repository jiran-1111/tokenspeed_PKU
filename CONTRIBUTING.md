# Contributing to the Ascend course fork

This fork keeps the original TokenSpeed architecture and licenses. Course
changes belong in a focused branch with reproducible checks and a clear account
of which source revision produced each measurement. Historical homework and
course delivery branches are immutable references; use a new branch for fixes.

1. Start from the intended base branch and inspect remote changes before editing.
   The validated course base is `course/final-20261018`; upstream `main` and
   independent experimental branches may have incompatible runtime changes.
2. Follow [AGENTS.md](AGENTS.md) and the relevant [design documents](docs/design/).
   Keep runtime changes separate from experiment drivers and reports.
3. Use a project-local environment and run the CPU checks in [README.md](README.md).
   Add a regression test for behavior changes. Hardware validation is required
   for affected device behavior; report skipped checks explicitly.
4. Before a commit, run `pre-commit run --all-files`, review the entire diff, and
   stage an explicit list of source, test and documentation paths. Keep models,
   caches, raw logs, host details and credentials out of Git.
5. Use a descriptive Conventional Commit and `git commit --signoff`. Open a PR
   against the intended base in this fork, describing the problem, resulting
   behavior, validation and any hardware limitations. Do not force-push shared
   branches or combine independently measured source snapshots without revalidation.

Do not change a numerical acceptance threshold, replace a failed input or remove
a failed run to make validation pass. Keep observed facts separate from inferred
causes and state the role of AI tools accurately.

## Formatting tools

Install `pre-commit==4.7.0` into the activated project environment. The hook
configuration requires an available Python 3.12 interpreter, even if the CPU
tests use Python 3.11. Keep the hook cache and temporary files in the checkout:

```bash
python -m pip install pre-commit==4.7.0
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
export VIRTUALENV_OVERRIDE_APP_DATA="$PWD/.cache/virtualenv"
export TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR"
pre-commit run --all-files
```

On PowerShell, set the same variables using `$env:NAME = Join-Path $PWD 'path'`.
Hook environments and cached tools are local development dependencies; they do
not belong in the runtime package or in Git.

## Upstream project ethos

TokenSpeed is intentionally built as a lean, engineering-focused infrastructure project.

In spirit, our development style is closer to projects like OpenAI Triton:
a lean and small core team, move fast, high technical ownership, and an extremely high quality bar.

As a result, we are intentionally selective about external contributions — not because we do not value community involvement, but because maintaining long-term simplicity, performance, and reliability matters deeply for infrastructure software.

TokenSpeed is built by contributors from different companies and open-source communities under the LightSeek Foundation ecosystem. Our mission is simple:

We believe the future of AI should be transparent, collaborative, and inclusive. LightSeek exists to accelerate the development of open, efficient, and trustworthy AI systems—so that progress in AI benefits everyone, not just a few.

## We Welcome External Contributions

We especially welcome:

- obvious bug fixes
- small and verifiable production-needed features
- performance optimizations that fit the existing codebase style and do not introduce unnecessary complexity
- documentation, tooling, and benchmarking improvements

For larger features or architectural changes, we generally recommend starting with an RFC or design discussion first before implementation.

## Engineering Principles

We value:

- reproducible performance
- correctness and stability
- production-oriented engineering
- maintainable implementations

We believe open-source infrastructure advances through open collaboration, healthy technical discussion, and shared engineering progress.

Thanks to everyone contributing code, ideas, reviews, benchmarks, and feedback.

— TokenSpeed Team
