# Contributing to GutenOCR

Thanks for your interest in contributing to GutenOCR! Whether you're fixing a bug, proposing a feature, or improving docs, we appreciate the help.

Check out our [roadmap and open issues](https://github.com/Roots-Automation/GutenOCR/issues) for contribution ideas.

## Dev Setup

```bash
# Clone the repo
git clone https://github.com/Roots-Automation/GutenOCR.git
cd GutenOCR
```

We use [uv](https://github.com/astral-sh/uv) for Python package management. Each component has independent dependencies, so install only what you need:

```bash
# Example: training dependencies
cd experiments/qwen-multigpu-sft && uv sync

# Example: evaluation dependencies
cd experiments/vllm-ocr-eval && uv sync
```

## Reporting Bugs

Please [open a bug report](https://github.com/Roots-Automation/GutenOCR/issues/new?template=bug_report.yml) using our issue template. Include:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Which component is affected (data pipelines, training, evaluation, etc.)

## Proposing Features

Have an idea? [Open a feature request](https://github.com/Roots-Automation/GutenOCR/issues/new?template=feature_request.yml). Describe the motivation and use case so we can understand the context.

## Pull Request Guidelines

1. **Fork** the repository and create a branch from `main`.
2. **Open an issue first** (or reference an existing one). Every PR should link to an issue.
3. **Keep changes focused.** One logical change per PR.
4. **Describe your changes** in the PR description.
5. **Test locally** before submitting.

### PR Checklist

- [ ] References an issue (e.g., `Closes #123`)
- [ ] Tested locally
- [ ] Updated docs if applicable

## Code Style

- Follow existing conventions in the file you're editing.
- Use `uv run python` to run scripts.
- Keep commits focused and write clear commit messages.

## License

By contributing to GutenOCR, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Questions?

If you're unsure about anything, feel free to open an issue and ask. We're happy to help!
