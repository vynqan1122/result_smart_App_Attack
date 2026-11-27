# Contributing

Thank you for your interest in improving this research reproduction. Contributions are welcome for documentation, reproducibility, testing, evaluation methodology and defensive AI security.

## Before contributing

- Read [README.md](README.md), [LICENSING.md](LICENSING.md) and [SECURITY.md](SECURITY.md).
- Check open issues and propose substantial changes before implementation.
- Only submit material you have the right to contribute. Identify third-party code, weights, data and figures, including their source and applicable terms. Do not assume that a public repository is licensed for redistribution.
- Do not include secrets, private datasets, proprietary models, or personal data.

## Workflow

1. Fork the repository or work on a separate branch.
2. Make a focused change with a descriptive commit message.
3. Run `python -m unittest discover -s tests -v`.
4. If changing experimental code, document the environment, dataset split, model configuration, seeds and any relevant validation performed.
5. Open a pull request explaining the motivation, changes, evidence, limitations and licensing/provenance information.

## Reproducibility checklist

For a new or changed benchmark, record the repository commit, hardware, operating system, Python/library versions, dataset origin and split, model and preprocessing details, evaluation settings, number of samples and seeds, clean performance, metric definition, and the raw output location. Include uncertainty or variation when available. Distinguish new measurements from reproduced or historical values.

Do not silently replace old results or present incomplete runs as complete. Avoid committing large model files or restricted datasets; provide documented retrieval procedures and checksums where distribution is permitted.

## Review expectations

Prefer small, testable changes. Document what has and has not been verified. The current CI is intentionally lightweight and is not a substitute for an authorized end-to-end scientific evaluation. Constructive review and respectful discussion are expected.
