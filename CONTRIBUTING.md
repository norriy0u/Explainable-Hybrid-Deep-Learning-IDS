# Contributing to Explainable Hybrid Deep Learning IDS

Thank you for your interest in contributing to this research project! This guide will help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please open a GitHub Issue with:
- A clear, descriptive title.
- Steps to reproduce the problem.
- Your Python version and OS.
- The error message or unexpected output.

### Suggesting Enhancements

Feature suggestions are welcome via GitHub Issues. Label them as `enhancement`.

Ideas for improvement:
- Real-time packet capture integration (e.g., via Scapy or PyShark).
- Additional benchmark datasets (CIC-IDS-2018, NSL-KDD).
- Transformer-based attention mechanisms.
- LIME or Integrated Gradients alongside SHAP.
- Containerisation via Docker.
- REST API inference endpoint (FastAPI).

### Pull Requests

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install dependencies** in a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Place the dataset** as described in `dataset/README.md`.

4. **Write clear, documented code** consistent with the existing style.

5. **Run the test suite** before submitting:
   ```powershell
   python -m pytest tests/ -v
   ```

6. **Submit a Pull Request** with a clear description of the changes and why they improve the project.

## Code Style

- Follow [PEP 8](https://pep8.org/).
- Use descriptive variable names.
- Include docstrings for all public functions and classes.
- Avoid committing dataset CSV files, model weights, or secrets.

## Academic Integrity

This project is a final-year engineering research submission. Any contributions should:
- Not fabricate evaluation results or metrics.
- Not introduce offensive network capabilities.
- Properly cite any new papers or resources used.
- Respect the UNSW-NB15 dataset license.

## Questions

Open a GitHub Discussion or Issue. PRs without an associated issue may be closed without review.
