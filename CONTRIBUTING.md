# Contributing to Explainable Hybrid Deep Learning IDS

Thank you for your interest in contributing! This guide explains how to get started.

## Project Context

This is a **Third-Year, Fifth-Semester Deep Learning Elective** project developed collaboratively by two students. Contributions from both project authors and external collaborators are welcome.

## How to Contribute

### Reporting Bugs

Open a GitHub Issue with:
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
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Linux/macOS
   pip install -r requirements.txt
   ```

3. **Place the dataset** as described in `dataset/README.md`.

4. **Write clear, documented code** consistent with the existing style.

5. **Run the test suite** before submitting:
   ```bash
   python -m pytest tests/ -v
   ```

6. **Submit a Pull Request** with a clear description of the changes.

## Collaboration Between Project Authors

For the two primary authors (Midhila M and Amrita Raj):

### Setting Up Your Git Identity

Each author should configure Git with their own identity before committing:

```bash
git config user.name "Your Name"
git config user.email "your-github-email@example.com"
```

This ensures commits are attributed correctly to your GitHub account.

### Recommended Workflow

1. Create a feature branch for your contribution:
   ```bash
   git checkout -b feature/your-contribution
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

3. Push your branch:
   ```bash
   git push origin feature/your-contribution
   ```

4. Open a Pull Request on GitHub to merge into `main`.

## Code Style

- Follow [PEP 8](https://pep8.org/).
- Use descriptive variable names.
- Include docstrings for public functions and classes.
- Avoid committing dataset CSV files, model weights over 50 MB, or secrets.

## Academic Integrity

This project was developed for academic learning purposes. Any contributions should:
- Not fabricate evaluation results or metrics.
- Not introduce offensive network capabilities.
- Properly cite any new papers or resources used.
- Respect the UNSW-NB15 dataset licence.

## Questions

Open a GitHub Discussion or Issue.
