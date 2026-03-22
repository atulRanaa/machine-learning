# AI & Machine Learning Engineering

A comprehensive open-source study resource and foundational guide for Artificial Intelligence, Machine Learning, and Deep Learning internals. 

This repository powers the MkDocs documentation site, providing:
- Deep mathematical rigor and derivations
- Python, Scikit-learn, and PyTorch coding labs
- Architectural overviews and structural engineering patterns for Scalable AI
- Curated Hall of Fame for the top ML/AI research papers

## Local Development

To run the documentation site locally, ensure you have Python installed, then follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/atulRanaa/machine-learning.git
   cd machine-learning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install mkdocs mkdocs-material pymdown-extensions
   ```

4. Serve the documentation site:
   ```bash
   mkdocs serve
   ```
   Open `http://localhost:8000` in your browser.

## Deployment

The site is automatically built and deployed to GitHub Pages using GitHub Actions whenever changes are pushed to the `main` branch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
