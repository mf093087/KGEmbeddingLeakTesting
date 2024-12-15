# Knowledge Graph Embedding and Graph-Based Models for Leak Testing System

This repository implements several Knowledge Graph Embedding (KGE) and Graph-Based models to analyze data from a leak testing system. The models are trained and evaluated on a dataset derived from the original Neo4j database of the leak testing system.

## Repository Structure

- **`Main.py`**: The main script to train and evaluate the models.
- **`dataset/`**: Contains the training, validation, and test datasets.
  - `train_enhanced.txt`: Training dataset.
  - `valid_enhanced.txt`: Validation dataset.
  - `test_enhanced.txt`: Testing dataset.
- **`results_1000.json`**: Results for Knowledge Graph Embedding models.
- **`results_graph_1000.json`**: Results for Graph-Based models.

## Prerequisites

Before running the code, ensure you have Python installed (>=3.8). The required dependencies are listed in the `requirements.txt` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
