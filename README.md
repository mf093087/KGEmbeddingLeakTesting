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
2. Install the required Python packages:
   pip install -r requirements.txt

   Dataset Description
The dataset contains enhanced data from a leak testing system:

Original Dataset: Derived from the leak testing system's Neo4j database.
Enhanced Dataset: Preprocessed and split into train_enhanced.txt, valid_enhanced.txt, and test_enhanced.txt for training, validation, and testing, respectively.

Running the Main Script
The main script is Main.py, which:

Loads the dataset from the dataset/ folder.
Trains several KGE and Graph-Based models.
Evaluates the models on the test dataset.
Saves the results in results_1000.json and results_graph_1000.json.
Run the script using the following command:


python Main.py
Results
results_1000.json: Contains evaluation metrics (e.g., Hits@K, MRR) for Knowledge Graph Embedding models.
results_graph_1000.json: Contains evaluation metrics for Graph-Based models.
Models Trained
Knowledge Graph Embedding Models:
TransE
RotatE
ComplEx
DistMult
TransH
TransR
TransD
Graph-Based Models:
GraphSAGE
R-GCN
A2N
CompGCN
SE-GNN
Contributing
Feel free to open an issue or submit a pull request if you would like to contribute to the project.

License
This project is licensed under the MIT License.

csharp
Copy code

You can copy and paste this text into your `README.md` file on GitHub.
