Knowledge Graph Embedding and Graph-Based Models for Leak Testing System

This repository implements several Knowledge Graph Embedding (KGE) and Graph-Based models to analyze data from a leak testing system. The models are trained and evaluated on a dataset derived from the original Neo4j database of the leak testing system.

--------------------------------------------------------------------------------

Repository Structure

- Main.py: The main script to train and evaluate the models.
- dataset/: Contains the training, validation, and test datasets.
  - train_enhanced.txt: Training dataset.
  - valid_enhanced.txt: Validation dataset.
  - test_enhanced.txt: Testing dataset.
- results_1000.json: Results for Knowledge Graph Embedding models.
- results_graph_1000.json: Results for Graph-Based models.

--------------------------------------------------------------------------------

Prerequisites

- Python >= 3.8
- Required packages listed in requirements.txt

--------------------------------------------------------------------------------

Installation

1. Clone the repository:
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. Install the required Python packages:
   pip install -r requirements.txt

--------------------------------------------------------------------------------

Dataset Description

- Original Dataset: Derived from the leak testing system and the corresponding Neo4j knowledge graph database.
- Enhanced Dataset: Preprocessed and split into:
  - train_enhanced.txt (training)
  - valid_enhanced.txt (validation)
  - test_enhanced.txt (testing)

--------------------------------------------------------------------------------

Running the Main Script

Main.py performs the following steps:
1. Loads the dataset from the dataset/ folder.
2. Trains several KGE and Graph-Based models.
3. Evaluates the models on the test dataset.
4. Saves the results to results_1000.json and results_graph_1000.json.

Run the script with:
python Main.py

--------------------------------------------------------------------------------

Results

- results_1000.json: Contains evaluation metrics (e.g., Hits@K, MRR) for Knowledge Graph Embedding models.
- results_graph_1000.json: Contains evaluation metrics for Graph-Based models.

--------------------------------------------------------------------------------

Models Trained

Knowledge Graph Embedding Models:
- TransE
- RotatE
- ComplEx
- DistMult
- TransH
- TransR
- TransD

Graph-Based Models:
- GraphSAGE
- R-GCN
- A2N
- CompGCN
- SE-GNN
- Convolutional GNN (if applicable)

--------------------------------------------------------------------------------

Contributing

Feel free to open an issue or submit a pull request if you would like to contribute to the project.

--------------------------------------------------------------------------------

License

This project is licensed under the MIT License.
