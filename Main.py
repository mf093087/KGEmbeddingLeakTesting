import torch
from torch.nn import functional as F
import os
import time
import json
import random

# -----------------------------------------------------------------------------
# Load dataset from train, valid, and test files
# -----------------------------------------------------------------------------
def load_dataset(path):
    def load_triples(file_path, entity_map, relation_map):
        triples = []
        with open(file_path, "r") as f:
            for line in f:
                try:
                    head, relation, tail = line.strip().split("\t")
                    head_id = entity_map[head]
                    relation_id = relation_map[relation]
                    tail_id = entity_map[tail]
                    triples.append((head_id, relation_id, tail_id))
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
        return triples

    # Build entity and relation maps
    entities = set()
    relations = set()
    for file_name in ["train_enhanced.txt", "valid_enhanced.txt", "test_enhanced.txt"]:
        with open(os.path.join(path, file_name), "r") as f:
            for line in f:
                try:
                    head, relation, tail = line.strip().split("\t")
                    entities.add(head)
                    entities.add(tail)
                    relations.add(relation)
                except ValueError:
                    print(f"Skipping malformed line during map creation: {line.strip()}")

    entity_map = {entity: idx for idx, entity in enumerate(sorted(entities))}
    relation_map = {relation: idx for idx, relation in enumerate(sorted(relations))}

    # Load triples with mapped IDs
    train_triples = load_triples(os.path.join(path, "train_enhanced.txt"), entity_map, relation_map)
    valid_triples = load_triples(os.path.join(path, "valid_enhanced.txt"), entity_map, relation_map)
    test_triples = load_triples(os.path.join(path, "test_enhanced.txt"), entity_map, relation_map)

    num_entities = len(entity_map)
    num_relations = len(relation_map)

    return train_triples, valid_triples, test_triples, num_entities, num_relations


# -----------------------------------------------------------------------------
# Knowledge Graph Embedding Models
# -----------------------------------------------------------------------------
# TransE Model
class TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        return -torch.norm(h + r - t, p=2, dim=1)  # Minimize distance


# RotatE Model
class RotatE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatE, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        # entity_emb is of size (num_entities, embedding_dim*2)
        h = self.entity_emb(head).view(-1, 2, self.entity_emb.embedding_dim // 2)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail).view(-1, 2, self.entity_emb.embedding_dim // 2)

        # r_phase shape: (batch_size, 2, embedding_dim) but effectively for Cos/Sin
        r_phase = torch.stack((torch.cos(r), torch.sin(r)), dim=1)

        # h_rot shape is (batch_size, 2, embedding_dim/2)
        h_rot = torch.stack(
            (
                h[:, 0] * r_phase[:, 0] - h[:, 1] * r_phase[:, 1],
                h[:, 0] * r_phase[:, 1] + h[:, 1] * r_phase[:, 0]
            ),
            dim=1
        )
        return -torch.norm(h_rot - t, dim=(1, 2))


# ComplEx Model
class ComplEx(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEx, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim * 2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h_real, h_imag = torch.chunk(self.entity_emb(head), 2, dim=1)
        r_real, r_imag = torch.chunk(self.relation_emb(relation), 2, dim=1)
        t_real, t_imag = torch.chunk(self.entity_emb(tail), 2, dim=1)

        score_real = h_real * r_real * t_real + h_imag * r_imag * t_imag
        score_imag = h_real * r_imag * t_imag - h_imag * r_real * t_real
        return -torch.sum(score_real + score_imag, dim=1)


# DistMult Model
class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        return torch.sum(h * r * t, dim=1)


# TransH Model
class TransH(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransH, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.normal_vec = torch.nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)
        torch.nn.init.xavier_uniform_(self.normal_vec.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        n = self.normal_vec(relation)

        # Project h and t onto the hyperplane
        h_proj = h - torch.sum(h * n, dim=1, keepdim=True) * n
        t_proj = t - torch.sum(t * n, dim=1, keepdim=True) * n

        return -torch.norm(h_proj + r - t_proj, p=2, dim=1)


# TransR Model
class TransR(torch.nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim):
        super(TransR, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, entity_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, relation_dim)
        self.transfer_matrix = torch.nn.Embedding(num_relations, entity_dim * relation_dim)
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)
        torch.nn.init.xavier_uniform_(self.transfer_matrix.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        t = self.entity_emb(tail)
        r = self.relation_emb(relation)
        M = self.transfer_matrix(relation).view(-1, self.entity_dim, self.relation_dim)

        h_proj = torch.matmul(h.unsqueeze(1), M).squeeze(1)
        t_proj = torch.matmul(t.unsqueeze(1), M).squeeze(1)

        return -torch.norm(h_proj + r - t_proj, p=2, dim=1)


# TransD Model
class TransD(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransD, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.entity_proj = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_proj = torch.nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)
        torch.nn.init.xavier_uniform_(self.entity_proj.weight)
        torch.nn.init.xavier_uniform_(self.relation_proj.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        h_proj = self.entity_proj(head)
        r_proj = self.relation_proj(relation)
        t_proj = self.entity_proj(tail)

        # Simple TransD formula:
        h_trans = h + torch.sum(h_proj * r_proj, dim=1, keepdim=True)
        t_trans = t + torch.sum(t_proj * r_proj, dim=1, keepdim=True)

        return -torch.norm(h_trans + r - t_trans, p=2, dim=1)


# -----------------------------------------------------------------------------
# Graph-based Models
# -----------------------------------------------------------------------------
# GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(GraphSAGE, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        # Simple linear layer that aggregates [entity, relation]
        self.sage_layer = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        combined = torch.cat([h, r], dim=1)
        projected = self.sage_layer(combined)
        return -torch.norm(projected - t, p=2, dim=1)


# Relational Graph Convolutional Network (R-GCN)
class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RGCN, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        # For simplicity, just a linear layer that aggregates [entity, relation]
        self.rgcn_layer = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        combined = torch.cat([h, r], dim=1)
        projected = self.rgcn_layer(combined)
        return -torch.norm(projected - t, p=2, dim=1)


# Attention Aggregation Network (A2N)
class A2N(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(A2N, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.attention_layer = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)
        torch.nn.init.xavier_uniform_(self.attention_layer.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        combined = torch.cat([h, r], dim=1)
        attended = self.attention_layer(combined)
        return -torch.norm(attended - t, p=2, dim=1)


# Compositional Graph Convolutional Network (CompGCN)
class CompGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(CompGCN, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.compgcn_layer = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        combined = torch.cat([h, r], dim=1)
        compositional = self.compgcn_layer(combined)
        return -torch.norm(compositional - t, p=2, dim=1)


# Structural Entropy Graph Neural Network (SE-GNN)
class SE_GNN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(SE_GNN, self).__init__()
        self.entity_emb = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = torch.nn.Embedding(num_relations, embedding_dim)
        self.structure_layer = torch.nn.Linear(embedding_dim * 2, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.entity_emb.weight)
        torch.nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        combined = torch.cat([h, r], dim=1)
        structured = self.structure_layer(combined)
        return -torch.norm(structured - t, p=2, dim=1)


# -----------------------------------------------------------------------------
# Evaluation: Hits@K (and MRR)
# -----------------------------------------------------------------------------
def evaluate_hits_k(model, triples, num_entities, k_values=[1, 3, 10]):
    model.eval()
    hits_at_k = {k: 0 for k in k_values}
    mrr = 0.0  # Mean Reciprocal Rank
    start_time = time.time()
    total_triples = len(triples)

    with torch.no_grad():
        for i, (head, relation, tail) in enumerate(triples):
            head_tensor = torch.tensor([head], dtype=torch.long)
            relation_tensor = torch.tensor([relation], dtype=torch.long)
            tail_tensor = torch.tensor([tail], dtype=torch.long)

            all_tails = torch.arange(num_entities, dtype=torch.long)
            scores = model(
                head_tensor.repeat(len(all_tails)),
                relation_tensor.repeat(len(all_tails)),
                all_tails
            )

            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == tail_tensor).nonzero(as_tuple=True)[0].item() + 1

            mrr += 1.0 / rank  # Reciprocal rank

            for k in k_values:
                if rank <= k:
                    hits_at_k[k] += 1

            # Optional: print progress
            if (i + 1) % 10 == 0 or (i + 1) == total_triples:
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (i + 1)) * (total_triples - (i + 1))
                print(f"Processed {i + 1}/{total_triples} triples. "
                      f"Elapsed time: {elapsed_time:.2f}s. "
                      f"Estimated remaining time: {remaining_time:.2f}s.")

    hits_at_k = {k: v / len(triples) for k, v in hits_at_k.items()}
    mrr /= len(triples)
    return hits_at_k, mrr


# -----------------------------------------------------------------------------
# Train Model for KGE Models
# -----------------------------------------------------------------------------
def train_model(model, optimizer, train_triples, valid_triples, num_entities, batch_size=4096, epochs=1000):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(train_triples), batch_size):
            batch = train_triples[i:i + batch_size]
            head, relation, tail = zip(*batch)
            head = torch.tensor(head, dtype=torch.long)
            relation = torch.tensor(relation, dtype=torch.long)
            tail = torch.tensor(tail, dtype=torch.long)

            # Negative sampling
            neg_tail = torch.randint(0, num_entities, tail.shape)

            optimizer.zero_grad()
            pos_score = model(head, relation, tail)
            neg_score = model(head, relation, neg_tail)

            loss = F.margin_ranking_loss(
                pos_score,
                neg_score,
                torch.ones_like(pos_score),
                margin=1.0
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


# -----------------------------------------------------------------------------
# Train Model with Early Stopping Based on Loss for Graph-based Models
# -----------------------------------------------------------------------------
def train_model_with_loss_early_stopping(model, optimizer, train_triples, valid_triples, num_entities, batch_size=4096, epochs=1000):
    # Optionally, you can set a patience mechanism.
    # For this demonstration, the code just runs for the specified epochs.

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(train_triples), batch_size):
            batch = train_triples[i:i + batch_size]
            head, relation, tail = zip(*batch)
            head = torch.tensor(head, dtype=torch.long)
            relation = torch.tensor(relation, dtype=torch.long)
            tail = torch.tensor(tail, dtype=torch.long)

            # Negative sampling
            neg_tail = torch.randint(0, num_entities, tail.shape)

            optimizer.zero_grad()
            pos_score = model(head, relation, tail)
            neg_score = model(head, relation, neg_tail)
            loss = F.margin_ranking_loss(pos_score, neg_score, torch.ones_like(pos_score), margin=1.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


# -----------------------------------------------------------------------------
# Save Results to File
# -----------------------------------------------------------------------------
def save_results(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


# -----------------------------------------------------------------------------
# Main Scripts
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = "D:\\Users\\mofan\\PycharmProjects\\leaktesting\\AnKGE-main\\AnKGE-main\\dataset\\leakenhanced"

    # Load the dataset
    train_triples, valid_triples, test_triples, num_entities, num_relations = load_dataset(dataset_path)
    training_sample = train_triples
    valid_sample = valid_triples
    test_sample = test_triples

    embedding_dim = 50
    results = {}

    # ---------------- KGE Models ----------------
    # 1. TransE
    transe_model = TransE(num_entities, num_relations, embedding_dim)
    transe_optimizer = torch.optim.Adam(transe_model.parameters(), lr=0.01)
    print("Training TransE...")
    train_model(transe_model, transe_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating TransE on Test Set...")
    transe_hits_k, transe_mrr = evaluate_hits_k(transe_model, test_sample, num_entities)
    results['TransE'] = {"Hits@K": transe_hits_k, "MRR": transe_mrr}

    # 2. RotatE
    rotate_model = RotatE(num_entities, num_relations, embedding_dim)
    rotate_optimizer = torch.optim.Adam(rotate_model.parameters(), lr=0.01)
    print("Training RotatE...")
    train_model(rotate_model, rotate_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating RotatE on Test Set...")
    rotate_hits_k, rotate_mrr = evaluate_hits_k(rotate_model, test_sample, num_entities)
    results['RotatE'] = {"Hits@K": rotate_hits_k, "MRR": rotate_mrr}

    # 3. ComplEx
    complex_model = ComplEx(num_entities, num_relations, embedding_dim)
    complex_optimizer = torch.optim.Adam(complex_model.parameters(), lr=0.01)
    print("Training ComplEx...")
    train_model(complex_model, complex_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating ComplEx on Test Set...")
    complex_hits_k, complex_mrr = evaluate_hits_k(complex_model, test_sample, num_entities)
    results['ComplEx'] = {"Hits@K": complex_hits_k, "MRR": complex_mrr}

    # 4. DistMult
    distmult_model = DistMult(num_entities, num_relations, embedding_dim)
    distmult_optimizer = torch.optim.Adam(distmult_model.parameters(), lr=0.01)
    print("Training DistMult...")
    train_model(distmult_model, distmult_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating DistMult on Test Set...")
    distmult_hits_k, distmult_mrr = evaluate_hits_k(distmult_model, test_sample, num_entities)
    results['DistMult'] = {"Hits@K": distmult_hits_k, "MRR": distmult_mrr}

    # 5. TransH
    transh_model = TransH(num_entities, num_relations, embedding_dim)
    transh_optimizer = torch.optim.Adam(transh_model.parameters(), lr=0.01)
    print("Training TransH...")
    train_model(transh_model, transh_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating TransH on Test Set...")
    transh_hits_k, transh_mrr = evaluate_hits_k(transh_model, test_sample, num_entities)
    results['TransH'] = {"Hits@K": transh_hits_k, "MRR": transh_mrr}

    # 6. TransR
    transr_model = TransR(num_entities, num_relations, embedding_dim, embedding_dim)
    transr_optimizer = torch.optim.Adam(transr_model.parameters(), lr=0.01)
    print("Training TransR...")
    train_model(transr_model, transr_optimizer, training_sample, valid_sample, num_entities)
    # Evaluate TransR on test set if desired (omitted in original snippet)

    # 7. TransD
    transd_model = TransD(num_entities, num_relations, embedding_dim)
    transd_optimizer = torch.optim.Adam(transd_model.parameters(), lr=0.01)
    print("Training TransD...")
    train_model(transd_model, transd_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating TransD on Test Set...")
    transd_hits_k, transd_mrr = evaluate_hits_k(transd_model, test_sample, num_entities)
    results['TransD'] = {"Hits@K": transd_hits_k, "MRR": transd_mrr}

    # Save KGE results to file
    save_results("results_1000.json", results)

    # ---------------- Graph-based Models ----------------
    embedding_dim_graph = 64  # e.g. 64 for graph models
    graph_results = {}

    # 1. GraphSAGE
    graphsage_model = GraphSAGE(num_entities, num_relations, embedding_dim_graph)
    graphsage_optimizer = torch.optim.Adam(graphsage_model.parameters(), lr=0.01)
    print("Training GraphSAGE...")
    train_model_with_loss_early_stopping(graphsage_model, graphsage_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating GraphSAGE on Test Set...")
    graphsage_hits_k, graphsage_mrr = evaluate_hits_k(graphsage_model, test_sample, num_entities)
    graph_results['GraphSAGE'] = {"Hits@K": graphsage_hits_k, "MRR": graphsage_mrr}

    # 2. R-GCN
    rgcn_model = RGCN(num_entities, num_relations, embedding_dim_graph)
    rgcn_optimizer = torch.optim.Adam(rgcn_model.parameters(), lr=0.01)
    print("Training R-GCN...")
    train_model_with_loss_early_stopping(rgcn_model, rgcn_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating R-GCN on Test Set...")
    rgcn_hits_k, rgcn_mrr = evaluate_hits_k(rgcn_model, test_sample, num_entities)
    graph_results['R-GCN'] = {"Hits@K": rgcn_hits_k, "MRR": rgcn_mrr}

    # 3. A2N
    a2n_model = A2N(num_entities, num_relations, embedding_dim_graph)
    a2n_optimizer = torch.optim.Adam(a2n_model.parameters(), lr=0.01)
    print("Training A2N...")
    train_model_with_loss_early_stopping(a2n_model, a2n_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating A2N on Test Set...")
    a2n_hits_k, a2n_mrr = evaluate_hits_k(a2n_model, test_sample, num_entities)
    graph_results['A2N'] = {"Hits@K": a2n_hits_k, "MRR": a2n_mrr}

    # 4. CompGCN
    compgcn_model = CompGCN(num_entities, num_relations, embedding_dim_graph)
    compgcn_optimizer = torch.optim.Adam(compgcn_model.parameters(), lr=0.01)
    print("Training CompGCN...")
    train_model_with_loss_early_stopping(compgcn_model, compgcn_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating CompGCN on Test Set...")
    compgcn_hits_k, compgcn_mrr = evaluate_hits_k(compgcn_model, test_sample, num_entities)
    graph_results['CompGCN'] = {"Hits@K": compgcn_hits_k, "MRR": compgcn_mrr}

    # 5. SE-GNN
    segnn_model = SE_GNN(num_entities, num_relations, embedding_dim_graph)
    segnn_optimizer = torch.optim.Adam(segnn_model.parameters(), lr=0.01)
    print("Training SE-GNN...")
    train_model_with_loss_early_stopping(segnn_model, segnn_optimizer, training_sample, valid_sample, num_entities)
    print("Evaluating SE-GNN on Test Set...")
    segnn_hits_k, segnn_mrr = evaluate_hits_k(segnn_model, test_sample, num_entities)
    graph_results['SE-GNN'] = {"Hits@K": segnn_hits_k, "MRR": segnn_mrr}

    # Save graph results to file
    save_results("results_graph_1000.json", graph_results)
