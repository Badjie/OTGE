import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import AGNNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC


class MGLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MGLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Transformation layer to match input and hidden dimension
        self.input_transform = nn.Linear(input_dim, hidden_dim)

        # AGNN layers with correct input and hidden state dimensions
        self.agnn_input = AGNNConv(requires_grad=True)
        self.agnn_hidden = AGNNConv(requires_grad=True)
        
        # Forget gate components
        self.agnn_forget_x = AGNNConv(requires_grad=True)
        self.agnn_forget_h = AGNNConv(requires_grad=True)
        
        # Input gate components
        self.agnn_input_x = AGNNConv(requires_grad=True)
        self.agnn_input_h = AGNNConv(requires_grad=True)
        
        # Output gate components
        self.agnn_output_x = AGNNConv(requires_grad=True)
        self.agnn_output_h = AGNNConv(requires_grad=True)
        
        # Candidate cell state components
        self.agnn_candidate_x = AGNNConv(requires_grad=True)
        self.agnn_candidate_h = AGNNConv(requires_grad=True)
        
        # Missing information prediction components
        self.W_gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_beta2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing_info(self, h_v, h_N_v):
        gamma = torch.tanh(self.W_gamma1(h_v) + self.W_gamma2(h_N_v))
        beta = torch.tanh(self.W_beta1(h_v) + self.W_beta2(h_N_v))
        r = torch.zeros_like(h_v)
        r_v = (gamma + 1) * r + beta
        m_v = h_v + r_v - h_N_v
        return m_v

    def forward(self, x, edge_index, h_c, batch=None):
        # Transform input to match hidden dimension
        x_transformed = self.input_transform(x)

        if h_c is None:
            h, c = (torch.zeros(x_transformed.size(0), self.hidden_dim, device=x.device),
                    torch.zeros(x_transformed.size(0), self.hidden_dim, device=x.device))
        else:
            h, c = h_c

        h_N = self.agnn_hidden(h, edge_index)
        m = self.predict_missing_info(h, h_N)
        h_N = h_N + m

        # Forget gate
        f = torch.sigmoid(self.agnn_forget_x(x_transformed, edge_index) + 
                          self.agnn_forget_h(h_N, edge_index))

        # Input gate
        i = torch.sigmoid(self.agnn_input_x(x_transformed, edge_index) + 
                          self.agnn_input_h(h_N, edge_index))

        # Candidate cell state
        c_tilde = torch.tanh(self.agnn_candidate_x(x_transformed, edge_index) + 
                             self.agnn_candidate_h(h_N, edge_index))

        # Update cell state
        c_new = f * c + i * c_tilde

        # Output gate
        o = torch.sigmoid(self.agnn_output_x(x_transformed, edge_index) + 
                          self.agnn_output_h(h_N, edge_index))

        # New hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new

'''
class EdgeClassifier:
    def __init__(self):
        self.model = SVC(probability=True)  # Enable probability estimates for ROC AUC calculation

    def fit(self, edge_embeddings, targets):
        self.model.fit(edge_embeddings, targets)

    def predict(self, edge_embeddings):
        return self.model.predict(edge_embeddings)

    def predict_proba(self, edge_embeddings):
        return self.model.predict_proba(edge_embeddings)[:, 1]  # Return probabilities for the positive class
'''

class TwoLayerSVMEdgeClassifier:
    def __init__(self):
        self.svm1 = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
        self.svm2 = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)

    def fit(self, edge_embeddings, targets):
        # Train the first-layer SVM
        self.svm1.fit(edge_embeddings, targets)

        # Use the first-layer SVM predictions as input to the second-layer SVM
        train_predictions_svm1 = self.svm1.predict(edge_embeddings)
        self.svm2.fit(np.column_stack((edge_embeddings, train_predictions_svm1)), targets)

    def predict(self, edge_embeddings):
        # Get the predictions from the first-layer SVM
        predictions_svm1 = self.svm1.predict(edge_embeddings)

        # Use the first-layer SVM predictions as input to the second-layer SVM
        predictions_svm2 = self.svm2.predict(np.column_stack((edge_embeddings, predictions_svm1)))
        return predictions_svm2

    def predict_proba(self, edge_embeddings):
        # Get the probability estimates from the second-layer SVM
        return self.svm2.predict_proba(np.column_stack((edge_embeddings, self.svm1.predict(edge_embeddings))))[:, 1]
    
class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EnhancedTemporalGraphNetwork, self).__init__()
        self.mglstm = MGLSTM(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.edge_classifier = TwoLayerSVMEdgeClassifier()  # Use the new EdgeClassifier

    def create_edge_embeddings(self, node_embeddings, edge_index):
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return edge_embeddings
    
    def forward(self, x, edge_index, batch=None):
        h_c = None
        for _ in range(self.num_layers):
            h, c = self.mglstm(x, edge_index, h_c, batch)
            h_c = (h, c)
        
        edge_embeddings = self.create_edge_embeddings(h, edge_index)
        return edge_embeddings, h  # Return edge embeddings instead of predictions
        
        
def create_edge_labels(G, labels, edge_index, node_to_idx):
    edge_labels = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()
        
        # Convert indices back to original node IDs
        src_node = list(G.nodes())[src_idx]
        dst_node = list(G.nodes())[dst_idx]
        
        # Create edge label based on source and destination node labels
        src_label = 1 if labels[src_node][0] == 'collection_abnormal' else 0
        dst_label = 1 if labels[dst_node][0] == 'collection_abnormal' else 0
        
        # Edge is labeled as abnormal if either source or destination is abnormal
        edge_labels.append(float(src_label or dst_label))
    
    return torch.tensor(edge_labels, dtype=torch.float)

def weighted_cross_entropy_loss(predictions, targets, pos_weight):
    """
    Custom weighted cross entropy loss
    N1: number of positive samples
    N2: number of negative samples
    """
    epsilon = 1e-7  # Small constant to prevent log(0)
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    
    # Calculate weighted loss
    loss = -(pos_weight * targets * torch.log(predictions) + 
             (1 - targets) * torch.log(1 - predictions))
    
    return loss.mean()

def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['time_stamp'])
    return G

def calculate_fraud_and_antifraud_scores(G):
    fraud_scores = nx.out_degree_centrality(G)
    antifraud_scores = nx.eigenvector_centrality(G, max_iter=1000)
    return fraud_scores, antifraud_scores

def label_nodes(fraud_scores, antifraud_scores, fraud_threshold=0.01, antifraud_threshold=0.01):
    labels = {}
    for node in fraud_scores:
        collection_label = 'collection_abnormal' if fraud_scores[node] > fraud_threshold else 'collection_normal'
        pay_label = 'pay_normal' if antifraud_scores[node] > antifraud_threshold else 'pay_abnormal'
        labels[node] = (collection_label, pay_label)
    return labels
def create_reachability_subgraph(G, node, max_depth=2):
    reachability_subgraph = nx.DiGraph()
    reachability_subgraph.add_node(node)
    current_level = {node}
    for depth in range(max_depth):
        next_level = set()
        for u in current_level:
            for v in G.successors(u):
                if v not in reachability_subgraph:
                    reachability_subgraph.add_edge(u, v, weight=G[u][v]['weight'])
                    next_level.add(v)
        current_level = next_level
    return reachability_subgraph


def label_edges(G, max_depth=2):
    reachability_networks = defaultdict(nx.DiGraph)
    for node in G.nodes:
        reachability_networks[node] = create_reachability_subgraph(G, node, max_depth)
    return reachability_networks


def count_edges(reachability_networks, label):
    count = 0
    for u, v, data in reachability_networks.edges(data=True):
        if label in data:
            count += 1
    return count

def common_eval(reachability_networks):
    neighbors = {}
    for node, reach_net in reachability_networks.items():
        neighbors[node] = list(reach_net.neighbors(node))
    return neighbors

def extract_features(G, node):
    reachability_networks = label_edges(G, max_depth=2)
    neighbors = common_eval(reachability_networks)
    T1 = count_edges(reachability_networks[node], label='collection_normal')
    T2 = count_edges(reachability_networks[node], label='collection_abnormal')
    T3 = count_edges(reachability_networks[node], label='payment_normal')
    T4 = count_edges(reachability_networks[node], label='payment_abnormal')
    
    node_features = [T1, T2, len(neighbors.get(node, [])), 
                    T3, T4, len(neighbors.get(node, [])), 
                    G.in_degree(node), G.out_degree(node)]
    return node_features

def create_data_list(G_list, labels_list):
    data_list = []
    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
        
        # Create edge index for the entire graph
        edge_index = torch.tensor([(node_to_idx[u], node_to_idx[v]) 
                                  for u, v in G.edges], dtype=torch.long).t().contiguous()
        
        # Create feature matrix for all nodes
        x = torch.tensor([extract_features(G, node) for node in G.nodes], 
                        dtype=torch.float)
        
        # Create edge labels
        edge_labels = create_edge_labels(G, labels, edge_index, node_to_idx)
        
        # Create a single Data object for the entire graph
        data = Data(x=x, edge_index=edge_index, y=edge_labels)
        data_list.append(data)
    
    return data_list
'''    
def train_model(model, train_loader, epochs=10):
    model.train()
    all_edge_embeddings = []
    all_labels = []
    
    for epoch in range(epochs):
        for data in train_loader:
            edge_embeddings, _ = model(data.x, data.edge_index, data.batch)
            all_edge_embeddings.append(edge_embeddings.detach().cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    # Concatenate all edge embeddings and labels
    all_edge_embeddings = np.concatenate(all_edge_embeddings)
    all_labels = np.concatenate(all_labels)

    # Train SVM on edge embeddings
    model.edge_classifier.fit(all_edge_embeddings, all_labels)

    print(f'Model trained for {epochs} epochs using SVM.')
    
    return model
'''
def train_model(model, train_loader, epochs=150):
    model.train()
    all_edge_embeddings = []
    all_labels = []

    for epoch in range(epochs):
        for data in train_loader:
            edge_embeddings, _ = model(data.x, data.edge_index, data.batch)
            all_edge_embeddings.append(edge_embeddings.detach().cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    # Concatenate all edge embeddings and labels
    all_edge_embeddings = np.concatenate(all_edge_embeddings)
    all_labels = np.concatenate(all_labels)

    # Train the two-layer SVM on edge embeddings
    model.edge_classifier.fit(all_edge_embeddings, all_labels)

    print(f'Model trained for {epochs} epochs using two-layer SVM.')

    return model

def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            edge_embeddings, _ = model(batch.x, batch.edge_index, batch.batch)
            predictions = model.edge_classifier.predict_proba(edge_embeddings)
            all_predictions.append(predictions)
            all_labels.append(batch.y.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    auc_score = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    recall = recall_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    
    return auc_score, precision, recall, f1

def main():
    
    # Load the dataset
    #file_path = 'reddit.csv'
    file_path = 'token_transfers_erc20.csv'
    data = pd.read_csv(file_path)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['time_stamp'], unit='s')

    # Sort data by timestamp
    data = data.sort_values(by='timestamp')

    # Split data into 31-day time slices
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31

    # Combine the last few sparse time slices into a single time slice
    threshold = 5  # Combine slices with fewer than 20 entries
    combined_time_slice = max(data['time_slice']) - 1
    data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    # Verify the new distribution of entries across time slices
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()

    # Display the new distribution
    print(new_time_slice_counts)

    
    # Create a list of graphs and labels for each time slice
    G_list = []
    labels_list = []
    
    for time_slice in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == time_slice]
        G = create_graph(slice_data)
        fraud_scores, antifraud_scores = calculate_fraud_and_antifraud_scores(G)
        labels = label_nodes(fraud_scores, antifraud_scores)
        G_list.append(G)
        labels_list.append(labels)
    
    # Check if we have enough data slices for splitting
    if len(G_list) > 2:
        # Split data into train, validation, and test sets (60%, 20%, 20%)
        train_G, temp_G, train_labels, temp_labels = train_test_split(G_list, labels_list, test_size=0.3, shuffle=False)
        val_G, test_G, val_labels, test_labels = train_test_split(temp_G, temp_labels, test_size=0.25, shuffle=False)
    else:
        # If there's not enough time slices, use all data for training and skip validation/testing
        train_G, train_labels = G_list, labels_list
        val_G, val_labels, test_G, test_labels = [], [], [], []
    
    # Create dataset and dataloader
    train_data_list = create_data_list(train_G, train_labels)
    val_data_list = create_data_list(val_G, val_labels) if val_G else []
    test_data_list = create_data_list(test_G, test_labels) if test_G else []
    
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=32, shuffle=False) if val_data_list else None
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False) if test_data_list else None
    
    # Initialize and train model
    model = EnhancedTemporalGraphNetwork(
        input_dim=8,
        hidden_dim=16,
        num_layers=2
    )
    
    # Train the model
    trained_model = train_model(model, train_loader)
    
    # Evaluate the model on the test set if it exists
    if test_loader:
        test_auc, precision, recall, f1 = evaluate_model(trained_model, test_loader)
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
    else:
        print("Not enough data to create a test set.")
    
    return trained_model

if __name__ == "__main__":
    trained_model = main()            
