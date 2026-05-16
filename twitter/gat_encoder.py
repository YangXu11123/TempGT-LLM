import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import warnings
import glob
import re
import random

warnings.filterwarnings('ignore')

class SingleLayerGAT(nn.Module):
    """单层 GAT（baseline）"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.gat = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class TwoLayerGAT(nn.Module):
    """双层 GAT（8 heads × 2 layers）"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.gat1 = GATConv(input_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class ThreeLayerGAT(nn.Module):
    """三层 GAT（8 heads × 3 layers）"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.gat1 = GATConv(input_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.gat3 = GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gat3(h, edge_index)
        h = self.norm3(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class SingleLayerSAGE(nn.Module):
    """单层 GraphSAGE（attention-free ablation）"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, aggr: str = "mean"):
        super().__init__()
        self.aggr = aggr
        self.sage = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, aggr=aggr)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.sage(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class TwoLayerSAGE(nn.Module):
    """双层 GraphSAGE（对齐 GAT 2-layer ablation）"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, aggr: str = "mean"):
        super().__init__()
        self.aggr = aggr

        self.sage1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, aggr=aggr)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.sage2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr=aggr)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.sage1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.sage2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class LinkPredictor(nn.Module):
    """链路预测头（自监督预训练使用）"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return torch.tensor([], device=node_embeddings.device)

        src = node_embeddings[edge_index[0]]
        dst = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([src, dst], dim=1)
        prob = self.predictor(edge_emb).squeeze()
        return prob


class ExperimentalTemporalGATEncoder(nn.Module):

    def __init__(self,
                 input_dim: int = 15,
                 hidden_dim: int = 128,
                 num_heads: int = 8,     # 仅 GAT 使用
                 dropout: float = 0.1,
                 device: str = 'auto',
                 graph_encoder_type: str = "gat",  # "gat" or "sage"
                 gnn_layers: int = 1,              # GAT: 1/2/3；SAGE: 1/2
                 gat_layers: Optional[int] = None,  
                 sage_aggr: str = "mean"):
        super().__init__()

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.graph_encoder_type = str(graph_encoder_type).lower().strip()
        if gat_layers is not None:
            gnn_layers = int(gat_layers)
        self.gnn_layers = int(gnn_layers)
        self.sage_aggr = str(sage_aggr)

        self.gnn = self._build_gnn()
        self.link_predictor = LinkPredictor(hidden_dim)

        self.gnn_frozen = False
        self.to(self.device)

    def _build_gnn(self) -> nn.Module:
        if self.graph_encoder_type not in {"gat", "sage"}:
            raise ValueError(f"graph_encoder_type must be 'gat' or 'sage', got {self.graph_encoder_type}")

        if self.graph_encoder_type == "gat":
            if self.gnn_layers not in {1, 2, 3}:
                raise ValueError(f"For GAT, gnn_layers must be 1/2/3, got {self.gnn_layers}")
            if self.gnn_layers == 1:
                return SingleLayerGAT(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)
            if self.gnn_layers == 2:
                return TwoLayerGAT(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)
            return ThreeLayerGAT(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)

        # GraphSAGE
        if self.gnn_layers not in {1, 2}:
            raise ValueError(f"For GraphSAGE, gnn_layers must be 1/2, got {self.gnn_layers}")
        if self.gnn_layers == 1:
            return SingleLayerSAGE(self.input_dim, self.hidden_dim, self.dropout, aggr=self.sage_aggr)
        return TwoLayerSAGE(self.input_dim, self.hidden_dim, self.dropout, aggr=self.sage_aggr)

    # ---- freeze/unfreeze ----
    def freeze_gat_parameters(self):
        self.freeze_gnn_parameters()

    def unfreeze_gat_parameters(self):
        self.unfreeze_gnn_parameters()

    def freeze_gnn_parameters(self):
        for p in self.gnn.parameters():
            p.requires_grad = False
        self.gnn_frozen = True

    def unfreeze_gnn_parameters(self):
        for p in self.gnn.parameters():
            p.requires_grad = True
        self.gnn_frozen = False

    # ---- features / edges / encode ----
    def create_node_features(self, subgraph_data: Data) -> torch.Tensor:
        num_nodes = subgraph_data.num_nodes

        if hasattr(subgraph_data, 'x') and subgraph_data.x is not None:
            node_features = subgraph_data.x
            if node_features.size(1) != self.input_dim:
                if node_features.size(1) < self.input_dim:
                    padding = torch.zeros(num_nodes, self.input_dim - node_features.size(1), device=self.device)
                    node_features = torch.cat([node_features.to(self.device), padding], dim=1)
                else:
                    node_features = node_features[:, :self.input_dim].to(self.device)
            else:
                node_features = node_features.to(self.device)
        else:
            node_features = torch.randn(num_nodes, self.input_dim, device=self.device) * 0.1

        return node_features

    def reverse_edges_for_out_subgraph(self, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index
        return torch.stack([edge_index[1], edge_index[0]], dim=0)

    def apply_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.stack([
                torch.arange(num_nodes, device=x.device, dtype=torch.long),
                torch.arange(num_nodes, device=x.device, dtype=torch.long)
            ], dim=0)
        return self.gnn(x, edge_index)

    def encode_subgraph_with_experimental_design(self,
                                                 subgraph_data: Data,
                                                 is_out_subgraph: bool = False) -> torch.Tensor:
        if subgraph_data is None or subgraph_data.num_nodes == 0:
            return torch.zeros(0, self.hidden_dim, device=self.device)

        self.eval()
        with torch.no_grad():
            subgraph_data = subgraph_data.to(self.device)
            node_features = self.create_node_features(subgraph_data)

            if is_out_subgraph and subgraph_data.edge_index.numel() > 0:
                edge_index = self.reverse_edges_for_out_subgraph(subgraph_data.edge_index)
            else:
                edge_index = subgraph_data.edge_index

            return self.apply_gnn(node_features, edge_index)

    def encode_temporal_subgraphs(self,
                                 temporal_subgraphs: Dict[int, Tuple[Optional[Data], Optional[Data]]]
                                 ) -> Dict[int, Dict[str, torch.Tensor]]:
        self.eval()
        time_steps = sorted(temporal_subgraphs.keys())
        temporal_node_embeddings: Dict[int, Dict[str, torch.Tensor]] = {}

        with torch.no_grad():
            for t in time_steps:
                out_sg, in_sg = temporal_subgraphs[t]
                out_repr = self.encode_subgraph_with_experimental_design(out_sg, is_out_subgraph=True)
                in_repr = self.encode_subgraph_with_experimental_design(in_sg, is_out_subgraph=False)

                temporal_node_embeddings[t] = {
                    'out_node_embeddings': out_repr.cpu(),
                    'in_node_embeddings': in_repr.cpu()
                }

        return temporal_node_embeddings

    def create_negative_edges(self, num_nodes: int, num_pos_edges: int, pos_edge_set: set = None) -> torch.Tensor:
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        neg_edges = []
        max_attempts = max(num_pos_edges * 10, 1000)
        attempts = 0

        while len(neg_edges) < num_pos_edges and attempts < max_attempts:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst:
                e = (src, dst)
                if pos_edge_set is None or e not in pos_edge_set:
                    neg_edges.append([src, dst])
            attempts += 1

        if len(neg_edges) == 0:
            if num_nodes > 1:
                neg_edges = [[0, 1]]
            else:
                return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        return torch.tensor(neg_edges, dtype=torch.long).t().to(self.device)


def load_subgraph_data(subgraph_file: str) -> Dict:
    try:
        with open(subgraph_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def pretrain_gat_with_link_prediction(subgraph_files: List[str],
                                      encoder: ExperimentalTemporalGATEncoder,
                                      num_epochs: int = 10,
                                      lr: float = 0.001,
                                      max_samples: int = 100,  
                                      patience: int = 20) -> float:
    """
    链路预测自监督预训练：对 GAT / SAGE 通用
    """

    if encoder.gnn_frozen:
        encoder.unfreeze_gnn_parameters()

    encoder.train()
    optimizer = torch.optim.Adam(encoder.gnn.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    valid_subgraphs = []
    for subgraph_file in subgraph_files:
        try:
            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data or 'temporal_subgraphs' not in subgraph_data:
                continue

            temporal_subgraphs = subgraph_data['temporal_subgraphs']
            for _, (out_sg, in_sg) in temporal_subgraphs.items():
                for sg in [out_sg, in_sg]:
                    if (sg is not None and hasattr(sg, 'edge_index')
                            and sg.edge_index.numel() > 0 and sg.num_nodes > 1):
                        valid_subgraphs.append(sg)
        except Exception:
            continue

    if not valid_subgraphs:
        return float('inf')

    print(f"预训练 GNN: type={encoder.graph_encoder_type}, layers={encoder.gnn_layers}, heads={encoder.num_heads} | 有效子图={len(valid_subgraphs)}")

    best_loss = float('inf')
    epochs_no_improve = 0
    final_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Pretraining Epochs"):
        epoch_losses = []
        random.shuffle(valid_subgraphs)

        for sg in tqdm(valid_subgraphs, desc=f"Epoch {epoch+1}", leave=False):
            try:
                optimizer.zero_grad()
                sg = sg.to(encoder.device)
                node_features = encoder.create_node_features(sg)

                pos_edge_index = sg.edge_index
                num_pos = pos_edge_index.size(1)
                if num_pos == 0:
                    continue

                pos_edge_set = {(pos_edge_index[0, i].item(), pos_edge_index[1, i].item()) for i in range(num_pos)}
                neg_edge_index = encoder.create_negative_edges(sg.num_nodes, num_pos, pos_edge_set)
                if neg_edge_index.size(1) == 0:
                    continue

                node_embeddings = encoder.gnn(node_features, sg.edge_index)

                all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                labels = torch.cat([
                    torch.ones(num_pos),
                    torch.zeros(neg_edge_index.size(1))
                ]).to(encoder.device)

                edge_probs = encoder.link_predictor(node_embeddings, all_edge_index)
                if edge_probs.numel() == 0:
                    continue

                loss = criterion(edge_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            except Exception:
                continue

        if not epoch_losses:
            continue

        avg_loss = float(np.mean(epoch_losses))
        final_loss = avg_loss

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early Stopping triggered at epoch {epoch+1}")
            break

        tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    return final_loss


def process_subgraph_data_with_pretrained_gat(subgraph_data: Dict,
                                             encoder: ExperimentalTemporalGATEncoder) -> Dict:
    if 'temporal_subgraphs' not in subgraph_data:
        return {}

    temporal_subgraphs = subgraph_data['temporal_subgraphs']
    center_node = subgraph_data.get('center_node', 'unknown')
    k = subgraph_data.get('k', 2)
    time_steps = sorted(temporal_subgraphs.keys())

    valid_timesteps = 0
    for _, (out_sg, in_sg) in temporal_subgraphs.items():
        out_valid = (out_sg is not None and getattr(out_sg, 'num_nodes', 0) > 0)
        in_valid = (in_sg is not None and getattr(in_sg, 'num_nodes', 0) > 0)
        if out_valid or in_valid:
            valid_timesteps += 1

    if valid_timesteps == 0:
        return {
            'center_node': center_node,
            'k': k,
            'temporal_node_embeddings': {},
            'timesteps': torch.tensor(time_steps, dtype=torch.long),
            'num_timesteps': len(time_steps),
            'hidden_dim': encoder.hidden_dim,
            'valid_timesteps': 0,
            'encoding_info': {
                'graph_encoder_type': encoder.graph_encoder_type,
                'gnn_layers': encoder.gnn_layers,
                'gat_num_heads': encoder.num_heads,
                'sage_aggr': encoder.sage_aggr,
                'pretrained': True,
                'out_in_concat': False,
                'graph_level_pooling_in_this_file': False
            }
        }

    try:
        if not encoder.gnn_frozen:
            encoder.freeze_gnn_parameters()

        temporal_node_embeddings = encoder.encode_temporal_subgraphs(temporal_subgraphs)

        return {
            'center_node': center_node,
            'k': k,
            'temporal_node_embeddings': temporal_node_embeddings,
            'timesteps': torch.tensor(time_steps, dtype=torch.long),
            'num_timesteps': len(time_steps),
            'hidden_dim': encoder.hidden_dim,
            'valid_timesteps': valid_timesteps,
            'encoding_info': {
                'graph_encoder_type': encoder.graph_encoder_type,
                'gnn_layers': encoder.gnn_layers,
                'gat_num_heads': encoder.num_heads,
                'sage_aggr': encoder.sage_aggr,
                'pretrained': True,
                'out_in_concat': False,
                'graph_level_pooling_in_this_file': False,
                'experimental_design': True,
                'need_downstream_attention_pooling': True
            }
        }

    except Exception as e:
        return {
            'center_node': center_node,
            'k': k,
            'temporal_node_embeddings': {},
            'timesteps': torch.tensor(time_steps, dtype=torch.long),
            'num_timesteps': len(time_steps),
            'hidden_dim': encoder.hidden_dim,
            'valid_timesteps': 0,
            'error': str(e)
        }


def find_subgraph_files(input_dir: str = "subgraphs") -> List[str]:
    if not os.path.exists(input_dir):
        return []
    files = glob.glob(os.path.join(input_dir, "subgraph_*_k2.pkl"))
    files.sort()
    return files


def extract_node_id_from_filename(filename: str) -> str:
    match = re.search(r'subgraph_(.+?)_k2\.pkl', os.path.basename(filename))
    return match.group(1) if match else "unknown"


def batch_encode_subgraphs_experimental(input_dir: str = "subgraphs",
                                       output_dir: str = "gat_encoded_subgraphs",
                                       config: Dict = None) -> Dict:
    if config is None:
        config = {
            'input_dim': 15,
            'hidden_dim': 128,
            'num_heads': 8,
            'dropout': 0.1,
            'device': 'auto',
            'graph_encoder_type': 'gat',
            'gnn_layers': 1,
            'sage_aggr': 'mean',
        }

    os.makedirs(output_dir, exist_ok=True)

    subgraph_files = find_subgraph_files(input_dir)
    if not subgraph_files:
        return {}

    encoder = ExperimentalTemporalGATEncoder(**config)

    pretrain_loss = pretrain_gat_with_link_prediction(
        subgraph_files, encoder,
        num_epochs=10, lr=0.001, max_samples=100
    )

    encoder.freeze_gnn_parameters()

    stats = {
        'total_files': len(subgraph_files),
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'output_files': [],
        'total_size_mb': 0.0,
        'pretrain_loss': pretrain_loss,
        'graph_encoder_type': encoder.graph_encoder_type,
        'gnn_layers': encoder.gnn_layers,
        'num_heads': encoder.num_heads,
        'sage_aggr': encoder.sage_aggr
    }

    for subgraph_file in tqdm(subgraph_files, desc="GNN编码"):
        try:
            node_id = extract_node_id_from_filename(subgraph_file)

            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data:
                stats['failed'] += 1
                stats['failed_files'].append(subgraph_file)
                continue

            result = process_subgraph_data_with_pretrained_gat(subgraph_data, encoder)
            if not result:
                stats['failed'] += 1
                stats['failed_files'].append(subgraph_file)
                continue

            output_filename = f"subgraph_{node_id}_k2_gat_encoded.pkl"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'wb') as f:
                pickle.dump(result, f)

            stats['successful'] += 1
            stats['output_files'].append(output_path)
            stats['total_size_mb'] += os.path.getsize(output_path) / 1024 / 1024

        except Exception:
            stats['failed'] += 1
            stats['failed_files'].append(subgraph_file)
            continue

    stats_file = os.path.join(output_dir, "experimental_gnn_encoding_stats.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)

    return stats


def main():
    print("=" * 60)
    print("时序图编码器（节点级嵌入；支持 GAT/SAGE 与 1/2/3 层消融）")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ===================== 编码配置 =====================
    config = {
        'input_dim': 15,
        'hidden_dim': 128,
        'num_heads': 8,
        'dropout': 0.1,
        'device': device,
        'graph_encoder_type': 'gat',  # 'gat' / 'sage'
        'gnn_layers': 2,              # GAT: 1/2/3 ; SAGE: 1/2
        'sage_aggr': 'mean',
    }
    # ===========================================================

    tasks = [
        {
            "input_dir": "twitter_subgraphs",
            "output_dir": "gat_encoded_subgraphs"
        },
        {
            "input_dir": "twitter_subgraphs_normal_6000",
            "output_dir": "gat_encoded_subgraphs_normal_6000"
        }
    ]

    for task in tasks:
        print("\n" + "-" * 60)
        print(f"开始编码: {task['input_dir']} → {task['output_dir']}")
        print("-" * 60)

        stats = batch_encode_subgraphs_experimental(
            input_dir=task["input_dir"],
            output_dir=task["output_dir"],
            config=config
        )

        if not stats:
            print(f"  编码失败: {task['input_dir']}")
            continue

        print(f"  编码完成: {task['input_dir']}")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  成功编码: {stats['successful']}")
        print(f"  编码失败: {stats['failed']}")
        print(f"  预训练损失: {stats['pretrain_loss']:.4f}")
        print(f"  输出大小: {stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
