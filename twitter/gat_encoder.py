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
import copy
import json
import hashlib
import datetime

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(CODE_DIR)
ARTIFACT_ROOT = os.path.join(WORKSPACE_ROOT, "new_exe_artifacts")

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
                                                 is_out_subgraph: bool = False,
                                                 edge_drop_rate: float = 0.0,
                                                 edge_drop_rng: Optional[np.random.RandomState] = None) -> torch.Tensor:
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

            if edge_drop_rate > 0.0 and edge_drop_rng is not None:
                edge_index = drop_edges(edge_index, edge_drop_rate, edge_drop_rng)

            return self.apply_gnn(node_features, edge_index)

    def encode_temporal_subgraphs(self,
                                 temporal_subgraphs: Dict[int, Tuple[Optional[Data], Optional[Data]]],
                                 edge_drop_rate: float = 0.0,
                                 edge_drop_seed: Optional[int] = None,
                                 file_seed: int = 0,
                                 ) -> Dict[int, Dict[str, torch.Tensor]]:
        self.eval()
        time_steps = sorted(temporal_subgraphs.keys())
        temporal_node_embeddings: Dict[int, Dict[str, torch.Tensor]] = {}

        with torch.no_grad():
            for t in time_steps:
                out_sg, in_sg = temporal_subgraphs[t]
                out_rng = in_rng = None
                if edge_drop_rate > 0.0 and edge_drop_seed is not None:
                    # 逐 (文件, 时间步, in/out) 独立确定性采样，保证增强视图可复现
                    out_rng = np.random.RandomState([int(edge_drop_seed), int(file_seed), int(t), 1])
                    in_rng = np.random.RandomState([int(edge_drop_seed), int(file_seed), int(t), 0])
                out_repr = self.encode_subgraph_with_experimental_design(
                    out_sg, is_out_subgraph=True,
                    edge_drop_rate=edge_drop_rate, edge_drop_rng=out_rng,
                )
                in_repr = self.encode_subgraph_with_experimental_design(
                    in_sg, is_out_subgraph=False,
                    edge_drop_rate=edge_drop_rate, edge_drop_rng=in_rng,
                )

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


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def drop_edges(edge_index: torch.Tensor, drop_rate: float, rng: np.random.RandomState) -> torch.Tensor:
    """随机丢弃比例 drop_rate 的边（L_cont 的图增强，修改方向.md 二.1）"""
    if edge_index is None or edge_index.numel() == 0 or drop_rate <= 0.0:
        return edge_index

    num_edges = edge_index.size(1)
    keep_mask = rng.rand(num_edges) >= float(drop_rate)
    if not keep_mask.any():
        return edge_index.new_zeros((2, 0))
    return edge_index[:, keep_mask]


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
    optimizer = torch.optim.Adam(
        list(encoder.gnn.parameters()) + list(encoder.link_predictor.parameters()),
        lr=lr,
    )
    criterion = torch.nn.BCELoss()

    valid_subgraphs = []
    seen_valid = 0
    reservoir_rng = random.Random(42)
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
                        # max_samples 原先未生效，会把近百万个累计快照同时留在内存中。
                        # 用确定性的 reservoir sampling 固定抽取预训练快照。
                        seen_valid += 1
                        if max_samples is None or max_samples <= 0:
                            valid_subgraphs.append(sg)
                        elif len(valid_subgraphs) < max_samples:
                            valid_subgraphs.append(sg)
                        else:
                            replace_idx = reservoir_rng.randrange(seen_valid)
                            if replace_idx < max_samples:
                                valid_subgraphs[replace_idx] = sg
        except Exception:
            continue

    if not valid_subgraphs:
        return float('inf')

    print(f"预训练 GNN: type={encoder.graph_encoder_type}, layers={encoder.gnn_layers}, heads={encoder.num_heads} | 有效子图={len(valid_subgraphs)}")

    best_loss = float('inf')
    best_state = None
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
            best_state = copy.deepcopy(encoder.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early Stopping triggered at epoch {epoch+1}")
            break

        tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    if best_state is None:
        raise RuntimeError("GNN 预训练没有产生任何有效 batch")
    encoder.load_state_dict(best_state, strict=True)
    return best_loss


def process_subgraph_data_with_pretrained_gat(subgraph_data: Dict,
                                             encoder: ExperimentalTemporalGATEncoder,
                                             edge_drop_rate: float = 0.0,
                                             edge_drop_seed: Optional[int] = None,
                                             file_seed: int = 0) -> Dict:
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
                'graph_level_pooling_in_this_file': False,
                'edge_drop_rate': float(edge_drop_rate),
                'edge_drop_seed': edge_drop_seed,
            }
        }

    try:
        if not encoder.gnn_frozen:
            encoder.freeze_gnn_parameters()

        temporal_node_embeddings = encoder.encode_temporal_subgraphs(
            temporal_subgraphs,
            edge_drop_rate=edge_drop_rate,
            edge_drop_seed=edge_drop_seed,
            file_seed=file_seed,
        )

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
                'need_downstream_attention_pooling': True,
                'edge_drop_rate': float(edge_drop_rate),
                'edge_drop_seed': edge_drop_seed,
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


def collect_encoded_ids(directory: str, suffix: str) -> List[str]:
    pattern = os.path.join(directory, f"subgraph_*_k2_{suffix}.pkl")
    ids = []
    marker = f"_k2_{suffix}.pkl"
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if name.startswith("subgraph_") and name.endswith(marker):
            ids.append(name[len("subgraph_"):-len(marker)])
    return ids


def collect_subgraph_ids(directory: str) -> List[str]:
    return [extract_node_id_from_filename(path) for path in find_subgraph_files(directory)]


def assert_exact_id_alignment(left_name: str, left_ids: List[str], right_name: str, right_ids: List[str]) -> None:
    left_set, right_set = set(left_ids), set(right_ids)
    only_left = sorted(left_set - right_set)
    only_right = sorted(right_set - left_set)
    if only_left or only_right or len(left_ids) != len(left_set) or len(right_ids) != len(right_set):
        raise RuntimeError(
            f"账号 ID 未严格对齐: {left_name}={len(left_ids)}, {right_name}={len(right_ids)}, "
            f"only_{left_name}={only_left[:10]}, only_{right_name}={only_right[:10]}"
        )


def write_gat_manifest(path: str, groups: List[Dict], edge_drop_dirs: Dict[str, str]) -> None:
    payload = {
        "created_at": datetime.datetime.now().isoformat(),
        "edge_drop_dirs": edge_drop_dirs,
        "groups": groups,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def batch_encode_subgraphs_experimental(input_dir: str = "subgraphs",
                                       output_dir: str = "gat_encoded_subgraphs",
                                       config: Dict = None,
                                       pretrained_encoder_state_path: Optional[str] = None) -> Dict:
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

    _set_global_seed(int(config.get('seed', 42)))

    encoder = ExperimentalTemporalGATEncoder(**config)

    if pretrained_encoder_state_path:
        if not os.path.isfile(pretrained_encoder_state_path):
            raise FileNotFoundError(
                f"共享 GNN 权重不存在: {pretrained_encoder_state_path}"
            )
        shared_checkpoint = torch.load(
            pretrained_encoder_state_path,
            map_location=encoder.device,
            weights_only=False,
        )
        encoder.load_state_dict(shared_checkpoint["encoder_state_dict"], strict=True)
        pretrain_loss = float(shared_checkpoint.get("pretrain_loss", float("nan")))
        print(f"复用主数据预训练的共享 GNN: {pretrained_encoder_state_path}")
    else:
        pretrain_loss = pretrain_gat_with_link_prediction(
            subgraph_files, encoder,
            num_epochs=10, lr=0.001, max_samples=100
        )

    encoder.freeze_gnn_parameters()

    # 保存编码器权重，供 edge-drop 增强视图复用同一冻结 GNN（保证两视角同源）
    encoder_state_path = os.path.join(output_dir, "gnn_encoder_state.pth")
    torch.save(
        {
            'encoder_state_dict': encoder.state_dict(),
            'encoder_config': config,
            'pretrain_loss': pretrain_loss,
            'shared_source_state_path': pretrained_encoder_state_path,
        },
        encoder_state_path,
    )

    stats = {
        'total_files': len(subgraph_files),
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'output_files': [],
        'total_size_mb': 0.0,
        'pretrain_loss': pretrain_loss,
        'encoder_state_path': encoder_state_path,
        'shared_source_state_path': pretrained_encoder_state_path,
        'graph_encoder_type': encoder.graph_encoder_type,
        'gnn_layers': encoder.gnn_layers,
        'num_heads': encoder.num_heads,
        'sage_aggr': encoder.sage_aggr
    }

    for file_idx, subgraph_file in enumerate(tqdm(subgraph_files, desc="GNN编码")):
        try:
            node_id = extract_node_id_from_filename(subgraph_file)

            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data:
                stats['failed'] += 1
                stats['failed_files'].append(subgraph_file)
                continue

            result = process_subgraph_data_with_pretrained_gat(subgraph_data, encoder, file_seed=file_idx)
            if not result or not result.get('temporal_node_embeddings'):
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


def batch_encode_subgraphs_edge_drop(input_dir: str = "subgraphs",
                                     output_dir: str = "gat_encoded_subgraphs_edge_drop",
                                     original_output_dir: str = "gat_encoded_subgraphs",
                                     config: Dict = None,
                                     edge_drop_rate: float = 0.2,
                                     edge_drop_seed: int = 42,
                                     stats_filename: str = "edge_drop_encoding_stats.pkl") -> Dict:
    """生成 L_cont 所需的 edge-dropout 增强结构视图（修改方向.md 二.1）。

    - 复用 original_output_dir 中保存的冻结 GNN 权重，保证增强视图与原始视图
      来自同一个编码器（正样本 = 同一账号的两个增强视角）；
    - 逐 (文件, 时间步, in/out) 独立确定性采样，重复运行结果完全一致；
    - 输出文件名与原始编码一致，train.py 的 edge-drop 加载逻辑可直接识别。
    """
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

    encoder_state_path = os.path.join(original_output_dir, "gnn_encoder_state.pth")
    if os.path.exists(encoder_state_path):
        checkpoint = torch.load(encoder_state_path, map_location=encoder.device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
        encoder.freeze_gnn_parameters()
        print(f"已加载原始视图的冻结 GNN 权重: {encoder_state_path}")
    else:
        raise FileNotFoundError(
            f"缺少原始视图对应的冻结 GNN 权重: {encoder_state_path}。"
            "为防止原始/增强视图来自不同编码器，禁止单独生成 edge-drop；"
            "请先运行原始 GAT 编码阶段。"
        )

    stats = {
        'total_files': len(subgraph_files),
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'output_files': [],
        'total_size_mb': 0.0,
        'edge_drop_rate': float(edge_drop_rate),
        'edge_drop_seed': int(edge_drop_seed),
        'encoder_state_path': encoder_state_path if os.path.exists(encoder_state_path) else None,
        'graph_encoder_type': encoder.graph_encoder_type,
        'gnn_layers': encoder.gnn_layers,
        'num_heads': encoder.num_heads,
        'sage_aggr': encoder.sage_aggr
    }

    for file_idx, subgraph_file in enumerate(tqdm(subgraph_files, desc="Edge-drop GNN编码")):
        try:
            node_id = extract_node_id_from_filename(subgraph_file)

            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data:
                stats['failed'] += 1
                stats['failed_files'].append(subgraph_file)
                continue

            result = process_subgraph_data_with_pretrained_gat(
                subgraph_data, encoder,
                edge_drop_rate=edge_drop_rate,
                edge_drop_seed=edge_drop_seed,
                file_seed=file_idx,
            )
            if not result or not result.get('temporal_node_embeddings'):
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

    stats_file = os.path.join(output_dir, stats_filename)
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
            "name": "main",
            "input_dir": os.path.join(WORKSPACE_ROOT, "twitter_subgraphs"),
            "text_dir": os.path.join(WORKSPACE_ROOT, "sbert_encoded_texts"),
            "output_dir": os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs"),
            "pretrained_encoder_state_path": None,
        },
        {
            "name": "normal_6000",
            "input_dir": os.path.join(WORKSPACE_ROOT, "twitter_subgraphs_normal_6000"),
            "text_dir": os.path.join(WORKSPACE_ROOT, "sbert_encoded_texts_normal_6000"),
            "output_dir": os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs_normal_6000"),
            # normal_6000 不再单独预训练；与主数据严格共享同一个 GAT 坐标系。
            "pretrained_encoder_state_path": os.path.join(
                ARTIFACT_ROOT, "gat_encoded_subgraphs", "gnn_encoder_state.pth"
            ),
        }
    ]

    # 固定复用现有子图和文本账号集合；这里只编码，不重新抽取任何用户。
    for task in tasks:
        task["node_ids"] = collect_subgraph_ids(task["input_dir"])
        text_ids = collect_encoded_ids(task["text_dir"], "text_encoded")
        assert_exact_id_alignment(
            f"{task['name']}_subgraphs", task["node_ids"],
            f"{task['name']}_texts", text_ids,
        )
        print(f"[预检通过] {task['name']}: 固定账号数={len(task['node_ids'])}，子图与文本严格对应")

    for task in tasks:
        print("\n" + "-" * 60)
        print(f"开始编码: {task['input_dir']} → {task['output_dir']}")
        print("-" * 60)

        stats = batch_encode_subgraphs_experimental(
            input_dir=task["input_dir"],
            output_dir=task["output_dir"],
            config=config,
            pretrained_encoder_state_path=task["pretrained_encoder_state_path"],
        )

        if not stats:
            print(f"  编码失败: {task['input_dir']}")
            continue

        output_ids = collect_encoded_ids(task["output_dir"], "gat_encoded")
        assert_exact_id_alignment(
            f"{task['name']}_subgraphs", task["node_ids"],
            f"{task['name']}_gat", output_ids,
        )

        print(f"  编码完成: {task['input_dir']}")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  成功编码: {stats['successful']}")
        print(f"  编码失败: {stats['failed']}")
        print(f"  预训练损失: {stats['pretrain_loss']:.4f}")
        print(f"  输出大小: {stats['total_size_mb']:.2f} MB")

    # edge-dropout 增强视图（L_cont 第二视角）：两个来源使用独立目录，
    # 防止重叠账号互相覆盖；各自复用对应原始编码的冻结 GNN 权重。
    edge_drop_rate = 0.2
    edge_drop_seed = 42
    edge_drop_tasks = [
        {
            "name": "main",
            "input_dir": tasks[0]["input_dir"],
            "original_output_dir": tasks[0]["output_dir"],
            "output_dir": os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs_edge_drop"),
        },
        {
            "name": "normal_6000",
            "input_dir": tasks[1]["input_dir"],
            "original_output_dir": tasks[1]["output_dir"],
            "output_dir": os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs_normal_6000_edge_drop"),
        },
    ]

    for task in edge_drop_tasks:
        print("\n" + "-" * 60)
        print(f"开始 edge-drop 编码: {task['input_dir']} → {task['output_dir']} (rate={edge_drop_rate}, seed={edge_drop_seed})")
        print("-" * 60)

        stats = batch_encode_subgraphs_edge_drop(
            input_dir=task["input_dir"],
            output_dir=task["output_dir"],
            original_output_dir=task["original_output_dir"],
            config=config,
            edge_drop_rate=edge_drop_rate,
            edge_drop_seed=edge_drop_seed,
            stats_filename=f"edge_drop_encoding_stats_{os.path.basename(task['input_dir'])}.pkl",
        )

        if not stats:
            print(f"  edge-drop 编码失败: {task['input_dir']}")
            continue

        print(f"  edge-drop 编码完成: {task['input_dir']}")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  成功编码: {stats['successful']}")
        print(f"  编码失败: {stats['failed']}")
        print(f"  输出大小: {stats['total_size_mb']:.2f} MB")

    for source_task, edge_task in zip(tasks, edge_drop_tasks):
        actual_edge_ids = collect_encoded_ids(edge_task["output_dir"], "gat_encoded")
        assert_exact_id_alignment(
            f"{source_task['name']}_expected_edge",
            source_task["node_ids"],
            f"{source_task['name']}_edge_drop_gat",
            actual_edge_ids,
        )

    manifest_groups = []
    for task in tasks:
        ids = sorted(task["node_ids"])
        manifest_groups.append({
            "name": task["name"],
            "input_dir": task["input_dir"],
            "text_dir": task["text_dir"],
            "gat_output_dir": task["output_dir"],
            "node_count": len(ids),
            "node_ids": ids,
            "node_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "gnn_encoder_state": os.path.join(task["output_dir"], "gnn_encoder_state.pth"),
            "shared_source_state_path": task["pretrained_encoder_state_path"],
        })
    manifest_path = os.path.join(ARTIFACT_ROOT, "gat_asset_manifest.json")
    write_gat_manifest(
        manifest_path,
        manifest_groups,
        {task["name"]: task["output_dir"] for task in edge_drop_tasks},
    )
    print(f"[最终核验通过] 原始 GAT、edge-drop GAT 与复用文本的账号集合严格对应")
    print(f"资产清单已保存: {manifest_path}")


if __name__ == "__main__":
    main()
