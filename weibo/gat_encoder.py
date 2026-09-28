import os
import re
import glob
import random
import pickle
import concurrent.futures
import multiprocessing as mp
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

warnings.filterwarnings('ignore')


#GNN 编码器模块：GAT / GraphSAGE + 1/2/3 层（用于消融切换）
class SingleLayerGAT(nn.Module):
    """单层 GAT（基线）"""

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
    """双层 GAT（多头注意力 × 2 层）"""

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
    """三层 GAT（多头注意力 × 3 层）"""

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
    """单层 GraphSAGE（无注意力消融）"""

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
    """双层 GraphSAGE（与 GAT 两层消融对齐）"""

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
    """链路预测头（用于自监督预训练）"""

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
        if edge_index is None or edge_index.numel() == 0 or edge_index.size(1) == 0:
            return torch.empty((0,), device=node_embeddings.device, dtype=torch.float)

        src = node_embeddings[edge_index[0]]
        dst = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([src, dst], dim=1)
        prob = self.predictor(edge_emb).squeeze()
        return prob


# 时序图编码器封装：统一 graph_encoder_type 切换
class ExperimentalTemporalGATEncoder(nn.Module):
    """
    兼容保留原类名，但内部支持 graph_encoder_type={gat,sage}
    - 仅输出节点级嵌入（不做图级池化、不做 out/in 拼接）
    - 预训练仍使用链路预测
    """

    def __init__(self,
                 input_dim: int = 15,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 device: object = 'auto',
                 graph_encoder_type: str = "gat",   # "gat" or "sage"
                 gnn_layers: int = 1,               # GAT: 1/2/3；SAGE: 1/2
                 gat_layers: Optional[int] = None,  
                 sage_aggr: str = "mean"):
        super().__init__()

        self.device = self._normalize_device(device)

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.graph_encoder_type = str(graph_encoder_type).lower().strip()
        if gat_layers is not None:
            gnn_layers = int(gat_layers)
        self.gnn_layers = int(gnn_layers)
        self.sage_aggr = str(sage_aggr)

        self.gnn = self._build_gnn()
        self.link_predictor = LinkPredictor(self.hidden_dim)

        self.gnn_frozen = False
        self.to(self.device)

    @staticmethod
    def _normalize_device(device: object) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, torch.device):
            return device
        return torch.device(str(device))

    def _build_gnn(self) -> nn.Module:
        if self.graph_encoder_type not in {"gat", "sage"}:
            raise ValueError(f"graph_encoder_type 必须为 'gat' 或 'sage'，当前为 {self.graph_encoder_type}")

        if self.graph_encoder_type == "gat":
            if self.gnn_layers not in {1, 2, 3}:
                raise ValueError(f"GAT 模式下 gnn_layers 只能为 1/2/3，当前为 {self.gnn_layers}")
            if self.gnn_layers == 1:
                return SingleLayerGAT(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)
            if self.gnn_layers == 2:
                return TwoLayerGAT(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)
            return ThreeLayerGAT(self.input_dim, self.hidden_dim, self.num_heads, self.dropout)

        if self.gnn_layers not in {1, 2}:
            raise ValueError(f"SAGE 模式下 gnn_layers 只能为 1/2，当前为 {self.gnn_layers}")
        if self.gnn_layers == 1:
            return SingleLayerSAGE(self.input_dim, self.hidden_dim, self.dropout, aggr=self.sage_aggr)
        return TwoLayerSAGE(self.input_dim, self.hidden_dim, self.dropout, aggr=self.sage_aggr)

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

    def create_node_features(self, subgraph_data: Data) -> torch.Tensor:
        num_nodes = subgraph_data.num_nodes

        if hasattr(subgraph_data, 'x') and subgraph_data.x is not None:
            node_features = subgraph_data.x
            if node_features.size(1) != self.input_dim:
                if node_features.size(1) < self.input_dim:
                    pad_dim = self.input_dim - node_features.size(1)
                    padding = torch.zeros((num_nodes, pad_dim), device=self.device, dtype=node_features.dtype)
                    node_features = torch.cat([node_features.to(self.device), padding], dim=1)
                else:
                    node_features = node_features[:, :self.input_dim].to(self.device)
            else:
                node_features = node_features.to(self.device)
        else:
            node_features = torch.randn(num_nodes, self.input_dim, device=self.device) * 0.1

        return node_features

    @staticmethod
    def reverse_edges_for_out_subgraph(edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return edge_index
        return torch.stack([edge_index[1], edge_index[0]], dim=0)

    @staticmethod
    def drop_edge(edge_index: torch.Tensor, edge_dropout_p: float = 0.0) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0 or edge_index.size(1) == 0:
            return edge_index
        p = float(edge_dropout_p)
        if p <= 0.0:
            return edge_index
        if p >= 1.0:
            p = 0.999

        num_edges = edge_index.size(1)
        keep = torch.rand(num_edges, device=edge_index.device) >= p
        if not bool(keep.any()):
            keep[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True
        return edge_index[:, keep]

    def apply_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index is None or edge_index.numel() == 0:
            idx = torch.arange(num_nodes, device=x.device, dtype=torch.long)
            edge_index = torch.stack([idx, idx], dim=0)
        return self.gnn(x, edge_index)

    def encode_subgraph_with_experimental_design(self,
                                                 subgraph_data: Data,
                                                 is_out_subgraph: bool = False,
                                                 edge_dropout_p: float = 0.0) -> torch.Tensor:
        if subgraph_data is None or subgraph_data.num_nodes == 0:
            return torch.zeros((0, self.hidden_dim), device=self.device)

        self.eval()
        with torch.no_grad():
            subgraph_data = subgraph_data.to(self.device)
            node_features = self.create_node_features(subgraph_data)

            if is_out_subgraph and subgraph_data.edge_index.numel() > 0:
                edge_index = self.reverse_edges_for_out_subgraph(subgraph_data.edge_index)
            else:
                edge_index = subgraph_data.edge_index
            edge_index = self.drop_edge(edge_index, edge_dropout_p=edge_dropout_p)

            return self.apply_gnn(node_features, edge_index)

    def encode_temporal_subgraphs(self,
                                 temporal_subgraphs: Dict[int, Tuple[Optional[Data], Optional[Data]]],
                                 edge_dropout_p: float = 0.0,
                                 ) -> Dict[int, Dict[str, torch.Tensor]]:
        self.eval()
        time_steps = sorted(temporal_subgraphs.keys())
        temporal_node_embeddings: Dict[int, Dict[str, torch.Tensor]] = {}

        with torch.no_grad():
            for t in time_steps:
                out_sg, in_sg = temporal_subgraphs[t]
                out_repr = self.encode_subgraph_with_experimental_design(out_sg, is_out_subgraph=True, edge_dropout_p=edge_dropout_p)
                in_repr = self.encode_subgraph_with_experimental_design(in_sg, is_out_subgraph=False, edge_dropout_p=edge_dropout_p)

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


def _encode_chunk_worker(payload) -> Dict:
    """Encode an independent file chunk in a forked CPU worker."""
    (
        subgraph_files,
        encoder_config,
        checkpoint_path,
        output_dir,
        edge_drop_output_dir,
        edge_dropout_p,
        edge_dropout_seed,
        worker_id,
    ) = payload

    worker_config = dict(encoder_config)
    worker_config['device'] = torch.device('cpu')
    encoder = ExperimentalTemporalGATEncoder(**worker_config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('encoder_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    encoder.load_state_dict(state_dict, strict=False)
    encoder.freeze_gnn_parameters()

    seed = int(edge_dropout_seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    stats = {
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'output_files': [],
        'total_size_mb': 0.0,
        'edge_drop_successful': 0,
        'edge_drop_failed': 0,
        'edge_drop_output_files': [],
    }

    for subgraph_file in subgraph_files:
        try:
            node_id = extract_node_id_from_filename(subgraph_file)
            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data:
                raise RuntimeError('empty subgraph payload')

            result = process_subgraph_data_with_pretrained_gat(subgraph_data, encoder, edge_dropout_p=0.0)
            if not result:
                raise RuntimeError('empty encoder result')

            output_filename = f"subgraph_{node_id}_k2_gat_encoded.pkl"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            stats['successful'] += 1
            stats['output_files'].append(output_path)
            stats['total_size_mb'] += os.path.getsize(output_path) / 1024 / 1024

            if edge_drop_output_dir and float(edge_dropout_p) > 0.0:
                edge_result = process_subgraph_data_with_pretrained_gat(
                    subgraph_data, encoder, edge_dropout_p=float(edge_dropout_p)
                )
                if edge_result:
                    edge_output_path = os.path.join(edge_drop_output_dir, output_filename)
                    with open(edge_output_path, 'wb') as f:
                        pickle.dump(edge_result, f)
                    stats['edge_drop_successful'] += 1
                    stats['edge_drop_output_files'].append(edge_output_path)
                else:
                    stats['edge_drop_failed'] += 1
        except Exception:
            stats['failed'] += 1
            stats['failed_files'].append(subgraph_file)

    return stats


def pretrain_gat_with_link_prediction(subgraph_files: List[str],
                                      encoder: ExperimentalTemporalGATEncoder,
                                      num_epochs: int = 10,
                                      lr: float = 0.001,
                                      max_samples: int = 100, 
                                      patience: int = 20) -> float:
    
    if encoder.gnn_frozen:
        encoder.unfreeze_gnn_parameters()

    encoder.train()
    optimizer = torch.optim.Adam(encoder.gnn.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    valid_subgraphs = []
    sample_limit = int(max_samples) if max_samples is not None and int(max_samples) > 0 else None
    for subgraph_file in subgraph_files:
        if sample_limit is not None and len(valid_subgraphs) >= sample_limit:
            break
        try:
            subgraph_data = load_subgraph_data(subgraph_file)
            if (not subgraph_data) or ('temporal_subgraphs' not in subgraph_data):
                continue

            temporal_subgraphs = subgraph_data['temporal_subgraphs']
            for _, (out_sg, in_sg) in temporal_subgraphs.items():
                for sg in (out_sg, in_sg):
                    if sample_limit is not None and len(valid_subgraphs) >= sample_limit:
                        break
                    if sg is None:
                        continue
                    if (hasattr(sg, 'edge_index') and sg.edge_index.numel() > 0 and getattr(sg, 'num_nodes', 0) > 1):
                        valid_subgraphs.append(sg)
                if sample_limit is not None and len(valid_subgraphs) >= sample_limit:
                    break
        except Exception:
            continue

    if not valid_subgraphs:
        return float('inf')

    # The caller intentionally bounds pretraining to a deterministic subset.
    # Without this slice, the max_samples argument is silently ignored and
    # every temporal graph is revisited for every pretraining epoch.
    if max_samples is not None and int(max_samples) > 0:
        valid_subgraphs = valid_subgraphs[:int(max_samples)]

    print(
        f"开始预训练 GNN：type={encoder.graph_encoder_type}, layers={encoder.gnn_layers}, "
        f"heads={encoder.num_heads}，有效子图数={len(valid_subgraphs)}"
    )

    best_loss = float('inf')
    epochs_no_improve = 0
    final_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="预训练轮次"):
        epoch_losses = []
        random.shuffle(valid_subgraphs)

        for sg in tqdm(valid_subgraphs, desc=f"第 {epoch + 1} 轮", leave=False):
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
            print(f"触发早停：epoch={epoch + 1}")
            break

        tqdm.write(f"epoch {epoch + 1}/{num_epochs} - 平均损失: {avg_loss:.4f}")

    return final_loss


def process_subgraph_data_with_pretrained_gat(subgraph_data: Dict,
                                             encoder: ExperimentalTemporalGATEncoder,
                                             edge_dropout_p: float = 0.0) -> Dict:
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
                'edge_dropout_p': float(edge_dropout_p),
                'out_in_concat': False,
                'graph_level_pooling_in_this_file': False
            }
        }

    try:
        if not encoder.gnn_frozen:
            encoder.freeze_gnn_parameters()

        temporal_node_embeddings = encoder.encode_temporal_subgraphs(
            temporal_subgraphs,
            edge_dropout_p=edge_dropout_p,
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
                'edge_dropout_p': float(edge_dropout_p),
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
                                       edge_drop_output_dir: Optional[str] = None,
                                       edge_dropout_p: float = 0.0,
                                       edge_dropout_seed: int = 20260911,
                                       encoder_checkpoint_path: Optional[str] = None,
                                       load_encoder_checkpoint: bool = False,
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
    if edge_drop_output_dir:
        os.makedirs(edge_drop_output_dir, exist_ok=True)

    subgraph_files = find_subgraph_files(input_dir)
    if not subgraph_files:
        return {}

    encoder = ExperimentalTemporalGATEncoder(**config)

    if encoder_checkpoint_path is None:
        encoder_checkpoint_path = os.path.join(output_dir, "gnn_encoder.pt")

    pretrain_loss = None
    if load_encoder_checkpoint:
        if not os.path.exists(encoder_checkpoint_path):
            raise FileNotFoundError(f"GNN encoder checkpoint not found: {encoder_checkpoint_path}")
        checkpoint = torch.load(encoder_checkpoint_path, map_location=encoder.device, weights_only=False)
        state_dict = checkpoint.get('encoder_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        encoder.load_state_dict(state_dict, strict=False)
        print(f"加载已预训练 GNN encoder: {encoder_checkpoint_path}")
    else:
        pretrain_loss = pretrain_gat_with_link_prediction(
            subgraph_files, encoder,
            num_epochs=20, lr=0.001, max_samples=100
        )
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': config,
            'pretrain_loss': pretrain_loss,
        }, encoder_checkpoint_path)
        print(f"保存 GNN encoder checkpoint: {encoder_checkpoint_path}")

    encoder.freeze_gnn_parameters()
    random.seed(int(edge_dropout_seed))
    np.random.seed(int(edge_dropout_seed))
    torch.manual_seed(int(edge_dropout_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(edge_dropout_seed))

    stats = {
        'total_files': len(subgraph_files),
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'output_files': [],
        'total_size_mb': 0.0,
        'pretrain_loss': pretrain_loss,
        'encoder_checkpoint_path': encoder_checkpoint_path,
        'edge_drop_output_dir': edge_drop_output_dir,
        'edge_dropout_p': float(edge_dropout_p),
        'edge_dropout_seed': int(edge_dropout_seed),
        'edge_drop_successful': 0,
        'edge_drop_failed': 0,
        'edge_drop_output_files': [],
        'graph_encoder_type': encoder.graph_encoder_type,
        'gnn_layers': encoder.gnn_layers,
        'num_heads': encoder.num_heads,
        'sage_aggr': encoder.sage_aggr
    }

    def _encode_one(subgraph_file: str) -> Dict:
        result_stats = {
            'successful': 0,
            'failed': 0,
            'failed_files': [],
            'output_files': [],
            'total_size_mb': 0.0,
            'edge_drop_successful': 0,
            'edge_drop_failed': 0,
            'edge_drop_output_files': [],
        }
        try:
            node_id = extract_node_id_from_filename(subgraph_file)
            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data:
                raise RuntimeError("empty subgraph payload")

            result = process_subgraph_data_with_pretrained_gat(
                subgraph_data, encoder, edge_dropout_p=0.0
            )
            if not result:
                raise RuntimeError("empty encoder result")

            output_filename = f"subgraph_{node_id}_k2_gat_encoded.pkl"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)

            result_stats['successful'] = 1
            result_stats['output_files'].append(output_path)
            result_stats['total_size_mb'] = os.path.getsize(output_path) / 1024 / 1024

            if edge_drop_output_dir and float(edge_dropout_p) > 0.0:
                edge_result = process_subgraph_data_with_pretrained_gat(
                    subgraph_data, encoder, edge_dropout_p=float(edge_dropout_p)
                )
                if edge_result:
                    edge_output_path = os.path.join(edge_drop_output_dir, output_filename)
                    with open(edge_output_path, 'wb') as f:
                        pickle.dump(edge_result, f)
                    result_stats['edge_drop_successful'] = 1
                    result_stats['edge_drop_output_files'].append(edge_output_path)
                else:
                    result_stats['edge_drop_failed'] = 1
        except Exception:
            result_stats['failed'] = 1
            result_stats['failed_files'].append(subgraph_file)
        return result_stats

    num_workers = max(1, int(os.environ.get("GAT_ENCODE_WORKERS", "1")))
    if num_workers > 1:
        torch.set_num_threads(1)
        chunks = [subgraph_files[i::num_workers] for i in range(num_workers)]
        payloads = [
            (
                chunk,
                config,
                encoder_checkpoint_path,
                output_dir,
                edge_drop_output_dir,
                float(edge_dropout_p),
                int(edge_dropout_seed),
                worker_id,
            )
            for worker_id, chunk in enumerate(chunks)
            if chunk
        ]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(payloads)) as pool:
            for item in tqdm(
                pool.imap_unordered(_encode_chunk_worker, payloads),
                total=len(payloads),
                desc=f"批量编码({len(payloads)}进程)",
            ):
                stats['successful'] += item['successful']
                stats['failed'] += item['failed']
                stats['failed_files'].extend(item['failed_files'])
                stats['output_files'].extend(item['output_files'])
                stats['total_size_mb'] += item['total_size_mb']
                stats['edge_drop_successful'] += item['edge_drop_successful']
                stats['edge_drop_failed'] += item['edge_drop_failed']
                stats['edge_drop_output_files'].extend(item['edge_drop_output_files'])
    else:
        for subgraph_file in tqdm(subgraph_files, desc="批量编码"):
            item = _encode_one(subgraph_file)
            stats['successful'] += item['successful']
            stats['failed'] += item['failed']
            stats['failed_files'].extend(item['failed_files'])
            stats['output_files'].extend(item['output_files'])
            stats['total_size_mb'] += item['total_size_mb']
            stats['edge_drop_successful'] += item['edge_drop_successful']
            stats['edge_drop_failed'] += item['edge_drop_failed']
            stats['edge_drop_output_files'].extend(item['edge_drop_output_files'])

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

    input_dir = os.environ.get("SUBGRAPH_INPUT_DIR", "subgraphs")
    output_dir = os.environ.get("GAT_OUTPUT_DIR", "gat_encoded_subgraphs")
    edge_drop_output_dir = os.environ.get("EDGE_DROP_GAT_OUTPUT_DIR", "gat_encoded_subgraphs_edge_drop")
    edge_dropout_p = float(os.environ.get("EDGE_DROPOUT_P", "0.2"))
    edge_dropout_seed = int(os.environ.get("EDGE_DROPOUT_SEED", "20260911"))
    encoder_checkpoint_path = os.environ.get("GAT_ENCODER_CHECKPOINT", os.path.join(output_dir, "gnn_encoder.pt"))
    load_encoder_checkpoint = os.environ.get("GAT_LOAD_CHECKPOINT", "0") == "1"

    print(f"输入目录: {input_dir}")
    print(f"原始 GAT 输出目录: {output_dir}")
    print(f"edge-drop GAT 输出目录: {edge_drop_output_dir}")
    print(f"edge_dropout_p: {edge_dropout_p}")
    print(f"GNN checkpoint: {encoder_checkpoint_path}, load={load_encoder_checkpoint}")

    stats = batch_encode_subgraphs_experimental(
        input_dir=input_dir,
        output_dir=output_dir,
        edge_drop_output_dir=edge_drop_output_dir,
        edge_dropout_p=edge_dropout_p,
        edge_dropout_seed=edge_dropout_seed,
        encoder_checkpoint_path=encoder_checkpoint_path,
        load_encoder_checkpoint=load_encoder_checkpoint,
        config=config,
    )

    if not stats:
        print("批量编码失败")
        return

    print("\n编码统计：")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功编码: {stats['successful']}")
    print(f"编码失败: {stats['failed']}")
    if stats['pretrain_loss'] is None:
        print("预训练损失: 复用已有 GNN checkpoint")
    else:
        print(f"预训练损失: {stats['pretrain_loss']:.4f}")
    print(f"edge-drop 成功编码: {stats.get('edge_drop_successful', 0)}")
    print(f"输出大小: {stats['total_size_mb']:.2f} MB")
    print(
        f"type={stats['graph_encoder_type']}, layers={stats['gnn_layers']}, "
        f"heads={stats['num_heads']}, sage_aggr={stats['sage_aggr']}"
    )


if __name__ == "__main__":
    main()
