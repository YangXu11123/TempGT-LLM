import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List
from utils import load_paired_embeddings, get_user_label, extract_node_id  
import pickle


class TemporalGraphTextDataset(Dataset):
    """时序图-文本数据集（节点级表示）"""

    def __init__(
        self,
        node_data: Dict[str, Dict],
        labels: Dict[str, int],
        max_seq_len: int = 100,
        is_training: bool = True,
        struct_node_dim: int = 128,
        text_dim: int = 384,
    ):
        """
        参数:
            node_data: 节点ID到特征字典的映射
            labels: 节点ID到标签的映射
            max_seq_len: 最大时间步长度
            is_training: 是否为训练模式
            struct_node_dim: GAT节点级嵌入维度
            text_dim: SBERT 文本嵌入维度
        """
        self.node_data = node_data
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.is_training = is_training
        self.struct_node_dim = struct_node_dim
        self.text_dim = text_dim

        # 有效节点：同时在 node_data 和 labels 中存在
        self.valid_node_ids: List[str] = [
            nid for nid in node_data.keys() if nid in labels
        ]

        print(f"数据集初始化: {len(self.valid_node_ids)} 个有效节点（节点级表示）")

    def __len__(self):
        return len(self.valid_node_ids)

    def _to_tensor(self, x, dim: int):
        """
        辅助函数：把 numpy / list / Tensor 统一转为 2D Tensor [N, dim]，
        如果为空则返回 [0, dim]。
        """
        if x is None:
            return torch.empty(0, dim, dtype=torch.float32)

        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x, dtype=torch.float32)

        if t.numel() == 0:
            return torch.empty(0, dim, dtype=torch.float32)

        if t.dim() == 1:
            # [dim] -> [1, dim]
            t = t.unsqueeze(0)

        # 保证最后一维是 dim，必要时裁剪或 padding
        if t.size(-1) < dim:
            pad = torch.zeros(
                t.size(0), dim - t.size(-1), dtype=t.dtype, device=t.device
            )
            t = torch.cat([t, pad], dim=-1)
        elif t.size(-1) > dim:
            t = t[..., :dim]

        return t

    def __getitem__(self, idx):
        node_id = self.valid_node_ids[idx]

        data = self.node_data[node_id]
        label = self.labels[node_id]

        temporal_node_embeddings = data.get("temporal_node_embeddings", {})
        temporal_text_embeddings = data.get("temporal_text_embeddings", {})
        timesteps = data.get("timesteps", [])

        # 处理 timesteps
        if isinstance(timesteps, torch.Tensor):
            all_ts = timesteps.long().tolist()
        else:
            all_ts = list(timesteps)

        # 排序时间步，确保时间顺序一致
        all_ts = sorted(all_ts)
        seq_len = len(all_ts)

        if seq_len == 0:
            # 极端情况：该节点没有任何时间步，直接返回空结构
            return {
                "node_id": node_id,
                "struct_in_list": [],
                "struct_out_list": [],
                "text_list": [],
                "timesteps": torch.empty(0, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long),
                "seq_len": torch.tensor(0, dtype=torch.long),
                "struct_node_dim": self.struct_node_dim,
                "text_dim": self.text_dim,
            }

        # 截断 / 选择时间步窗口
        if seq_len > self.max_seq_len:
            if self.is_training:
                # 训练时随机截一段连续窗口
                start_idx = np.random.randint(0, seq_len - self.max_seq_len + 1)
                selected_ts = all_ts[start_idx : start_idx + self.max_seq_len]
            else:
                # 验证 / 测试固定取前 max_seq_len 个
                selected_ts = all_ts[: self.max_seq_len]
        else:
            selected_ts = all_ts

        L = len(selected_ts)

        struct_in_list: List[torch.Tensor] = []
        struct_out_list: List[torch.Tensor] = []
        text_list: List[torch.Tensor] = []

        for t in selected_ts:
            # 结构侧
            node_dict = temporal_node_embeddings.get(int(t), {})

            in_emb = node_dict.get("in_node_embeddings", None)
            out_emb = node_dict.get("out_node_embeddings", None)

            in_emb = self._to_tensor(in_emb, self.struct_node_dim)
            out_emb = self._to_tensor(out_emb, self.struct_node_dim)

            

            struct_in_list.append(in_emb)
            struct_out_list.append(out_emb)

            # 文本侧
            txt_emb = temporal_text_embeddings.get(int(t), None)
            txt_emb = self._to_tensor(txt_emb, self.text_dim)
            text_list.append(txt_emb)

        return {
            "node_id": node_id,
            "struct_in_list": struct_in_list,   # List[T']: Tensor [N_in_t,  struct_node_dim]
            "struct_out_list": struct_out_list, # List[T']: Tensor [N_out_t, struct_node_dim]
            "text_list": text_list,             # List[T']: Tensor [N_txt_t, text_dim]
            "timesteps": torch.tensor(selected_ts, dtype=torch.long),  # [T']
            "label": torch.tensor(label, dtype=torch.long),
            "seq_len": torch.tensor(L, dtype=torch.long),
            "struct_node_dim": self.struct_node_dim,
            "text_dim": self.text_dim,
        }


def collate_fn(batch):
    """批处理函数：将节点级列表 padding 成 4D Tensor + mask"""
    batch_size = len(batch)

    if batch_size == 0:
        return {}

    # 维度信息
    struct_node_dim = batch[0]["struct_node_dim"]
    text_dim = batch[0]["text_dim"]

    # 时间维度最大长度 
    T_max = max(item["timesteps"].shape[0] for item in batch)

    # 节点数的最大值
    max_N_in = 0
    max_N_out = 0
    max_N_txt = 0

    for item in batch:
        struct_in_list = item["struct_in_list"]
        struct_out_list = item["struct_out_list"]
        text_list = item["text_list"]

        for emb in struct_in_list:
            if emb is not None:
                max_N_in = max(max_N_in, emb.shape[0])
        for emb in struct_out_list:
            if emb is not None:
                max_N_out = max(max_N_out, emb.shape[0])
        for emb in text_list:
            if emb is not None:
                max_N_txt = max(max_N_txt, emb.shape[0])

    # 避免 0 造成全 0 尺寸
    if max_N_in == 0:
        max_N_in = 1
    if max_N_out == 0:
        max_N_out = 1
    if max_N_txt == 0:
        max_N_txt = 1

    # 初始化批级张量
    struct_in_embeddings = torch.zeros(
        batch_size, T_max, max_N_in, struct_node_dim, dtype=torch.float32
    )
    struct_out_embeddings = torch.zeros(
        batch_size, T_max, max_N_out, struct_node_dim, dtype=torch.float32
    )
    text_node_embeddings = torch.zeros(
        batch_size, T_max, max_N_txt, text_dim, dtype=torch.float32
    )

    struct_in_mask = torch.zeros(
        batch_size, T_max, max_N_in, dtype=torch.bool
    )
    struct_out_mask = torch.zeros(
        batch_size, T_max, max_N_out, dtype=torch.bool
    )
    text_mask = torch.zeros(
        batch_size, T_max, max_N_txt, dtype=torch.bool
    )

    timesteps = torch.zeros(batch_size, T_max, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, T_max, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    node_ids: List[str] = []

    # 填充每个样本
    for i, item in enumerate(batch):
        ts = item["timesteps"]          # [T_i]
        L_i = ts.shape[0]
        L_use = min(L_i, T_max)

        struct_in_list = item["struct_in_list"]
        struct_out_list = item["struct_out_list"]
        text_list = item["text_list"]

        for t_idx in range(L_use):
            in_emb = struct_in_list[t_idx]   # [N_in_t,  D_s]
            out_emb = struct_out_list[t_idx] # [N_out_t, D_s]
            txt_emb = text_list[t_idx]       # [N_txt_t, D_t]

            n_in = in_emb.shape[0]
            n_out = out_emb.shape[0]
            n_txt = txt_emb.shape[0]

            if n_in > 0:
                struct_in_embeddings[i, t_idx, :n_in, :] = in_emb
                struct_in_mask[i, t_idx, :n_in] = True

            if n_out > 0:
                struct_out_embeddings[i, t_idx, :n_out, :] = out_emb
                struct_out_mask[i, t_idx, :n_out] = True

            if n_txt > 0:
                text_node_embeddings[i, t_idx, :n_txt, :] = txt_emb
                text_mask[i, t_idx, :n_txt] = True

            timesteps[i, t_idx] = ts[t_idx]
            attention_mask[i, t_idx] = 1.0

        labels[i] = item["label"]
        seq_lens[i] = L_use
        node_ids.append(item["node_id"])

    return {
        "struct_in_embeddings": struct_in_embeddings,    # [B, T_max, max_N_in,  D_s]
        "struct_out_embeddings": struct_out_embeddings,  # [B, T_max, max_N_out, D_s]
        "text_node_embeddings": text_node_embeddings,    # [B, T_max, max_N_txt, D_t]
        "struct_in_mask": struct_in_mask,                # [B, T_max, max_N_in]
        "struct_out_mask": struct_out_mask,              # [B, T_max, max_N_out]
        "text_mask": text_mask,                          # [B, T_max, max_N_txt]
        "timesteps": timesteps,                          # [B, T_max]
        "attention_mask": attention_mask,                # [B, T_max]
        "labels": labels,                                # [B]
        "node_ids": node_ids,
        "seq_lens": seq_lens,                            # [B]
    }


def create_data_loaders(
    train_node_ids: List[str],
    val_node_ids: List[str],
    node_data: Dict,
    labels: Dict,
    config,
):
    """创建训练和验证数据加载器（节点级表示）"""
    struct_node_dim = config.structure_dim // 2  
    text_dim = config.text_dim                 

    # 训练集
    train_dataset = TemporalGraphTextDataset(
        {nid: node_data[nid] for nid in train_node_ids if nid in node_data},
        {nid: labels[nid] for nid in train_node_ids if nid in labels},
        max_seq_len=config.max_sequence_length,
        is_training=True,
        struct_node_dim=struct_node_dim,
        text_dim=text_dim,
    )

    # 验证集
    val_dataset = TemporalGraphTextDataset(
        {nid: node_data[nid] for nid in val_node_ids if nid in node_data},
        {nid: labels[nid] for nid in val_node_ids if nid in labels},
        max_seq_len=config.max_sequence_length,
        is_training=False,
        struct_node_dim=struct_node_dim,
        text_dim=text_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset
