import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from typing import Dict, Iterator, List, Optional


class TemporalGraphTextDataset(Dataset):
    """
    时序图-文本数据集

    node_data[nid] 预期结构（由前面的 GAT / SBERT 代码给出）：
        {
            'temporal_node_embeddings': {
                t1: {
                    'in_node_embeddings':  Tensor [N_in_t1, 128],
                    'out_node_embeddings': Tensor [N_out_t1,128],
                },
                t2: { ... },
                ...
            },
            'temporal_text_embeddings': {
                t1: Tensor [N_txt_t1, 384],
                t2: Tensor [N_txt_t2, 384],
                ...
            },
            'timesteps': Tensor [T]  或 list[int] 
        }

    本 Dataset 不做池化，只负责：
      - 根据 max_seq_len 选择时间步子序列
      - 把各时间步对应的节点级嵌入打包成列表
    """

    def __init__(
        self,
        node_data: Dict[str, Dict],
        labels: Dict[str, int],
        max_seq_len: int = 100,
        is_training: bool = True,
        struct_node_dim: int = 128,
        text_dim: int = 384,
    ):
       
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
            t = t.unsqueeze(0)

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
        temporal_node_embeddings_aug = data.get("temporal_node_embeddings_aug", None)
        temporal_text_embeddings = data.get("temporal_text_embeddings", {})
        timesteps = data.get("timesteps", [])

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

        if seq_len > self.max_seq_len:
            if self.is_training:
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
        struct_in_aug_list: List[torch.Tensor] = []
        struct_out_aug_list: List[torch.Tensor] = []
        text_list: List[torch.Tensor] = []

        for t in selected_ts:
            node_dict = temporal_node_embeddings.get(int(t), {})

            in_emb = node_dict.get("in_node_embeddings", None)
            out_emb = node_dict.get("out_node_embeddings", None)

            in_emb = self._to_tensor(in_emb, self.struct_node_dim)
            out_emb = self._to_tensor(out_emb, self.struct_node_dim)

            struct_in_list.append(in_emb)
            struct_out_list.append(out_emb)

            if isinstance(temporal_node_embeddings_aug, dict):
                aug_node_dict = temporal_node_embeddings_aug.get(int(t), {})
                in_aug = self._to_tensor(aug_node_dict.get("in_node_embeddings", None), self.struct_node_dim)
                out_aug = self._to_tensor(aug_node_dict.get("out_node_embeddings", None), self.struct_node_dim)
            else:
                in_aug = torch.empty(0, self.struct_node_dim, dtype=torch.float32)
                out_aug = torch.empty(0, self.struct_node_dim, dtype=torch.float32)

            struct_in_aug_list.append(in_aug)
            struct_out_aug_list.append(out_aug)

            txt_emb = temporal_text_embeddings.get(int(t), None)
            txt_emb = self._to_tensor(txt_emb, self.text_dim)
            text_list.append(txt_emb)

        return {
            "node_id": node_id,
            "struct_in_list": struct_in_list,   
            "struct_out_list": struct_out_list, 
            "struct_in_aug_list": struct_in_aug_list,
            "struct_out_aug_list": struct_out_aug_list,
            "has_edge_drop_aug": isinstance(temporal_node_embeddings_aug, dict),
            "text_list": text_list,            
            "timesteps": torch.tensor(selected_ts, dtype=torch.long), 
            "label": torch.tensor(label, dtype=torch.long),
            "seq_len": torch.tensor(L, dtype=torch.long),
            "struct_node_dim": self.struct_node_dim,
            "text_dim": self.text_dim,
        }


def collate_fn(batch):
    """
    批处理函数：将节点级列表 padding 成 4D Tensor + mask

    输入 batch 中每个元素的字段：
        'struct_in_list' : List[T_i] of [N_in_t,  D_s]
        'struct_out_list': List[T_i] of [N_out_t, D_s]
        'text_list'      : List[T_i] of [N_txt_t, D_t]
        'timesteps'      : Tensor [T_i]
        'label'          : Tensor []
        'seq_len'        : Tensor []
        'node_id'        : str
        'struct_node_dim': int
        'text_dim'       : int

    输出：
        struct_in_embeddings : [B, T_max, max_N_in,  D_s]
        struct_out_embeddings: [B, T_max, max_N_out, D_s]
        text_node_embeddings : [B, T_max, max_N_txt, D_t]
        struct_in_mask       : [B, T_max, max_N_in]
        struct_out_mask      : [B, T_max, max_N_out]
        text_mask            : [B, T_max, max_N_txt]
        timesteps            : [B, T_max]
        attention_mask       : [B, T_max]
        labels               : [B]
        node_ids             : list[str]
        seq_lens             : [B]
    """
    batch_size = len(batch)

    if batch_size == 0:
        return {}

    # 维度信息（所有样本一致）
    struct_node_dim = batch[0]["struct_node_dim"]
    text_dim = batch[0]["text_dim"]

    # 时间维度最大长度 T_max
    T_max = max(item["timesteps"].shape[0] for item in batch)

    # 节点数的最大值（对 in/out/text 分别统计）
    max_N_in = 0
    max_N_out = 0
    max_N_in_aug = 0
    max_N_out_aug = 0
    max_N_txt = 0
    has_any_edge_aug = any(bool(item.get("has_edge_drop_aug", False)) for item in batch)

    for item in batch:
        struct_in_list = item["struct_in_list"]
        struct_out_list = item["struct_out_list"]
        struct_in_aug_list = item.get("struct_in_aug_list", [])
        struct_out_aug_list = item.get("struct_out_aug_list", [])
        text_list = item["text_list"]

        for emb in struct_in_list:
            if emb is not None:
                max_N_in = max(max_N_in, emb.shape[0])
        for emb in struct_out_list:
            if emb is not None:
                max_N_out = max(max_N_out, emb.shape[0])
        for emb in struct_in_aug_list:
            if emb is not None:
                max_N_in_aug = max(max_N_in_aug, emb.shape[0])
        for emb in struct_out_aug_list:
            if emb is not None:
                max_N_out_aug = max(max_N_out_aug, emb.shape[0])
        for emb in text_list:
            if emb is not None:
                max_N_txt = max(max_N_txt, emb.shape[0])

    # 避免 0 造成全 0 尺寸；如果所有都是空，仍然让维度 > 0，后续 mask 会是 False
    if max_N_in == 0:
        max_N_in = 1
    if max_N_out == 0:
        max_N_out = 1
    if max_N_in_aug == 0:
        max_N_in_aug = 1
    if max_N_out_aug == 0:
        max_N_out_aug = 1
    if max_N_txt == 0:
        max_N_txt = 1

    # 初始化批级张量
    struct_in_embeddings = torch.zeros(
        batch_size, T_max, max_N_in, struct_node_dim, dtype=torch.float32
    )
    struct_out_embeddings = torch.zeros(
        batch_size, T_max, max_N_out, struct_node_dim, dtype=torch.float32
    )
    struct_in_embeddings_aug = torch.zeros(
        batch_size, T_max, max_N_in_aug, struct_node_dim, dtype=torch.float32
    )
    struct_out_embeddings_aug = torch.zeros(
        batch_size, T_max, max_N_out_aug, struct_node_dim, dtype=torch.float32
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
    struct_in_mask_aug = torch.zeros(
        batch_size, T_max, max_N_in_aug, dtype=torch.bool
    )
    struct_out_mask_aug = torch.zeros(
        batch_size, T_max, max_N_out_aug, dtype=torch.bool
    )
    text_mask = torch.zeros(
        batch_size, T_max, max_N_txt, dtype=torch.bool
    )

    timesteps = torch.zeros(batch_size, T_max, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, T_max, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    node_ids: List[str] = []


    for i, item in enumerate(batch):
        ts = item["timesteps"]          
        L_i = ts.shape[0]
        L_use = min(L_i, T_max)

        struct_in_list = item["struct_in_list"]
        struct_out_list = item["struct_out_list"]
        struct_in_aug_list = item.get("struct_in_aug_list", [])
        struct_out_aug_list = item.get("struct_out_aug_list", [])
        text_list = item["text_list"]

        for t_idx in range(L_use):
            in_emb = struct_in_list[t_idx]   
            out_emb = struct_out_list[t_idx] 
            in_aug = struct_in_aug_list[t_idx] if t_idx < len(struct_in_aug_list) else torch.empty(0, struct_node_dim)
            out_aug = struct_out_aug_list[t_idx] if t_idx < len(struct_out_aug_list) else torch.empty(0, struct_node_dim)
            txt_emb = text_list[t_idx]       

            n_in = in_emb.shape[0]
            n_out = out_emb.shape[0]
            n_in_aug = in_aug.shape[0]
            n_out_aug = out_aug.shape[0]
            n_txt = txt_emb.shape[0]

            if n_in > 0:
                struct_in_embeddings[i, t_idx, :n_in, :] = in_emb
                struct_in_mask[i, t_idx, :n_in] = True

            if n_out > 0:
                struct_out_embeddings[i, t_idx, :n_out, :] = out_emb
                struct_out_mask[i, t_idx, :n_out] = True

            if n_in_aug > 0:
                struct_in_embeddings_aug[i, t_idx, :n_in_aug, :] = in_aug
                struct_in_mask_aug[i, t_idx, :n_in_aug] = True

            if n_out_aug > 0:
                struct_out_embeddings_aug[i, t_idx, :n_out_aug, :] = out_aug
                struct_out_mask_aug[i, t_idx, :n_out_aug] = True

            if n_txt > 0:
                text_node_embeddings[i, t_idx, :n_txt, :] = txt_emb
                text_mask[i, t_idx, :n_txt] = True

            timesteps[i, t_idx] = ts[t_idx]
            attention_mask[i, t_idx] = 1.0

        labels[i] = item["label"]
        seq_lens[i] = L_use
        node_ids.append(item["node_id"])

    result = {
        "struct_in_embeddings": struct_in_embeddings,   
        "struct_out_embeddings": struct_out_embeddings,  
        "text_node_embeddings": text_node_embeddings,  
        "struct_in_mask": struct_in_mask,            
        "struct_out_mask": struct_out_mask,         
        "text_mask": text_mask,             
        "timesteps": timesteps,          
        "attention_mask": attention_mask,            
        "labels": labels,                          
        "node_ids": node_ids,
        "seq_lens": seq_lens,                   
    }

    if has_any_edge_aug:
        result.update({
            "struct_in_embeddings_aug": struct_in_embeddings_aug,
            "struct_out_embeddings_aug": struct_out_embeddings_aug,
            "struct_in_mask_aug": struct_in_mask_aug,
            "struct_out_mask_aug": struct_out_mask_aug,
        })

    return result


class BalancedPairBatchSampler(Sampler[List[int]]):
    """Yield batches containing both CIB and normal users for pairwise losses."""

    def __init__(self, labels: List[int], batch_size: int, seed: int = 42):
        if batch_size < 2:
            raise ValueError("pairwise_train_batches=True requires batch_size >= 2")
        self.labels = [int(x) for x in labels]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.pos_indices = [i for i, y in enumerate(self.labels) if y == 1]
        self.neg_indices = [i for i, y in enumerate(self.labels) if y == 0]
        self.epoch = 0
        if not self.pos_indices or not self.neg_indices:
            raise ValueError("Pairwise batch sampler requires at least one positive and one negative sample")

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        pos = rng.permutation(self.pos_indices).tolist()
        neg = rng.permutation(self.neg_indices).tolist()
        n_pos = max(1, self.batch_size // 2)
        n_neg = max(1, self.batch_size - n_pos)
        num_batches = min(len(pos) // n_pos, len(neg) // n_neg)
        for b in range(num_batches):
            batch = pos[b * n_pos:(b + 1) * n_pos] + neg[b * n_neg:(b + 1) * n_neg]
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        n_pos = max(1, self.batch_size // 2)
        n_neg = max(1, self.batch_size - n_pos)
        return min(len(self.pos_indices) // n_pos, len(self.neg_indices) // n_neg)


def create_data_loaders(
    train_node_ids: List[str],
    val_node_ids: List[str],
    node_data: Dict,
    labels: Dict,
    config,
):
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

    train_labels = [labels[nid] for nid in train_dataset.valid_node_ids]
    if getattr(config, "pairwise_train_batches", False):
        train_sampler: Optional[BalancedPairBatchSampler] = BalancedPairBatchSampler(
            train_labels,
            batch_size=config.batch_size,
            seed=int(getattr(config, "current_run_seed", getattr(config, "split_seed", 42))),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
    else:
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
