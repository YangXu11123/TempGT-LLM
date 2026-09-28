import datetime
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from cee_dynamics import CEEDynamicsHead
from config import TrainingConfig
from data_loader import TemporalGraphTextDataset, collate_fn
from train import (
    assert_split_protocol,
    build_split_payload,
    can_reuse_split,
    load_all_data,
    save_json,
    split_train_balanced_val_test_imbalanced,
)
from unified_model import LearnableAttentionPooling


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenCEEStateBuilder(nn.Module):
    """Initial main-model state builder used only to define x_t for CEE pretraining."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        struct_node_dim = config.structure_dim // 2
        self.struct_att_pool = LearnableAttentionPooling(struct_node_dim, config.attention_hidden_dim)
        self.text_att_pool = LearnableAttentionPooling(config.text_dim, config.attention_hidden_dim)
        self.text_to_struct_proj = nn.Linear(config.text_dim, config.structure_dim, bias=False)
        self.config = config
        for param in self.parameters():
            param.requires_grad = False

    def export_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "struct_att_pool": self.struct_att_pool.state_dict(),
            "text_att_pool": self.text_att_pool.state_dict(),
            "text_to_struct_proj": self.text_to_struct_proj.state_dict(),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.config.device
        struct_in = batch["struct_in_embeddings"].to(device)
        struct_out = batch["struct_out_embeddings"].to(device)
        text_nodes = batch["text_node_embeddings"].to(device)
        struct_in_mask = batch["struct_in_mask"].to(device)
        struct_out_mask = batch["struct_out_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        time_mask = batch["attention_mask"].to(device)

        batch_size, seq_len, n_in, dim_in = struct_in.shape
        _, _, n_out, dim_out = struct_out.shape
        _, _, n_txt, dim_txt = text_nodes.shape

        in_flat = struct_in.view(batch_size * seq_len, n_in, dim_in)
        out_flat = struct_out.view(batch_size * seq_len, n_out, dim_out)
        text_flat = text_nodes.view(batch_size * seq_len, n_txt, dim_txt)

        h_in = self.struct_att_pool(in_flat, struct_in_mask.view(batch_size * seq_len, n_in)).view(batch_size, seq_len, dim_in)
        h_out = self.struct_att_pool(out_flat, struct_out_mask.view(batch_size * seq_len, n_out)).view(batch_size, seq_len, dim_out)
        h_text = self.text_att_pool(text_flat, text_mask.view(batch_size * seq_len, n_txt)).view(batch_size, seq_len, dim_txt)

        structure_states = torch.cat([h_in, h_out], dim=-1)
        text_states = self.text_to_struct_proj(h_text)
        states = torch.cat([structure_states, text_states], dim=-1)
        return states, time_mask


def get_or_create_split(config: TrainingConfig, node_data: Dict, labels: Dict[str, int]) -> Tuple[List[str], List[str], List[str]]:
    split_path = os.path.join(config.output_dir, "dataset_split.json")
    expected_ratio = getattr(config, "val_test_neg_pos_ratio", None)
    os.makedirs(config.output_dir, exist_ok=True)

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        if can_reuse_split(split_data, config, labels, node_data):
            return split_data["train_node_ids"], split_data["val_node_ids"], split_data["test_node_ids"]

    node_ids = list(labels.keys())
    train_ids, val_ids, test_ids = split_train_balanced_val_test_imbalanced(
        node_ids,
        labels,
        train_ratio=getattr(config, "train_ratio", 0.6),
        val_ratio=getattr(config, "val_ratio", 0.2),
        test_ratio=getattr(config, "test_ratio", 0.2),
        seed=getattr(config, "split_seed", 42),
        val_test_neg_pos_ratio=expected_ratio,
    )
    payload = build_split_payload(train_ids, val_ids, test_ids, labels, config)
    assert_split_protocol(payload, labels, expected_ratio)
    save_json(split_path, payload)
    return train_ids, val_ids, test_ids


def main() -> None:
    config = TrainingConfig()
    # CEE dynamics pretraining uses x_t -> x_{t+1}; edge-drop views are only for L_cont in main training.
    config.edge_drop_gat_encoded_dir = ""
    set_seed(int(getattr(config, "cee_state_seed", 20260911)))
    os.makedirs(config.output_dir, exist_ok=True)

    node_data, labels, _ = load_all_data(config)
    if not node_data:
        raise RuntimeError("没有可用节点数据，无法预训练 CEE dynamics head")

    train_ids, _, _ = get_or_create_split(config, node_data, labels)
    train_node_data = {nid: node_data[nid] for nid in train_ids if nid in node_data}
    train_labels = {nid: 0 for nid in train_node_data.keys()}

    dataset = TemporalGraphTextDataset(
        train_node_data,
        train_labels,
        max_seq_len=config.max_sequence_length,
        is_training=True,
        struct_node_dim=config.structure_dim // 2,
        text_dim=config.text_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    state_builder = FrozenCEEStateBuilder(config).to(config.device).eval()
    head = CEEDynamicsHead(2 * config.structure_dim, getattr(config, "cee_hidden_dim", 512)).to(config.device)
    optimizer = optim.AdamW(head.parameters(), lr=float(getattr(config, "cee_pretrain_lr", 1e-3)), weight_decay=1e-5)

    history = []
    for epoch in range(int(getattr(config, "cee_pretrain_epochs", 20))):
        head.train()
        total_loss = 0.0
        total_pairs = 0

        for batch in tqdm(loader, desc=f"CEE pretrain epoch {epoch + 1}"):
            with torch.no_grad():
                states, mask = state_builder(batch)
            valid_pairs = (mask[:, :-1].bool() & mask[:, 1:].bool()).sum().item() if mask.size(1) >= 2 else 0
            if valid_pairs <= 0:
                continue

            scores = head.compute_cee(
                states,
                attention_mask=mask,
                sigma=getattr(config, "cee_sigma", 1.0),
                include_constant=False,
            )
            loss = scores.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu().item()) * int(valid_pairs)
            total_pairs += int(valid_pairs)

        avg_loss = total_loss / max(total_pairs, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "transition_pairs": total_pairs})
        print(f"epoch={epoch + 1} loss={avg_loss:.6f} transition_pairs={total_pairs}")

    head.freeze()

    cee_path = getattr(config, "cee_head_path", "")
    if not os.path.isabs(cee_path):
        cee_path = os.path.abspath(cee_path)
    os.makedirs(os.path.dirname(cee_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": head.state_dict(),
        "input_dim": int(2 * config.structure_dim),
        "hidden_dim": int(getattr(config, "cee_hidden_dim", 512)),
        "sigma": float(getattr(config, "cee_sigma", 1.0)),
        "state_builder_state_dict": state_builder.export_state_dict(),
        "train_node_count": int(len(train_node_data)),
        "used_cib_labels_in_loss": False,
        "task": "unsupervised_autoregressive_dynamics_xt_to_xt_plus_1",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    torch.save(checkpoint, cee_path)

    summary = {
        "cee_head_path": cee_path,
        "train_node_count": int(len(train_node_data)),
        "used_cib_labels_in_loss": False,
        "num_epochs": int(getattr(config, "cee_pretrain_epochs", 20)),
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_json(os.path.join(config.output_dir, "cee_pretrain_summary.json"), summary)
    save_json(os.path.join(config.output_dir, "cee_pretrain_config.json"), {k: str(v) if isinstance(v, torch.device) else v for k, v in config.__dict__.items()})
    print(f"CEE dynamics head saved: {cee_path}")


if __name__ == "__main__":
    main()
