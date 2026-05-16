import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score

from config import TrainingConfig
from utils import (
    find_paired_files,
    load_paired_embeddings,
    load_original_graph_data,
    get_user_label,
)
from data_loader import TemporalGraphTextDataset, collate_fn
from unified_model import UnifiedTemporalGTLLM
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== 超参数 =====
TEMPERATURE = 1.8


# ===================== 数据加载 =====================
def load_all_data(config: TrainingConfig) -> Tuple[Dict, Dict, Dict]:
    print("=" * 60)
    print("Few-shot Prompt 阶段：加载数据（全体用户，节点级表示）...")
    print("=" * 60)

    paired_files = find_paired_files(config.gat_encoded_dir, config.text_encoded_dir)
    graph_data = load_original_graph_data(config.original_graph_data_path)

    node_data: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}
    successful_nodes = 0

    print(f"找到配对文件数: {len(paired_files)}")

    for gat_file, text_file in tqdm(paired_files, desc="加载节点数据"):
        embeddings = load_paired_embeddings(gat_file, text_file)
        if embeddings is None:
            continue

        node_id = embeddings.get("node_id", None)
        if node_id is None:
            continue

        has_label, label = get_user_label(node_id, graph_data)
        if not has_label:
            continue

        temporal_node_embeddings = embeddings.get("temporal_node_embeddings", {})
        temporal_text_embeddings = embeddings.get("temporal_text_embeddings", {})
        timesteps = embeddings.get("timesteps", [])

        timesteps_list: List[int] = []
        if isinstance(timesteps, torch.Tensor):
            timesteps_list = timesteps.view(-1).long().tolist()
        elif isinstance(timesteps, np.ndarray):
            timesteps_list = [int(x) for x in timesteps.tolist()]
        elif isinstance(timesteps, (list, tuple)):
            timesteps_list = [int(x) for x in timesteps]
        else:
            timesteps_list = []

        if len(timesteps_list) == 0:
            ts_keys = set()
            if isinstance(temporal_node_embeddings, dict):
                ts_keys.update(list(temporal_node_embeddings.keys()))
            if isinstance(temporal_text_embeddings, dict):
                ts_keys.update(list(temporal_text_embeddings.keys()))
            timesteps_list = [int(t) for t in ts_keys]

        if len(timesteps_list) == 0:
            continue

        timesteps_sorted = sorted(int(t) for t in timesteps_list)

        if (
            (not isinstance(temporal_node_embeddings, dict) or len(temporal_node_embeddings) == 0)
            and (not isinstance(temporal_text_embeddings, dict) or len(temporal_text_embeddings) == 0)
        ):
            continue

        node_data[node_id] = {
            "temporal_node_embeddings": temporal_node_embeddings,
            "temporal_text_embeddings": temporal_text_embeddings,
            "timesteps": timesteps_sorted,
        }
        labels[node_id] = int(label)
        successful_nodes += 1

    print(f"数据加载完成: {successful_nodes} 个节点(用户)")
    print(f"标签分布: 恶意={sum(labels.values())}, 正常={len(labels) - sum(labels.values())}")
    return node_data, labels, graph_data


# ===================== 支撑集采样 =====================
def select_support_pairs(malicious_ids: List[str], benign_ids: List[str], k: int, seed: int = 42) -> List[str]:
    rng = np.random.RandomState(seed)
    if len(malicious_ids) < k or len(benign_ids) < k:
        raise ValueError(f"支撑集采样失败：需要各 {k} 条，实际恶意={len(malicious_ids)}, 正常={len(benign_ids)}")
    mal = rng.choice(malicious_ids, size=k, replace=False).tolist()
    ben = rng.choice(benign_ids, size=k, replace=False).tolist()
    return mal + ben


# ===================== 预计算时序序列（保留完整时序） =====================
def precompute_time_sequences(model: UnifiedTemporalGTLLM, loader, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    返回: {node_id: time_seq [T, 512]}，保留完整时序，不平均
    """
    model.eval()
    nid_to_seq = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="预计算时序序列"):
            struct_in = batch["struct_in_embeddings"].to(device)
            struct_out = batch["struct_out_embeddings"].to(device)
            text_nodes = batch["text_node_embeddings"].to(device)

            struct_in_mask = batch["struct_in_mask"].to(device)
            struct_out_mask = batch["struct_out_mask"].to(device)
            text_mask = batch["text_mask"].to(device)

            time_mask = batch["attention_mask"].to(device)
            node_ids = batch["node_ids"]

            structure_embeddings, text_embeddings, attention_mask = model._pool_from_node_level(
                struct_in, struct_out, text_nodes,
                struct_in_mask, struct_out_mask, text_mask, time_mask
            )
            text_aligned = model.text_to_struct_proj(text_embeddings)
            fused = torch.cat([structure_embeddings, text_aligned], dim=-1)  # [B, T, 512]

            for i, nid in enumerate(node_ids):
                valid_len = int(attention_mask[i].sum().item())
                if valid_len > 0:
                    nid_to_seq[nid] = fused[i, :valid_len].clone()  # [T, 512] 保留完整时序
    return nid_to_seq


# ===================== 构建时序级对比前缀 =====================
def _build_temporal_diff_prefix(
    model: UnifiedTemporalGTLLM,
    tokenizer,
    support_ids: List[str],
    labels: Dict[str, int],
    support_seqs: Dict[str, torch.Tensor],  # {nid: [T_s, 512]}
    test_seq: torch.Tensor,  # [T_t, 512] 待测样本的完整时序
    device: torch.device,
) -> torch.Tensor:
    """
    时序级对比：
    1. 对每个支撑样本，计算其时序与待测时序的逐步差异
    2. 保留动态信息，不做平均
    """
    if len(support_ids) == 0:
        return model._get_or_build_time_prefix(tokenizer, device)

    # === 文本前缀 ===
    system_prompt = (
        "你是社交网络安全专家。模型已在Twitter完成训练，现迁移到Weibo。\n"
        "你将看到支撑示例与待测用户的**时序级差异向量**。\n"
        "每个差异向量捕捉了两者在对应时间步的行为距离，时序差异模式能揭示用户类别。"
    )
    
    example_list = []
    for i, nid in enumerate(support_ids, 1):
        lbl_txt = "恶意" if labels.get(nid, 0) == 1 else "正常"
        example_list.append(f"支撑{i}: [时序序列, 逐步差异向量, 标签={lbl_txt}]")
    support_desc = "以下是支撑样本（下方软token包含完整时序+差异）：\n" + "\n".join(example_list)
    
    cot_guide = (
        "\n推理步骤：\n"
        "1. 观察待测用户与每个支撑样本的时序差异模式\n"
        "2. 差异小且模式相似表示同类，差异大且模式不同表示异类\n"
        "3. 综合判断输出 0（正常）或 1（恶意）"
    )
    
    text_prefix = (
        "[INST]<<SYS>>" + system_prompt + "<</SYS>>\n" +
        support_desc + cot_guide + "[/INST]"
    )
    
    with torch.no_grad():
        text_ids = tokenizer(text_prefix, return_tensors="pt").input_ids.to(device)
        text_emb = model.llm.get_input_embeddings()(text_ids)[0]

    # === 构建时序级对比软token ===
    soft_tokens = []
    max_pos = model.config.max_sequence_length
    
    test_seq = test_seq.to(device)  # [T_t, 512]
    T_t = test_seq.size(0)
    
    for nid in support_ids:
        if nid not in support_seqs:
            continue
        support_seq = support_seqs[nid].to(device)  # [T_s, 512]
        T_s = support_seq.size(0)
        
        # 对齐长度：截断或补零到相同长度
        if T_s > T_t:
            support_seq = support_seq[:T_t]
        elif T_s < T_t:
            pad = torch.zeros(T_t - T_s, 512, device=device)
            support_seq = torch.cat([support_seq, pad], dim=0)
        
        # 逐时间步差异
        diff_seq = test_seq - support_seq  # [T_t, 512]
        
        # 拼接：[支撑时序, 差异时序]
        combined = torch.cat([support_seq, diff_seq], dim=0)  # [2*T_t, 512]
        
        # 位置编码 + 投影
        pos_ids = torch.arange(combined.size(0), device=device)
        pos_ids = torch.clamp(pos_ids, 0, max_pos - 1)
        pos = model.temporal_position_embedding(pos_ids)
        
        fused = combined + pos[:combined.size(0)]
        fused = model.fusion_norm(fused)
        llm_tokens = model.llm_projector(fused)  # [2*T_t, d_llm]
        
        soft_tokens.append(llm_tokens)
        
        # 分隔符
        sep = torch.zeros(1, model.llm.config.hidden_size, device=device)
        soft_tokens.append(sep)
    
    if len(soft_tokens) > 0:
        soft_cat = torch.cat(soft_tokens, dim=0)
    else:
        soft_cat = torch.zeros(0, model.llm.config.hidden_size, device=device)

    prefix = torch.cat([text_emb, soft_cat], dim=0)
    return prefix


# ===================== 逐样本推理（时序级差异） =====================
def compute_scores_with_temporal_diff(
    model: UnifiedTemporalGTLLM,
    data_loader,
    support_ids: List[str],
    labels: Dict[str, int],
    support_seqs: Dict[str, torch.Tensor],
    test_seqs: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    temperature: float = TEMPERATURE
):
    """
    逐样本推理：为每个待测样本构建时序级差异前缀
    """
    model.eval()
    all_s_gen: List[float] = []
    all_labels: List[int] = []
    all_node_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="计算 s_gen (时序级差异)"):
            struct_in = batch["struct_in_embeddings"].to(device)
            struct_out = batch["struct_out_embeddings"].to(device)
            text_nodes = batch["text_node_embeddings"].to(device)

            struct_in_mask = batch["struct_in_mask"].to(device)
            struct_out_mask = batch["struct_out_mask"].to(device)
            text_mask = batch["text_mask"].to(device)

            timesteps = batch["timesteps"].to(device)
            time_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            node_ids = batch["node_ids"]

            for i in range(len(node_ids)):
                nid = node_ids[i]
                if nid not in test_seqs:
                    continue
                
                # 构建时序级差异前缀
                test_seq = test_seqs[nid]
                prefix_emb = _build_temporal_diff_prefix(
                    model, tokenizer, support_ids, labels, support_seqs, test_seq, device
                )
                model._time_prefix_cache["time_prefix"] = prefix_emb
                
                # 单样本推理
                outputs = model(
                    struct_in_embeddings=struct_in[i:i+1],
                    struct_out_embeddings=struct_out[i:i+1],
                    text_node_embeddings=text_nodes[i:i+1],
                    struct_in_mask=struct_in_mask[i:i+1],
                    struct_out_mask=struct_out_mask[i:i+1],
                    text_mask=text_mask[i:i+1],
                    timesteps=timesteps[i:i+1],
                    attention_mask=time_mask[i:i+1],
                    labels=None,
                )

                logits = outputs["logits"]
                last_token_logits = logits[0, -1, :]

                benign_token = model.llm_tokenizer("0", return_tensors="pt", add_special_tokens=False)["input_ids"][0][-1].item()
                malicious_token = model.llm_tokenizer("1", return_tensors="pt", add_special_tokens=False)["input_ids"][0][-1].item()

                pair_logits = torch.tensor(
                    [last_token_logits[benign_token], last_token_logits[malicious_token]],
                    device=device
                )
                pair_logits = pair_logits / temperature
                pair_prob = F.softmax(pair_logits, dim=-1)
                p_malicious = pair_prob[1].item()

                all_s_gen.append(p_malicious)
                all_labels.append(batch_labels[i].item())
                all_node_ids.append(nid)

    return np.array(all_s_gen), np.array(all_labels), all_node_ids


def apply_single_threshold(scores: np.ndarray, y_true: np.ndarray, theta: float) -> Dict:
    y_pred = (scores > theta).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]

    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    J = tpr - fpr
    acc = (tp + tn) / max((tp + tn + fp + fn), 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        if len(np.unique(y_true)) == 2:
            roc_auc = float(roc_auc_score(y_true, scores))
        else:
            roc_auc = float("nan")
    except Exception:
        roc_auc = float("nan")

    return {
        "theta": float(theta),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "TPR": float(tpr),
        "FPR": float(fpr),
        "Youden_J": float(J),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


def grid_search_theta(scores: np.ndarray, y_true: np.ndarray, num_steps: int = 101) -> Dict:
    best = None
    best_f1 = -1.0
    eps = 1e-12

    for i in range(num_steps):
        theta = i / (num_steps - 1)
        metrics = apply_single_threshold(scores, y_true, theta)
        f1 = metrics["f1"]
        if (f1 > best_f1 + eps) or (abs(f1 - best_f1) <= eps and best is not None and metrics["precision"] > best["precision"] + eps):
            best_f1 = f1
            best = metrics

    return best


def compute_recall_at_k_with_counts(
    s_gen: np.ndarray,
    y_true: np.ndarray,
    k_list: List[int],
) -> Dict[str, Dict[str, float]]:
    assert s_gen.shape == y_true.shape
    total_pos = int(np.sum(y_true == 1))

    results: Dict[str, Dict[str, float]] = {}

    if total_pos == 0:
        for k in k_list:
            results[f"R@{k}"] = {"hit": 0, "total_pos": 0, "recall": 0.0}
        return results

    sorted_idx = np.argsort(-s_gen)
    n = len(s_gen)

    for k in k_list:
        k_eff = min(k, n)
        topk_idx = sorted_idx[:k_eff]
        hit = int(np.sum(y_true[topk_idx] == 1))
        results[f"R@{k}"] = {
            "hit": hit,
            "total_pos": total_pos,
            "recall": float(hit / total_pos),
        }

    return results


def build_dataloader_for_ids(
    ids: List[str],
    node_data: Dict[str, Dict],
    labels: Dict[str, int],
    config: TrainingConfig,
    is_training: bool = False,
):
    from torch.utils.data import DataLoader

    subset_node_data = {nid: node_data[nid] for nid in ids}
    subset_labels = {nid: labels[nid] for nid in ids}

    struct_node_dim = config.structure_dim // 2
    text_dim = config.text_dim

    dataset = TemporalGraphTextDataset(
        node_data=subset_node_data,
        labels=subset_labels,
        max_seq_len=config.max_sequence_length,
        is_training=is_training,
        struct_node_dim=struct_node_dim,
        text_dim=text_dim,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, max(len(dataset), 1)),
        shuffle=is_training,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return dataset, loader


# ===================== 主流程 =====================
def main():
    print("=" * 80)
    print("Few-shot Prompt 迁移（时序级对比，保留完整时序信息）")
    print(f"温度参数 T={TEMPERATURE}")
    print("=" * 80)

    config = TrainingConfig()
    device = config.device
    print(f"使用设备: {device}")

    base_output_dir = os.path.join(config.output_dir, "few_shot_temporal_diff")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"结果输出目录: {os.path.abspath(base_output_dir)}")

    node_data, labels, _ = load_all_data(config)
    if len(node_data) == 0 or len(labels) == 0:
        print("❌ 没有有效数据，退出")
        return

    split_path = os.path.join(config.output_dir, "dataset_split.json")
    if not os.path.exists(split_path):
        print(f"❌ 未找到训练划分文件: {split_path}")
        return
    with open(split_path, "r") as f:
        split_info = json.load(f)
    train_ids = split_info.get("train_node_ids", [])
    val_ids = split_info.get("val_node_ids", [])
    test_ids = split_info.get("test_node_ids", [])

    support_pool = [nid for nid in train_ids + val_ids if nid in labels]
    eval_ids_fixed = [nid for nid in test_ids if nid in labels]

    print(f"固定评测集: {len(eval_ids_fixed)}")

    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    twitter_best_model_path = "/home/llm/xy_tweets_data/ablation_exe/joint_lora_output/fold_1/fold_1_best_model.pth"
    if not os.path.exists(twitter_best_model_path):
        print(f"❌ 未找到模型文件: {twitter_best_model_path}")
        return

    print("加载模型参数...")
    model = UnifiedTemporalGTLLM(config)
    model.set_tokenizer(tokenizer)
    model.load_model(twitter_best_model_path)
    model.to(device)
    model.eval()

    # === 预计算评测集时序（只算一次，保留完整时序） ===
    print("\n预计算评测集时序...")
    eval_dataset, eval_loader = build_dataloader_for_ids(
        eval_ids_fixed, node_data, labels, config, is_training=False
    )
    test_seqs = precompute_time_sequences(model, eval_loader, device)

    support_pairs_list = [0, 10, 20, 30]
    all_results = []

    for k in support_pairs_list:
        print("\n" + "=" * 60)
        print(f"[时序级对比] 支撑集规模: {k} 对")
        print("=" * 60)

        if k > 0:
            support_ids = select_support_pairs(
                [nid for nid in support_pool if labels[nid] == 1],
                [nid for nid in support_pool if labels[nid] == 0],
                k=k,
                seed=42 + k,
            )
            _, support_loader = build_dataloader_for_ids(
                support_ids, node_data, labels, config, is_training=False
            )
            support_seqs = precompute_time_sequences(model, support_loader, device)
        else:
            support_ids = []
            support_seqs = {}

        print(f"支撑集: {len(support_ids)} | 评测集: {len(eval_ids_fixed)}")

        s_gen, y_true, node_ids_eval = compute_scores_with_temporal_diff(
            model, eval_loader, support_ids, labels, support_seqs, test_seqs,
            tokenizer, device, temperature=TEMPERATURE
        )

        if len(y_true) == 0:
            print("⚠️ 评测集为空，跳过。")
            continue

        best_metrics = grid_search_theta(s_gen, y_true, num_steps=101)
        recalls = compute_recall_at_k_with_counts(s_gen, y_true, k_list=[10, 30, 50])

        print(
            f"  最优阈值 θ={best_metrics['theta']:.4f} | F1={best_metrics['f1']:.4f} | "
            f"P={best_metrics['precision']:.4f} R={best_metrics['recall']:.4f}"
        )
        print(f"  Recall@10/30/50: {recalls}")

        result = {
            "support_pairs": k,
            "theta": best_metrics["theta"],
            "metrics": best_metrics,
            "recall_at_k": recalls,
            "temperature": TEMPERATURE,
        }
        all_results.append(result)

        out_path = os.path.join(base_output_dir, f"temporal_diff_{k}pairs.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存: {out_path}")

    summary = {
        "support_pairs_list": support_pairs_list,
        "temperature": TEMPERATURE,
        "method": "时序级对比（保留完整时序）",
        "results": all_results,
    }
    summary_path = os.path.join(base_output_dir, "temporal_diff_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n✅ 完成时序级对比评测:", summary_path)


if __name__ == "__main__":
    main()