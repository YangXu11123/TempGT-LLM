import os
import argparse
import json
import csv
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

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


# 加载所有节点数据
def load_all_data(config: TrainingConfig) -> Tuple[Dict, Dict, Dict, List[str]]:
    print("=" * 60)
    print("推理阶段：加载数据（全体用户，节点级表示）...")
    print("=" * 60)

    paired_files_main = find_paired_files(config.gat_encoded_dir, config.text_encoded_dir)
    paired_files_normal = find_paired_files(
        config.normal_gat_encoded_dir,
        config.normal_text_encoded_dir,
    )

    paired_files = paired_files_main + paired_files_normal
    print(
        f"总配对文件数 = 原始 {len(paired_files_main)} + "
        f"正常6000 {len(paired_files_normal)} = {len(paired_files)}"
    )

    graph_data = load_original_graph_data(config.original_graph_data_path)

    node_data: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}
    successful_nodes = 0
    normal_6000_node_ids: List[str] = []

    for gat_file, text_file in tqdm(paired_files, desc="加载节点数据"):
        embeddings = load_paired_embeddings(gat_file, text_file)
        if embeddings is None:
            continue

        node_id = embeddings.get("node_id", None)
        if node_id is None:
            continue
        node_id = str(node_id)

        has_label, label = get_user_label(node_id, graph_data)
        if not has_label:
            continue

        temporal_node_embeddings = embeddings.get("temporal_node_embeddings", {})
        temporal_text_embeddings = embeddings.get("temporal_text_embeddings", {})
        timesteps = embeddings.get("timesteps", [])

        timesteps_list: List[int] = []
        if isinstance(timesteps, torch.Tensor):
            timesteps_list = timesteps.long().tolist()
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

        from_normal6000 = config.normal_gat_encoded_dir in os.path.abspath(gat_file)
        # 与训练阶段一致：重复账号保留主数据版本，避免被另一套 GAT 表示覆盖。
        if from_normal6000 and node_id in node_data:
            continue

        node_data[node_id] = {
            "temporal_node_embeddings": temporal_node_embeddings,
            "temporal_text_embeddings": temporal_text_embeddings,
            "timesteps": timesteps_sorted,
            "source": "normal_6000" if from_normal6000 else "main",
        }
        labels[node_id] = int(label)
        successful_nodes += 1

        # 记录来自 normal_6000 目录的节点
        if from_normal6000:
            normal_6000_node_ids.append(node_id)

    normal_6000_node_ids = sorted(list(set(normal_6000_node_ids)))

    print(f"数据加载完成: {successful_nodes} 个节点(用户)")
    print(f"标签分布: 恶意={sum(labels.values())}, 正常={len(labels) - sum(labels.values())}")
    print(f"normal_6000 中可用节点数: {len(normal_6000_node_ids)}")
    return node_data, labels, graph_data, normal_6000_node_ids


# 计算 s_gen + 逐账号 raw CEE
def compute_scores_for_loader(
    model: UnifiedTemporalGTLLM,
    data_loader,
    config: TrainingConfig,
):
    model.eval()
    device = config.device

    all_s_gen: List[float] = []
    all_labels: List[int] = []
    all_node_ids: List[str] = []
    all_cee: List[float] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="计算 s_gen + CEE（测试集用户）"):
            struct_in = batch["struct_in_embeddings"].to(device)
            struct_out = batch["struct_out_embeddings"].to(device)
            text_nodes = batch["text_node_embeddings"].to(device)

            struct_in_mask = batch["struct_in_mask"].to(device)
            struct_out_mask = batch["struct_out_mask"].to(device)
            text_mask = batch["text_mask"].to(device)

            timesteps = batch["timesteps"].to(device)
            time_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            node_ids = batch["node_ids"]

            outputs = model(
                struct_in_embeddings=struct_in,
                struct_out_embeddings=struct_out,
                text_node_embeddings=text_nodes,
                struct_in_mask=struct_in_mask,
                struct_out_mask=struct_out_mask,
                text_mask=text_mask,
                timesteps=timesteps,
                attention_mask=time_mask,
                labels=None,
            )

            # forward 已按每个样本的最后一个有效位置提取 {'0','1'} logits。
            pair_logits = outputs["logits_2"]
            pair_prob = F.softmax(pair_logits, dim=-1)
            p_malicious = pair_prob[:, 1]

            # raw CEE（逐样本标量）
            structure_embeddings, text_embeddings, pooled_time_mask = model._pool_from_node_level(
                struct_in,
                struct_out,
                text_nodes,
                struct_in_mask,
                struct_out_mask,
                text_mask,
                time_mask,
            )

            b_cur, t_cur, struct_dim = structure_embeddings.shape
            text_dim = text_embeddings.size(-1)

            proj_text = model.text_to_struct_proj(
                text_embeddings.view(b_cur * t_cur, text_dim)
            ).view(b_cur, t_cur, struct_dim)
            fused_states = torch.cat([structure_embeddings, proj_text], dim=-1)

            batch_cee = []
            for i_b in range(b_cur):
                mask_i = pooled_time_mask[i_b].bool()
                states_i = fused_states[i_b, mask_i]
                if states_i.size(0) < 2:
                    cee_i = torch.tensor(0.0, device=device)
                else:
                    cee_i = model._compute_cee_for_states(states_i)
                batch_cee.append(float(cee_i.detach().cpu().item()))

            all_s_gen.extend(p_malicious.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_node_ids.extend(node_ids)
            all_cee.extend(batch_cee)

    return np.array(all_s_gen), np.array(all_labels), all_node_ids, np.array(all_cee)


def file_sha256(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_cee_groups(cee_scores: np.ndarray, labels: np.ndarray, config: TrainingConfig) -> Dict:
    """冻结 CEE head 的组间打分（修改方向.md 一.5）：normal vs malicious 的分布统计
    + Mann-Whitney U 双侧检验（p 值、效应量 r）。"""
    cee_scores = np.asarray(cee_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    out = {
        "score_definition": "mean conditional log likelihood; higher means more predictable",
        "cee_head_path": os.path.abspath(getattr(config, "cee_head_path", "")),
        "cee_head_sha256": file_sha256(os.path.abspath(getattr(config, "cee_head_path", ""))),
        "groups": {},
    }

    for name, value in [("normal", 0), ("malicious", 1)]:
        arr = cee_scores[labels == value]
        if arr.size == 0:
            out["groups"][name] = {"n": 0}
            continue
        out["groups"][name] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "q1": float(np.quantile(arr, 0.25)),
            "q3": float(np.quantile(arr, 0.75)),
        }

    normal = cee_scores[labels == 0]
    malicious = cee_scores[labels == 1]
    if normal.size > 0 and malicious.size > 0:
        try:
            from scipy.stats import mannwhitneyu

            u_stat, p_value = mannwhitneyu(malicious, normal, alternative="two-sided")
            n_total = malicious.size + normal.size
            _, ties = np.unique(np.concatenate([malicious, normal]), return_counts=True)
            variance = malicious.size*normal.size/12.0 * (n_total+1 - float(np.sum(ties**3-ties))/(n_total*(n_total-1)))
            z_approx = (float(u_stat)-malicious.size*normal.size/2.0)/np.sqrt(variance) if variance > 0 else 0.0
            out["mann_whitney_u"] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "effect_size_r_approx": float(abs(z_approx) / np.sqrt(n_total)),
            }
        except Exception as exc:
            out["mann_whitney_u"] = {"error": str(exc)}
    return out


def save_abc_cee_analysis(
    s_gen: np.ndarray,
    cee_scores: np.ndarray,
    labels: np.ndarray,
    node_ids: List[str],
    output_dir: str,
    config: TrainingConfig,
) -> str:
    """构造关闭 L_CEE 模型的 Group A/B/C，并保存 CDF 与统计检验。"""
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    s_gen = np.asarray(s_gen, dtype=np.float64)
    cee_scores = np.asarray(cee_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(node_ids)
    if not (s_gen.size == cee_scores.size == labels.size == n):
        raise ValueError("ABC arrays and node_ids are not aligned")

    k = min(100, n)
    order = np.argsort(s_gen, kind="stable")
    group_indices = {
        "A_ground_truth_CIB": np.flatnonzero(labels == 1),
        "B_top_100_CIB_score": order[-k:][::-1],
        "C_bottom_100_benign_oriented": order[:k],
    }

    result = {
        "score_definition": "mean conditional log likelihood; higher means more predictable",
        "definition": {
            "A": "all ground-truth CIB accounts in the test set",
            "B": f"top-{k} test accounts by CIB classification score from lambda_CEE=0 model",
            "C": f"bottom-{k} test accounts by CIB classification score from the same lambda_CEE=0 model",
        },
        "cee_head_path": os.path.abspath(config.cee_head_path),
        "cee_head_sha256": file_sha256(os.path.abspath(config.cee_head_path)),
        "groups": {},
        "pairwise_mann_whitney_u": {},
    }

    group_values = {}
    for name, idx in group_indices.items():
        values = cee_scores[idx]
        group_values[name] = values
        result["groups"][name] = {
            "n": int(values.size),
            "node_ids": [str(node_ids[i]) for i in idx],
            "classification_scores": s_gen[idx].tolist(),
            "cee_scores": values.tolist(),
            "mean": float(np.mean(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "std": float(np.std(values)) if values.size else None,
        }

    pairs = [
        ("A_ground_truth_CIB", "B_top_100_CIB_score"),
        ("A_ground_truth_CIB", "C_bottom_100_benign_oriented"),
        ("B_top_100_CIB_score", "C_bottom_100_benign_oriented"),
    ]
    from abc_statistics import compare_groups, display_label, X_LABEL, Y_LABEL
    for left, right in pairs:
        a, b = group_values[left], group_values[right]
        result['pairwise_mann_whitney_u'][f'{left}_vs_{right}'] = compare_groups(
            a, b, result['groups'][left]['node_ids'], result['groups'][right]['node_ids'])

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "cee_groups_abc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(7, 5))
    for name, values in group_values.items():
        sorted_values = np.sort(values)
        cumulative = np.arange(1, values.size + 1) / values.size
        plt.step(sorted_values, cumulative, where="post", label=display_label(name))
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cee_groups_abc_cdf.png"), dpi=300)
    plt.close()
    return json_path


# 单阈值分类 + 指标计算
def apply_dual_threshold(
    s_gen: np.ndarray,
    y_true: np.ndarray,
    theta: float,
) -> Dict:
    y_pred = (s_gen > theta).astype(int)

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
            roc_auc_gen = float(roc_auc_score(y_true, s_gen))
        else:
            roc_auc_gen = float("nan")
    except Exception:
        roc_auc_gen = float("nan")

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
        "roc_auc_gen": roc_auc_gen,
        "y_pred": y_pred.tolist(),
    }


# R@K（命中数/恶意总数=比例）
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
        k_eff = min(int(k), n)
        topk_idx = sorted_idx[:k_eff]
        hit = int(np.sum(y_true[topk_idx] == 1))
        recall = hit / total_pos
        results[f"R@{k}"] = {"hit": hit, "total_pos": total_pos, "recall": float(recall)}

    return results


# 保存用户级详细分类结果和得分统计
def save_user_classification_details(
    node_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s_gen: np.ndarray,
    output_path: str,
):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_id",
                "true_label",
                "pred_label",
                "score",
                "label_desc",
                "pred_desc",
                "classification_result",
            ]
        )

        for i in range(len(node_ids)):
            node_id = node_ids[i]
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            score = float(s_gen[i])

            true_desc = "恶意" if true_label == 1 else "正常"
            pred_desc = "恶意" if pred_label == 1 else "正常"

            if true_label == pred_label:
                if true_label == 1:
                    result = "TP(正确识别恶意)"
                else:
                    result = "TN(正确识别正常)"
            else:
                if true_label == 1:
                    result = "FN(漏报恶意)"
                else:
                    result = "FP(误报正常)"

            writer.writerow(
                [
                    node_id,
                    true_label,
                    pred_label,
                    f"{score:.6f}",
                    true_desc,
                    pred_desc,
                    result,
                ]
            )

    print(f"用户分类详情已保存到: {output_path}")


def save_score_statistics(
    y_true: np.ndarray,
    s_gen: np.ndarray,
    output_path: str,
):
    malicious_mask = y_true == 1
    benign_mask = y_true == 0

    mal_scores = s_gen[malicious_mask]
    ben_scores = s_gen[benign_mask]

    stats = {
        "恶意节点得分统计": {
            "数量": int(len(mal_scores)),
            "最小值": float(np.min(mal_scores)) if len(mal_scores) > 0 else None,
            "最大值": float(np.max(mal_scores)) if len(mal_scores) > 0 else None,
            "均值": float(np.mean(mal_scores)) if len(mal_scores) > 0 else None,
            "中位数": float(np.median(mal_scores)) if len(mal_scores) > 0 else None,
            "标准差": float(np.std(mal_scores)) if len(mal_scores) > 0 else None,
            "25分位数": float(np.percentile(mal_scores, 25)) if len(mal_scores) > 0 else None,
            "75分位数": float(np.percentile(mal_scores, 75)) if len(mal_scores) > 0 else None,
        },
        "正常节点得分统计": {
            "数量": int(len(ben_scores)),
            "最小值": float(np.min(ben_scores)) if len(ben_scores) > 0 else None,
            "最大值": float(np.max(ben_scores)) if len(ben_scores) > 0 else None,
            "均值": float(np.mean(ben_scores)) if len(ben_scores) > 0 else None,
            "中位数": float(np.median(ben_scores)) if len(ben_scores) > 0 else None,
            "标准差": float(np.std(ben_scores)) if len(ben_scores) > 0 else None,
            "25分位数": float(np.percentile(ben_scores, 25)) if len(ben_scores) > 0 else None,
            "75分位数": float(np.percentile(ben_scores, 75)) if len(ben_scores) > 0 else None,
        },
        "整体得分统计": {
            "总数量": int(len(s_gen)),
            "最小值": float(np.min(s_gen)),
            "最大值": float(np.max(s_gen)),
            "均值": float(np.mean(s_gen)),
            "中位数": float(np.median(s_gen)),
            "标准差": float(np.std(s_gen)),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def summarize_mean_std(values: List[float]) -> Dict[str, float]:
    valid_values = [float(v) for v in values if v is not None and not np.isnan(v)]
    if len(valid_values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "mean_pm_std": "nan±nan"}

    mean_v = float(np.mean(valid_values))
    std_v = float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0
    return {
        "mean": mean_v,
        "std": std_v,
        "mean_pm_std": f"{mean_v:.4f}±{std_v:.4f}",
    }


def main():
    print("=" * 80)
    print("LLM 推理阶段：按 1:10~1:50 比例 + 5 随机种子评估（仅测试集 normal_6000 池采样）")
    print("=" * 80)

    # 固定实验配置：5个比例、5个随机种子
    ratio_list = [10, 20, 30, 40, 50]
    seed_list = [42, 43, 44, 45, 46]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--lambda-cee", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from config import load_config
    config = load_config(args.config)
    ratio_list = list(config.evaluation_ratios)
    seed_list = list(config.evaluation_seeds)
    if args.lambda_cee is not None:
        config.lambda_cee = float(args.lambda_cee)
    if args.output_dir:
        config.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"使用设备: {config.device}")

    node_data, labels, graph_data, normal_6000_node_ids = load_all_data(config)
    if len(node_data) == 0 or len(labels) == 0:
        raise RuntimeError('No valid inference data.')

    split_path = os.path.join(config.output_dir, "dataset_split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f'Missing fixed dataset split: {split_path}')

    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    test_ids_raw = split_info.get("test_node_ids", [])
    test_ids = [str(x) for x in test_ids_raw]

    # 测试集中的已加载节点
    test_ids_loaded = [nid for nid in test_ids if nid in node_data and nid in labels]
    if len(test_ids_loaded) != len(test_ids):
        raise RuntimeError('Fixed test accounts are missing; refusing evaluation on a silently reduced test set.')

    # 恶意固定集合（来自测试集）
    test_malicious_ids = [nid for nid in test_ids_loaded if int(labels[nid]) == 1]

    # 关键防泄露：正常采样池必须来自“测试集中的normal_6000部分”
    normal_6000_set = set(str(x) for x in normal_6000_node_ids)
    test_normal_pool_from_6000 = [
        nid for nid in test_ids_loaded
        if int(labels[nid]) == 0 and nid in normal_6000_set
    ]

    print(f"\n测试集总可用节点: {len(test_ids_loaded)}")
    print(f"测试集恶意节点数(固定): {len(test_malicious_ids)}")
    print(f"测试集 normal_6000 正常采样池: {len(test_normal_pool_from_6000)}")
    print("说明：后续正常用户仅从该采样池抽取，避免Val/Test泄露。")

    if len(test_malicious_ids) == 0:
        raise RuntimeError('No CIB users in the fixed test set; ratio evaluation cannot run.')

    # 加载模型与阈值（与原脚本衔接）
    fold_idx = 1
    fold_output_dir = os.path.join(config.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    best_model_path = os.path.join(fold_output_dir, f"fold_{fold_idx}_best_model.pth")
    fold_summary_path = os.path.join(fold_output_dir, "fold_summary.json")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f'Missing best model: {best_model_path}')
    if not os.path.exists(fold_summary_path):
        raise FileNotFoundError(f'Missing validation threshold summary: {fold_summary_path}')

    with open(fold_summary_path, "r", encoding="utf-8") as f:
        fold_summary = json.load(f)

    if "best_thresholds" not in fold_summary or "theta" not in fold_summary["best_thresholds"]:
        raise RuntimeError('fold_summary.json has no best_thresholds.theta.')

    theta = float(fold_summary["best_thresholds"]["theta"])
    print(f"\n使用阈值: θ={theta:.4f}")

    print("\n加载分词器...")
    from backbone_adapter import load_backbone_tokenizer
    tokenizer = load_backbone_tokenizer(config.llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型...")
    model = UnifiedTemporalGTLLM(config)
    model.set_tokenizer(tokenizer)
    model.load_model(best_model_path)
    model.to(config.device)
    model.eval()

    struct_node_dim = config.structure_dim // 2
    text_dim = config.text_dim

    ratio_eval_root = os.path.join(fold_output_dir, "ratio_seed_eval")
    os.makedirs(ratio_eval_root, exist_ok=True)

    # 全测试集冻结 CEE 打分（修改方向.md 一.5：报告组间 Mann-Whitney U 的 p 值与效应量 r）
    print("\n开始全测试集冻结 CEE head 打分...")
    full_test_dataset = TemporalGraphTextDataset(
        node_data={nid: node_data[nid] for nid in test_ids_loaded},
        labels={nid: labels[nid] for nid in test_ids_loaded},
        max_seq_len=config.max_sequence_length,
        is_training=False,
        struct_node_dim=struct_node_dim,
        text_dim=text_dim,
    )
    full_test_loader = DataLoader(
        full_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    s_gen_full, y_full, ids_full, cee_full = compute_scores_for_loader(model, full_test_loader, config)
    cee_full_summary = summarize_cee_groups(cee_full, y_full, config)

    print("[CEE] 全测试集组间统计:")
    for g_name, g_stats in cee_full_summary["groups"].items():
        print(f"  {g_name}: {g_stats}")
    if "mann_whitney_u" in cee_full_summary:
        print(f"  Mann-Whitney U: {cee_full_summary['mann_whitney_u']}")

    cee_full_path = os.path.join(fold_output_dir, "cee_full_test_summary.json")
    with open(cee_full_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "node_ids": ids_full,
                "y_true": y_full.tolist(),
                "cee_raw": cee_full.tolist(),
                **cee_full_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[CEE] 全测试集 CEE 摘要已保存: {cee_full_path}")

    if float(config.lambda_cee) == 0.0:
        abc_path = save_abc_cee_analysis(
            s_gen=s_gen_full,
            cee_scores=cee_full,
            labels=y_full,
            node_ids=ids_full,
            output_dir=fold_output_dir,
            config=config,
        )
        print(f"[CEE] Group A/B/C 分析已保存: {abc_path}")

    all_run_records: List[Dict] = []
    by_ratio_records: Dict[int, List[Dict]] = {r: [] for r in ratio_list}

    for ratio in ratio_list:
        target_normal = len(test_malicious_ids) * ratio
        print("\n" + "=" * 70)
        print(f"开始比例评估: 1:{ratio} | 恶意={len(test_malicious_ids)}, 目标正常={target_normal}")
        print("=" * 70)

        if target_normal > len(test_normal_pool_from_6000):
            print(
                f"跳过比例1:{ratio}，原因：采样池不足 "
                f"(需要{target_normal}，可用{len(test_normal_pool_from_6000)})"
            )
            continue

        for seed in seed_list:
            rng = np.random.RandomState(seed)

            sampled_normal = rng.choice(
                test_normal_pool_from_6000,
                size=target_normal,
                replace=False,
            ).tolist()

            eval_ids = list(test_malicious_ids) + sampled_normal
            rng.shuffle(eval_ids)

            eval_node_data = {nid: node_data[nid] for nid in eval_ids}
            eval_labels = {nid: labels[nid] for nid in eval_ids}

            test_dataset = TemporalGraphTextDataset(
                node_data=eval_node_data,
                labels=eval_labels,
                max_seq_len=config.max_sequence_length,
                is_training=False,
                struct_node_dim=struct_node_dim,
                text_dim=text_dim,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True,
            )

            print(
                f"\n[比例1:{ratio} | seed={seed}] "
                f"样本数={len(test_dataset)} (恶意={len(test_malicious_ids)}, 正常={target_normal})"
            )

            s_gen_all, y_all, node_ids_all, cee_all = compute_scores_for_loader(
                model, test_loader, config
            )

            metrics = apply_dual_threshold(s_gen_all, y_all, theta)
            recall_at_k = compute_recall_at_k_with_counts(s_gen_all, y_all, k_list=[10, 30, 50])
            cee_summary = summarize_cee_groups(cee_all, y_all, config)

            print(
                f"[比例1:{ratio} | seed={seed}] "
                f"F1={metrics['f1']:.4f}, "
                f"AUC={metrics['roc_auc_gen']:.4f}, "
                f"Youden={metrics['Youden_J']:.4f}, "
                f"R@10={recall_at_k['R@10']['recall']:.4f}, "
                f"R@30={recall_at_k['R@30']['recall']:.4f}, "
                f"R@50={recall_at_k['R@50']['recall']:.4f}"
            )

            run_dir = os.path.join(
                ratio_eval_root,
                f"ratio_1to{ratio}",
                f"seed_{seed}",
            )
            os.makedirs(run_dir, exist_ok=True)

            inference_results = {
                "fold_idx": fold_idx,
                "theta": metrics["theta"],
                "ratio": f"1:{ratio}",
                "ratio_normal_multiplier": ratio,
                "seed": seed,
                "num_malicious": len(test_malicious_ids),
                "num_normal": target_normal,
                "sampled_malicious_ids": test_malicious_ids,
                "sampled_normal_ids": sampled_normal,
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "TPR": metrics["TPR"],
                "FPR": metrics["FPR"],
                "Youden_J": metrics["Youden_J"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc_gen": metrics["roc_auc_gen"],
                "recall_at_k": recall_at_k,
                "node_ids": node_ids_all,
                "s_gen": s_gen_all.tolist(),
                "y_true": y_all.tolist(),
                "y_pred": metrics["y_pred"],
                "cee_raw": cee_all.tolist(),
                "cee_summary": cee_summary,
            }

            inference_path = os.path.join(run_dir, "inference_test_users.json")
            with open(inference_path, "w", encoding="utf-8") as f:
                json.dump(inference_results, f, indent=2, ensure_ascii=False)

            user_details_path = os.path.join(run_dir, "user_classification_details.csv")
            save_user_classification_details(
                node_ids_all,
                y_all,
                np.array(metrics["y_pred"]),
                s_gen_all,
                user_details_path,
            )

            score_stats_path = os.path.join(run_dir, "score_statistics.json")
            save_score_statistics(y_all, s_gen_all, score_stats_path)

            run_record = {
                "ratio": ratio,
                "seed": seed,
                "num_malicious": len(test_malicious_ids),
                "num_normal": target_normal,
                "f1": float(metrics["f1"]),
                "roc_auc_gen": float(metrics["roc_auc_gen"]),
                "Youden_J": float(metrics["Youden_J"]),
                "R@10": float(recall_at_k["R@10"]["recall"]),
                "R@30": float(recall_at_k["R@30"]["recall"]),
                "R@50": float(recall_at_k["R@50"]["recall"]),
                "output_dir": run_dir,
                "inference_json": inference_path,
            }

            all_run_records.append(run_record)
            by_ratio_records[ratio].append(run_record)

    # 聚合：每个比例下，对5个seed的指标做 mean±std
    metric_keys = ["f1", "roc_auc_gen", "Youden_J", "R@10", "R@30", "R@50"]
    ratio_aggregate = {}

    for ratio in ratio_list:
        records = by_ratio_records.get(ratio, [])
        if len(records) == 0:
            ratio_aggregate[f"1:{ratio}"] = {
                "num_runs": 0,
                "metrics": {},
                "note": "该比例未执行（通常是采样池不足）",
            }
            continue

        per_metric = {}
        for mk in metric_keys:
            vals = [float(r[mk]) for r in records]
            per_metric[mk] = summarize_mean_std(vals)

        ratio_aggregate[f"1:{ratio}"] = {
            "num_runs": len(records),
            "num_malicious": int(records[0]["num_malicious"]),
            "num_normal": int(records[0]["num_normal"]),
            "metrics": per_metric,
        }

    aggregate_output = {
        "description": "比例采样推理聚合结果（正常仅从测试集normal_6000池抽样）",
        "fold_idx": fold_idx,
        "theta": theta,
        "ratio_list": ratio_list,
        "seed_list": seed_list,
        "cee_full_test_summary_path": cee_full_path,
        "config": {k: str(v) if isinstance(v, torch.device) else v for k, v in config.__dict__.items()},
        "sampling_policy": {
            "malicious_source": "test_ids 中 label==1 的全部用户",
            "normal_source": "test_ids ∩ normal_6000_node_ids ∩ label==0",
            "leakage_avoidance": "不使用验证集normal_6000，不使用test中的非normal_6000正常用户",
        },
        "pool_stats": {
            "test_ids_loaded": len(test_ids_loaded),
            "test_malicious_fixed": len(test_malicious_ids),
            "test_normal_pool_from_6000": len(test_normal_pool_from_6000),
        },
        "ratio_aggregate": ratio_aggregate,
        "all_run_records": all_run_records,
    }

    aggregate_path = os.path.join(ratio_eval_root, "ratio_seed_aggregate_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("比例+多种子推理完成")
    print("=" * 80)
    print(f"聚合结果已保存: {aggregate_path}")

    print("\n各比例关键指标(mean±std):")
    for ratio in ratio_list:
        key = f"1:{ratio}"
        info = ratio_aggregate.get(key, {})
        if info.get("num_runs", 0) == 0:
            print(f"  {key}: 未执行")
            continue

        m = info["metrics"]
        print(
            f"  {key} | "
            f"F1={m['f1']['mean_pm_std']} | "
            f"AUC={m['roc_auc_gen']['mean_pm_std']} | "
            f"Youden={m['Youden_J']['mean_pm_std']} | "
            f"R@10={m['R@10']['mean_pm_std']} | "
            f"R@30={m['R@30']['mean_pm_std']} | "
            f"R@50={m['R@50']['mean_pm_std']}"
        )

    del model
    if config.device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
