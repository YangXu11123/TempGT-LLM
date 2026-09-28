import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import shutil
import os
import json
import datetime
import random
import time
import gc
import glob
from typing import Dict, List, Tuple
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from config import TrainingConfig
from utils import (
    find_paired_files, load_paired_embeddings, load_original_graph_data,
    get_user_label, split_train_balanced_val_test_imbalanced, extract_node_id,
    load_gat_embeddings,
)
from data_loader import create_data_loaders
from unified_model import UnifiedTemporalGTLLM
from transformers import AutoTokenizer
import pickle
import matplotlib.pyplot as plt
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def is_interactive_output() -> bool:
    """
    判断当前是否为交互式终端。
    前台运行时一般为 True；
    nohup / 重定向到日志文件时一般为 False。
    """
    try:
        return sys.stdout.isatty() or sys.stderr.isatty()
    except Exception:
        return False


def smart_tqdm(iterable, **kwargs):
    """
    前台运行显示 tqdm；
    后台 nohup 运行时自动关闭 tqdm，避免日志刷屏。
    """
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("disable", not is_interactive_output())
    kwargs.setdefault("mininterval", 5.0)
    return tqdm(iterable, **kwargs)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def threshold_metrics_from_scores(s_gen, labels, theta: float) -> Dict[str, float]:
    s_gen = np.asarray(s_gen, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    pred = (s_gen > theta).astype(np.int32)

    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    tp = int(np.sum((pred == 1) & (labels == 1)))

    accuracy = float((tp + tn) / max(len(labels), 1))
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-12))
    tpr = recall
    fpr = float(fp / (fp + tn + 1e-12))
    youden_j = float(tpr - fpr)

    if len(np.unique(labels)) >= 2:
        roc_auc = float(roc_auc_score(labels, s_gen))
    else:
        roc_auc = 0.5

    return {
        "theta": float(theta),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TPR": float(tpr),
        "FPR": float(fpr),
        "Youden_J": youden_j,
        "roc_auc_gen": roc_auc,
        "y_pred": pred.tolist(),
    }


def compute_recall_at_k_with_counts(scores, labels, k_list=(10, 30, 50)):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    order = np.argsort(-scores)
    total_pos = int(np.sum(labels == 1))

    results = {}
    for k in k_list:
        topk = order[: min(k, len(order))]
        hit = int(np.sum(labels[topk] == 1))
        recall = float(hit / total_pos) if total_pos > 0 else 0.0
        results[f"R@{k}"] = {"hit": hit, "total_pos": total_pos, "recall": recall}
    return results


def summarize_run_results(run_results: List[Dict]) -> Dict:
    summary = {
        "num_runs": len(run_results),
        "runs": run_results,
    }

    metric_map = {
        "test_f1": [r["test_metrics"]["f1"] for r in run_results],
        "test_auc": [r["test_metrics"]["roc_auc_gen"] for r in run_results],
        "test_youden": [r["test_metrics"]["Youden_J"] for r in run_results],
        "test_r10": [r["test_metrics"]["recall_at_k"]["R@10"]["recall"] for r in run_results],
        "test_r30": [r["test_metrics"]["recall_at_k"]["R@30"]["recall"] for r in run_results],
        "test_r50": [r["test_metrics"]["recall_at_k"]["R@50"]["recall"] for r in run_results],
        "best_val_f1": [r.get("best_val_f1", 0.0) for r in run_results],
        "best_epoch_by_f1": [r.get("best_epoch_by_f1", 0) or 0 for r in run_results],
        "time_to_best_epoch_sec": [r.get("time_to_best_epoch_sec", 0.0) for r in run_results],
    }

    aggregate = {}
    for name, values in metric_map.items():
        arr = np.asarray(values, dtype=np.float64)
        aggregate[name] = {
            "mean": float(arr.mean()) if arr.size > 0 else 0.0,
            "std": float(arr.std(ddof=0)) if arr.size > 0 else 0.0,
            "values": [float(x) for x in arr.tolist()],
        }
    summary["aggregate"] = aggregate
    return summary


def write_summary_csv(summary: Dict, csv_path: str):
    header = "metric,mean,std,values\n"
    lines = [header]
    for metric, payload in summary.get("aggregate", {}).items():
        value_str = "|".join(f"{v:.6f}" for v in payload["values"])
        lines.append(f"{metric},{payload['mean']:.6f},{payload['std']:.6f},{value_str}\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def export_best_model_across_runs(run_results: List[Dict], config: TrainingConfig, summary: Dict = None) -> Dict:
    """
    从一次“完整运行”内部的 num_runs 个 run 中，选择总体最优的 1 个 run，
    将其最佳模型额外复制到 selected_best_models_dir 中。

    选择规则：
    1) best_val_f1 最大者优先；
    2) 若并列，则 test_f1 更高者优先；
    3) 若仍并列，则 run_idx 更小者优先。
    """
    candidates = []
    for idx, r in enumerate(run_results):
        model_path = r.get("best_model_path")
        if model_path and os.path.exists(model_path):
            candidates.append((idx, r))

    if not candidates:
        print("⚠️ 没有可导出的最佳模型，跳过 selected_best_models_dir 导出。")
        return {}

    def sort_key(item):
        idx, r = item
        return (
            float(r.get("best_val_f1", -1e9)),
            float(r.get("test_metrics", {}).get("f1", -1e9)),
            -idx,
        )

    best_idx, best_run = max(candidates, key=sort_key)

    selected_dir = config.selected_best_models_dir
    os.makedirs(selected_dir, exist_ok=True)

    tag = str(config.outer_experiment_tag)
    src_model_path = best_run["best_model_path"]
    dst_model_path = os.path.join(selected_dir, f"{tag}_best_model.pth")
    shutil.copy2(src_model_path, dst_model_path)

    # 同时复制一次完整运行对应的数据划分，供后续 inference 直接复用
    split_src = os.path.join(config.output_dir, "dataset_split.json")
    split_dst = os.path.join(selected_dir, "dataset_split.json")
    if os.path.exists(split_src) and not os.path.exists(split_dst):
        shutil.copy2(split_src, split_dst)

    meta = {
        "outer_experiment_tag": tag,
        "experiment_name": config.experiment_name,
        "backbone_init_mode": config.backbone_init_mode,
        "selected_model_path": dst_model_path,
        "source_output_dir": config.output_dir,
        "source_run_idx": int(best_idx + 1),
        "source_run_seed": int(best_run.get("run_seed", -1)),
        "source_best_model_path": src_model_path,
        "selection_metric": "best_val_f1",
        "best_val_f1": float(best_run.get("best_val_f1", 0.0)),
        "best_epoch_by_f1": int(best_run.get("best_epoch_by_f1", 0) or 0),
        "best_val_theta": float(best_run.get("best_val_theta", 0.5) if best_run.get("best_val_theta", None) is not None else 0.5),
        "test_metrics_of_selected_run": best_run.get("test_metrics", {}),
        "all_run_best_val_f1": [float(r.get("best_val_f1", 0.0)) for r in run_results],
        "all_run_test_f1": [float(r.get("test_metrics", {}).get("f1", 0.0)) for r in run_results],
        "all_run_seeds": [int(r.get("run_seed", -1)) for r in run_results],
    }
    if summary is not None:
        meta["this_execution_summary"] = summary.get("aggregate", {})

    meta_path = os.path.join(selected_dir, f"{tag}_meta.json")
    save_json(meta_path, meta)

    # 保存一个简短说明文件，便于后续人工核对
    note_path = os.path.join(selected_dir, f"{tag}_README.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(
            f"outer_experiment_tag={tag}\n"
            f"selected_from_run={best_idx + 1}\n"
            f"selected_seed={best_run.get('run_seed', -1)}\n"
            f"best_val_f1={best_run.get('best_val_f1', 0.0):.6f}\n"
            f"best_val_theta={meta['best_val_theta']:.6f}\n"
            f"selected_model_path={dst_model_path}\n"
        )

    print("\n" + "=" * 80)
    print("已导出本次完整运行的总体最优模型")
    print("=" * 80)
    print(f"  选择自 run_{best_idx + 1} (seed={best_run.get('run_seed', -1)})")
    print(f"  选择标准: best_val_f1 = {best_run.get('best_val_f1', 0.0):.4f}")
    print(f"  导出模型: {dst_model_path}")
    print(f"  导出元信息: {meta_path}")
    print(f"  汇总文件夹: {os.path.abspath(selected_dir)}")

    return meta


def evaluate_best_model_on_test(best_model_path: str, theta: float, test_loader, config, tokenizer) -> Dict:
    """
    使用 best checkpoint 在测试集上评估。
    关键点：
    1) 在构造新模型前先清理显存；
    2) 配合 unified_model.py 中“checkpoint 先加载到 CPU”的 load_model()，避免测试阶段 OOM。
    """
    gc.collect()
    if getattr(config.device, "type", "") == "cuda":
        torch.cuda.empty_cache()

    model = UnifiedTemporalGTLLM(config)
    model.set_tokenizer(tokenizer)
    model.load_model(best_model_path)
    model.to(config.device)
    model.eval()

    s_gen_test, y_test, test_node_ids = compute_scores_for_loader(model, test_loader, config)
    metrics = threshold_metrics_from_scores(s_gen_test, y_test, theta)
    recall_at_k = compute_recall_at_k_with_counts(s_gen_test, y_test, k_list=(10, 30, 50))

    del model
    gc.collect()
    if getattr(config.device, "type", "") == "cuda":
        torch.cuda.empty_cache()

    return {
        **metrics,
        "recall_at_k": recall_at_k,
        "num_test_samples": int(len(y_test)),
        "num_test_pos": int(np.sum(y_test == 1)),
        "num_test_neg": int(np.sum(y_test == 0)),
        "node_ids": test_node_ids,
    }


def compute_scores_for_loader(model, data_loader, config):
    """
    在验证 / 阈值搜索阶段，根据模型输出的生成置信度 s_gen 评估节点。

    与论文一致的实现：
        s_gen = P_LLM("Malicious" | prompt)
    其中 "Malicious" 对应的 token，这里仍然用 '1' 作为恶意类、'0' 作为 Benign 类。
    """
    model.eval()
    device = config.device

    all_s_gen = []
    all_labels = []
    all_node_ids = []

    with torch.no_grad():
        for batch in data_loader:
            struct_in = batch["struct_in_embeddings"].to(device)
            struct_out = batch["struct_out_embeddings"].to(device)
            text_nodes = batch["text_node_embeddings"].to(device)

            struct_in_mask = batch["struct_in_mask"].to(device)
            struct_out_mask = batch["struct_out_mask"].to(device)
            text_mask = batch["text_mask"].to(device)

            timesteps = batch["timesteps"].to(device)        # [B, T]
            time_mask = batch["attention_mask"].to(device)   # [B, T]
            labels = batch["labels"].to(device)              # [B]
            node_ids = batch["node_ids"]

            # 1) 正常前向：包含 soft token + prompt + LLM
            outputs = model(
                struct_in_embeddings=struct_in,
                struct_out_embeddings=struct_out,
                text_node_embeddings=text_nodes,
                struct_in_mask=struct_in_mask,
                struct_out_mask=struct_out_mask,
                text_mask=text_mask,
                timesteps=timesteps,
                attention_mask=time_mask,
                struct_in_embeddings_aug=batch.get("struct_in_embeddings_aug", None).to(device) if "struct_in_embeddings_aug" in batch else None,
                struct_out_embeddings_aug=batch.get("struct_out_embeddings_aug", None).to(device) if "struct_out_embeddings_aug" in batch else None,
                struct_in_mask_aug=batch.get("struct_in_mask_aug", None).to(device) if "struct_in_mask_aug" in batch else None,
                struct_out_mask_aug=batch.get("struct_out_mask_aug", None).to(device) if "struct_out_mask_aug" in batch else None,
                labels=None,     # 推理阶段不算监督 loss
            )

            # 2) 取 LLM 原始 logits，形状 [B, L, vocab]
            logits = outputs["logits"]
            batch_size, seq_len, vocab_size = logits.shape

            # 3) 取最后一个 token 的 logits 作为决策位置
            last_pos = torch.full(
                (batch_size,),
                seq_len - 1,
                device=device,
                dtype=torch.long,
            )
            batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
            last_token_logits = logits[batch_idx, last_pos, :]  # [B, vocab]

            # 4) 使用 '0' 和 '1' 的 token id 作为 Benign / Malicious
            benign_token = model.llm_tokenizer(
                "0", return_tensors="pt", add_special_tokens=False
            )["input_ids"][0][-1].item()
            malicious_token = model.llm_tokenizer(
                "1", return_tensors="pt", add_special_tokens=False
            )["input_ids"][0][-1].item()

            # 5) 只在 {'0','1'} 两个 token 上做 softmax，取 "1" 的概率作为 s_gen
            pair_logits = torch.stack(
                [
                    last_token_logits[:, benign_token],    # [B]
                    last_token_logits[:, malicious_token], # [B]
                ],
                dim=-1,  # [B, 2]
            )
            pair_prob = F.softmax(pair_logits, dim=-1)        # [B, 2]
            p_malicious = pair_prob[:, 1]                     # 概率 P("1"|prompt)

            s_gen_batch = p_malicious.detach().cpu().tolist()

            all_s_gen.extend(s_gen_batch)
            all_labels.extend(labels.detach().cpu().tolist())
            all_node_ids.extend(node_ids)

    return np.array(all_s_gen), np.array(all_labels), all_node_ids


def grid_search_gen_threshold(s_gen, labels):
    """
    基于验证集的生成置信度分数 s_gen，进行单阈值网格搜索。

    说明：
        旧版本使用 Youden's J = TPR - FPR 作为阈值选择准则；
        为了直接优化你关心的 F1（Precision/Recall 折中），这里改为在验证集上选取 F1 最大的阈值。

    输入:
        s_gen   numpy [N]  # 生成置信度分数（越大越倾向 Malicious）
        labels  numpy [N]  # 0/1 标签（0=Benign, 1=Malicious）

    输出:
        best_theta    # 最优置信度阈值（max-F1）
        best_f1       # 对应的 F1
        best_stats    # 混淆矩阵等统计信息（含 precision/recall/f1/tpr/fpr）
    """
    theta_grid = np.linspace(0.0, 1.0, 101)

    best_f1 = -1.0
    best_theta = 0.5
    best_stats = None

    labels = labels.astype(int)
    for th in theta_grid:
        pred = (s_gen > th).astype(int)

        tp = np.sum((pred == 1) & (labels == 1))
        tn = np.sum((pred == 0) & (labels == 0))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))

        # rates
        TPR = tp / (tp + fn + 1e-12)
        FPR = fp / (fp + tn + 1e-12)

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

        # 以 F1 最大为准；若 F1 相同，优先 precision 更高（减少误报）
        if (f1 > best_f1 + 1e-12) or (abs(f1 - best_f1) <= 1e-12 and best_stats is not None and precision > best_stats.get("precision", 0.0) + 1e-12):
            best_f1 = float(f1)
            best_theta = float(th)
            best_stats = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "TPR": float(TPR),
                "FPR": float(FPR),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    return best_theta, best_f1, best_stats

# ===================== Pareto 阈值选择 + 画图（F1 vs J vs θ）=====================

def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def _confusion_from_pred(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def _metrics_from_confusion(tn: int, fp: int, fn: int, tp: int):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # TPR
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    j = recall - fpr  # Youden's J = TPR - FPR
    return precision, recall, fpr, f1, j


def build_pareto_frontier(records):
    """
    Pareto(frontier) for maximizing (f1, j).
    records: list[dict] with keys: theta,f1,j,...
    """
    pareto = []
    for r in records:
        dominated = False
        for q in records:
            if (q["f1"] >= r["f1"] and q["j"] >= r["j"]) and (q["f1"] > r["f1"] or q["j"] > r["j"]):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: float(x["theta"]))
    return pareto


def pareto_search_threshold(s_gen, labels, num_thresholds: int = 201, tie_break: str = "higher_precision"):
    """
    不设 Recall 下限：
      1) 扫描 θ ∈ [0,1]
      2) 构建 Pareto 前沿（max F1 & max J）
      3) 在 Pareto 前沿内选 max-F1 点（并列用 tie-break）

    返回:
      best_rec: dict
      pareto_frontier: list[dict]
      records_all: list[dict]
    """
    s_gen = np.asarray(s_gen, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)

    # 去掉 NaN/Inf
    valid = np.isfinite(s_gen)
    s_gen = s_gen[valid]
    labels = labels[valid]

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    records = []
    for th in thresholds:
        y_pred = (s_gen > th).astype(np.int32)
        tn, fp, fn, tp = _confusion_from_pred(labels, y_pred)
        precision, recall, fpr, f1, j = _metrics_from_confusion(tn, fp, fn, tp)
        records.append({
            "theta": float(th),
            "f1": float(f1),
            "j": float(j),
            "precision": float(precision),
            "recall": float(recall),
            "fpr": float(fpr),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

    pareto = build_pareto_frontier(records)

    # Pareto 内 max-F1 选择
    eps = 1e-12
    best = None
    for r in pareto:
        if best is None:
            best = r
            continue
        if r["f1"] > best["f1"] + eps:
            best = r
        elif abs(r["f1"] - best["f1"]) <= eps:
            if tie_break == "higher_precision":
                if r["precision"] > best["precision"] + eps:
                    best = r
                elif abs(r["precision"] - best["precision"]) <= eps:
                    if r["FP"] < best["FP"]:
                        best = r
                    elif r["FP"] == best["FP"]:
                        if r["theta"] > best["theta"]:
                            best = r
            elif tie_break == "higher_j":
                if r["j"] > best["j"] + eps:
                    best = r
                elif abs(r["j"] - best["j"]) <= eps:
                    if r["theta"] > best["theta"]:
                        best = r
            else:
                if r["theta"] > best["theta"]:
                    best = r

    return best, pareto, records


def choose_validation_threshold(records_all, pareto_frontier, config):
    """Select the classification threshold from validation records."""
    rule = getattr(config, "threshold_selection_rule", "pareto_max_f1")
    target_f1 = float(getattr(config, "target_f1", 0.65))
    baseline_youden = float(getattr(config, "baseline_youden", 0.677000))

    if rule == "val_target_balance":
        return max(
            records_all,
            key=lambda r: (
                min(r["f1"] / target_f1, r["j"] / baseline_youden),
                r["f1"],
                r["j"],
            ),
        )
    if rule == "val_max_f1":
        return max(records_all, key=lambda r: (r["f1"], r["precision"], -r["FP"], r["theta"]))
    if rule == "val_max_j":
        return max(records_all, key=lambda r: (r["j"], r["f1"], r["precision"], r["theta"]))
    if rule == "val_max_j_f1_ge_065":
        eligible = [r for r in records_all if r["f1"] >= target_f1]
        return max(eligible or records_all, key=lambda r: (r["j"], r["f1"], r["precision"], r["theta"]))
    if rule == "val_max_j_f1_ge_070":
        eligible = [r for r in records_all if r["f1"] >= 0.70]
        return max(eligible or records_all, key=lambda r: (r["j"], r["f1"], r["precision"], r["theta"]))
    if rule == "pareto_max_f1":
        return max(pareto_frontier, key=lambda r: (r["f1"], r["precision"], -r["FP"], r["theta"]))

    raise ValueError(f"unknown threshold_selection_rule: {rule}")


def plot_pareto_curves(records_all, pareto_frontier, best_rec, out_dir: str, tag: str):
    """
    画两张图：
      1) 2D: F1 vs J（全体阈值点 + Pareto 前沿 + best）
      2) 3D: θ vs F1 vs J
    不手动指定颜色（使用 matplotlib 默认）。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 2D: F1 vs J
    fig2d = plt.figure()
    ax = fig2d.add_subplot(111)

    f1_all = [r["f1"] for r in records_all]
    j_all = [r["j"] for r in records_all]
    ax.scatter(f1_all, j_all, marker=".", label="All thresholds")

    f1_p = [r["f1"] for r in pareto_frontier]
    j_p = [r["j"] for r in pareto_frontier]
    ax.plot(f1_p, j_p, marker="o", linestyle="-", label="Pareto frontier")

    ax.scatter([best_rec["f1"]], [best_rec["j"]], marker="*", s=200, label="Selected (Pareto max-F1)")

    ax.set_xlabel("F1")
    ax.set_ylabel("Youden's J (TPR - FPR)")
    ax.set_title(f"Pareto (F1 vs J) - {tag}")
    ax.grid(True)
    ax.legend()

    out2d = os.path.join(out_dir, f"pareto_f1_j_{tag}.png")
    fig2d.savefig(out2d, dpi=200, bbox_inches="tight")
    plt.close(fig2d)

    # 3D: theta vs F1 vs J
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig3d = plt.figure()
    ax3 = fig3d.add_subplot(111, projection="3d")

    th_all = [r["theta"] for r in records_all]
    ax3.scatter(th_all, f1_all, j_all, marker=".", label="All thresholds")

    th_p = [r["theta"] for r in pareto_frontier]
    ax3.plot(th_p, f1_p, j_p, marker="o", linestyle="-", label="Pareto frontier")

    ax3.scatter([best_rec["theta"]], [best_rec["f1"]], [best_rec["j"]], marker="*", s=200, label="Selected")

    ax3.set_xlabel("theta")
    ax3.set_ylabel("F1")
    ax3.set_zlabel("Youden's J")
    ax3.set_title(f"Pareto (theta vs F1 vs J) - {tag}")
    ax3.legend()

    out3d = os.path.join(out_dir, f"pareto_theta_f1_j_{tag}.png")
    fig3d.savefig(out3d, dpi=200, bbox_inches="tight")
    plt.close(fig3d)

    return out2d, out3d

# ===================== Pareto 阈值选择 + 画图结束 =====================


def _embedding_dir_pairs(config: TrainingConfig) -> List[Tuple[str, str, str]]:
    pairs = [(config.gat_encoded_dir, config.text_encoded_dir, "primary")]
    for idx, (gat_dir, text_dir) in enumerate(zip(config.extra_gat_encoded_dirs, config.extra_text_encoded_dirs), start=1):
        pairs.append((gat_dir, text_dir, f"extra_normal_{idx}"))
    return pairs


def _normalize_timesteps(temporal_node_embeddings, temporal_text_embeddings, timesteps) -> List[int]:
    if isinstance(timesteps, torch.Tensor):
        timesteps_list = timesteps.view(-1).long().tolist()
    elif isinstance(timesteps, np.ndarray):
        timesteps_list = [int(x) for x in timesteps.reshape(-1).tolist()]
    else:
        timesteps_list = list(timesteps) if timesteps is not None else []

    if len(timesteps_list) == 0:
        ts_keys = set()
        if isinstance(temporal_node_embeddings, dict):
            ts_keys.update(list(temporal_node_embeddings.keys()))
        if isinstance(temporal_text_embeddings, dict):
            ts_keys.update(list(temporal_text_embeddings.keys()))
        timesteps_list = list(ts_keys)

    return sorted(int(t) for t in timesteps_list)


def _load_edge_drop_map(edge_drop_dir: str, map_name: str = "edge-drop") -> Dict[str, Dict]:
    if not edge_drop_dir or not os.path.exists(edge_drop_dir):
        print(f"[EdgeDrop] {map_name} 增强目录不存在: {edge_drop_dir}")
        return {}

    gat_files = glob.glob(os.path.join(edge_drop_dir, "subgraph_*_k2_gat_encoded.pkl"))
    edge_map: Dict[str, Dict] = {}
    duplicate_nodes = 0
    for gat_file in smart_tqdm(gat_files, desc="加载 edge-drop GAT"):
        payload = load_gat_embeddings(gat_file)
        if payload is None:
            continue
        node_id = str(payload.get("node_id", extract_node_id(gat_file)))
        if node_id in edge_map:
            duplicate_nodes += 1
        edge_map[node_id] = payload.get("temporal_node_embeddings", {})

    print(f"[EdgeDrop] 加载 {map_name} 增强结构视图: {len(edge_map)} 个节点, duplicate_nodes={duplicate_nodes}")
    return edge_map


def _edge_drop_lookup_ids(node_id: str, graph_data: Dict) -> List[str]:
    lookup_ids = [str(node_id)]
    if not isinstance(graph_data, dict):
        return lookup_ids

    id_to_user = graph_data.get("id_to_user", {}) or {}
    user_id_to_names = graph_data.get("user_id_to_names", {}) or {}
    screen_name_to_user_id = graph_data.get("screen_name_to_user_id", {}) or {}
    user_to_id = graph_data.get("user_to_id", {}) or {}
    user_to_primary_id = graph_data.get("user_to_primary_id", {}) or {}

    def add_user_name_and_ids(user_name: str):
        if user_name is None:
            return
        user_name = str(user_name)
        lookup_ids.append(user_name)
        for mapping in (screen_name_to_user_id, user_to_id, user_to_primary_id):
            if user_name in mapping:
                lookup_ids.append(str(mapping[user_name]))

    add_user_name_and_ids(str(node_id))

    try:
        node_id_int = int(node_id)
        for names_key in (node_id, node_id_int):
            if names_key in user_id_to_names:
                for user_name in user_id_to_names[names_key]:
                    add_user_name_and_ids(user_name)
        if node_id_int in id_to_user:
            add_user_name_and_ids(id_to_user[node_id_int])
    except Exception:
        if node_id in user_id_to_names:
            for user_name in user_id_to_names[node_id]:
                add_user_name_and_ids(user_name)
        pass

    seen = set()
    deduped = []
    for value in lookup_ids:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _load_edge_drop_for_node(edge_drop_dir: str, node_id: str, graph_data: Dict) -> Dict:
    if not edge_drop_dir or not os.path.exists(edge_drop_dir):
        return {}
    for lookup_id in _edge_drop_lookup_ids(str(node_id), graph_data):
        edge_path = os.path.join(edge_drop_dir, f"subgraph_{lookup_id}_k2_gat_encoded.pkl")
        if not os.path.exists(edge_path):
            continue
        payload = load_gat_embeddings(edge_path)
        if payload is None:
            continue
        return payload.get("temporal_node_embeddings", {}) or {}
    return {}


def load_all_data(config: TrainingConfig) -> Tuple[Dict, Dict, Dict]:
    """加载主 embedding + extra normal embedding，并挂载 edge-drop 结构视图。"""
    print("=" * 60)
    print("加载数据...(主目录 + normal_42000 + edge-drop 结构视图)")
    print("=" * 60)

    graph_data = load_original_graph_data(config.original_graph_data_path)
    lazy_edge_drop = bool(getattr(config, "lazy_load_edge_drop", False))
    if lazy_edge_drop:
        print("[EdgeDrop] 使用按节点懒加载，避免全量读取 edge-drop 目录。")
        shared_edge_drop_map = {}
        primary_edge_drop_map = {}
    else:
        shared_edge_drop_map = _load_edge_drop_map(getattr(config, "edge_drop_gat_encoded_dir", ""), "shared")
        primary_edge_drop_map = _load_edge_drop_map(
            getattr(config, "primary_edge_drop_gat_encoded_dir", ""),
            "primary_patch",
        )

    node_data: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}
    source_stats: Dict[str, Dict[str, int]] = {}

    for gat_dir, text_dir, source_name in _embedding_dir_pairs(config):
        paired_files = find_paired_files(gat_dir, text_dir)
        max_extra = getattr(config, "max_extra_normal_samples", None)
        if source_name.startswith("extra_normal") and max_extra is not None:
            max_extra = max(0, int(max_extra))
            if len(paired_files) > max_extra:
                print(f"[DataLoad] {source_name} 仅加载前 {max_extra}/{len(paired_files)} 对，用于满足 1:50 评估比例。")
                paired_files = paired_files[:max_extra]
        source_stats[source_name] = {"paired": len(paired_files), "loaded": 0, "duplicates_skipped": 0}

        for gat_file, text_file in smart_tqdm(paired_files, desc=f"加载节点数据:{source_name}"):
            embeddings = load_paired_embeddings(gat_file, text_file)
            if embeddings is None:
                continue

            node_id = embeddings.get("node_id", None)
            if node_id is None:
                continue
            node_id = str(node_id)

            if node_id in node_data:
                source_stats[source_name]["duplicates_skipped"] += 1
                continue

            has_label, label = get_user_label(node_id, graph_data)
            if not has_label:
                continue

            temporal_node_embeddings = embeddings.get("temporal_node_embeddings", {})
            temporal_text_embeddings = embeddings.get("temporal_text_embeddings", {})
            timesteps_sorted = _normalize_timesteps(
                temporal_node_embeddings,
                temporal_text_embeddings,
                embeddings.get("timesteps", []),
            )

            if len(timesteps_sorted) == 0:
                continue
            if (not isinstance(temporal_node_embeddings, dict) or len(temporal_node_embeddings) == 0) and \
               (not isinstance(temporal_text_embeddings, dict) or len(temporal_text_embeddings) == 0):
                continue

            item = {
                "temporal_node_embeddings": temporal_node_embeddings,
                "temporal_text_embeddings": temporal_text_embeddings,
                "timesteps": timesteps_sorted,
                "embedding_source": source_name,
            }
            if lazy_edge_drop:
                if source_name == "primary":
                    aug = _load_edge_drop_for_node(getattr(config, "primary_edge_drop_gat_encoded_dir", ""), node_id, graph_data)
                    if not aug:
                        aug = _load_edge_drop_for_node(getattr(config, "edge_drop_gat_encoded_dir", ""), node_id, graph_data)
                else:
                    aug = _load_edge_drop_for_node(getattr(config, "edge_drop_gat_encoded_dir", ""), node_id, graph_data)
                if aug:
                    item["temporal_node_embeddings_aug"] = aug
            else:
                if source_name == "primary" and node_id in primary_edge_drop_map:
                    item["temporal_node_embeddings_aug"] = primary_edge_drop_map[node_id]
                elif node_id in shared_edge_drop_map:
                    item["temporal_node_embeddings_aug"] = shared_edge_drop_map[node_id]

            node_data[node_id] = item
            labels[node_id] = int(label)
            source_stats[source_name]["loaded"] += 1

    edge_aug_count = sum(1 for v in node_data.values() if "temporal_node_embeddings_aug" in v)
    print(f"数据加载完成: {len(node_data)} 个节点")
    print(f"标签分布: 恶意={sum(labels.values())}, 正常={len(labels) - sum(labels.values())}")
    print(f"edge-drop 增强可用: {edge_aug_count}/{len(node_data)}")
    print(f"来源统计: {source_stats}")

    return node_data, labels, graph_data


def train_one_epoch(model, train_loader, optimizer, scheduler, config, epoch, fold_idx):
    model.train()
    device = config.device

    total_loss = 0.0
    total_align_loss = 0.0
    total_gen_loss = 0.0
    total_cee_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_node_ids = []

    progress_bar = smart_tqdm(train_loader, desc=f"Train Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
        struct_in = batch["struct_in_embeddings"].to(device)
        struct_out = batch["struct_out_embeddings"].to(device)
        text_nodes = batch["text_node_embeddings"].to(device)

        struct_in_mask = batch["struct_in_mask"].to(device)       # [B, T, N_in]
        struct_out_mask = batch["struct_out_mask"].to(device)     # [B, T, N_out]
        text_mask = batch["text_mask"].to(device)                 # [B, T, N_txt]

        timesteps = batch["timesteps"].to(device)                 # [B, T]
        time_mask = batch["attention_mask"].to(device)            # [B, T]
        labels = batch["labels"].to(device)                       # [B]
        node_ids = batch["node_ids"]                              # list[str] 或 list[int]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            struct_in_embeddings=struct_in,
            struct_out_embeddings=struct_out,
            text_node_embeddings=text_nodes,
            struct_in_mask=struct_in_mask,
            struct_out_mask=struct_out_mask,
            text_mask=text_mask,
            timesteps=timesteps,
            attention_mask=time_mask,
            struct_in_embeddings_aug=batch.get("struct_in_embeddings_aug", None).to(device) if "struct_in_embeddings_aug" in batch else None,
            struct_out_embeddings_aug=batch.get("struct_out_embeddings_aug", None).to(device) if "struct_out_embeddings_aug" in batch else None,
            struct_in_mask_aug=batch.get("struct_in_mask_aug", None).to(device) if "struct_in_mask_aug" in batch else None,
            struct_out_mask_aug=batch.get("struct_out_mask_aug", None).to(device) if "struct_out_mask_aug" in batch else None,
            labels=labels,
        )

        loss = outputs["total_loss"]
        align_loss = outputs.get("align_loss", None)
        gen_loss = outputs.get("gen_loss", None)
        cee_loss = outputs.get("cee_loss", None)

        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        if align_loss is not None:
            total_align_loss += align_loss.item() * batch_size
        if gen_loss is not None:
            total_gen_loss += gen_loss.item() * batch_size
        if cee_loss is not None:
            total_cee_loss += cee_loss.item() * batch_size
        total_samples += batch_size

        with torch.no_grad():
            class_logits = outputs["logits_2"]  # [B, 2]
            preds = class_logits.argmax(dim=-1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_node_ids.extend(node_ids)

        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_align = total_align_loss / total_samples
            avg_gen = total_gen_loss / total_samples
            avg_cee = total_cee_loss / total_samples
            tmp_acc = (np.array(all_preds, dtype=np.int32) ==
                       np.array(all_labels, dtype=np.int32)).mean()
        else:
            avg_loss = avg_align = avg_gen = avg_cee = 0.0
            tmp_acc = 0.0

        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            align=f"{avg_align:.4f}",
            gen=f"{avg_gen:.4f}",
            cee=f"{avg_cee:.4f}",
            acc=f"{tmp_acc:.4f}",
        )

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_align = total_align_loss / total_samples
        avg_gen = total_gen_loss / total_samples
        avg_cee = total_cee_loss / total_samples
    else:
        avg_loss = avg_align = avg_gen = avg_cee = 0.0

    if len(all_labels) > 0:
        y_true = np.array(all_labels, dtype=np.int32)
        y_pred = np.array(all_preds, dtype=np.int32)
        acc = (y_true == y_pred).mean()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        acc = 0.0
        tn = fp = fn = tp = 0

    return {
        "train_loss": float(avg_loss),
        "train_align_loss": float(avg_align),
        "train_gen_loss": float(avg_gen),
        "train_cee_loss": float(avg_cee),
        "train_accuracy": float(acc),
        "train_tn": int(tn),
        "train_fp": int(fp),
        "train_fn": int(fn),
        "train_tp": int(tp),
        "train_predictions": all_preds,
        "train_labels": all_labels,
        "train_node_ids": all_node_ids,
    }


def validate(model, val_loader, config, tokenizer, epoch, fold_idx):
    model.eval()
    device = config.device

    total_loss = 0.0
    total_align_loss = 0.0
    total_gen_loss = 0.0
    total_cee_loss = 0.0
    total_samples = 0

    # ===== 新增：按类别统计 gen loss（label==1: 恶意, label==0: 正常）=====
    pos_gen_loss_sum = 0.0
    neg_gen_loss_sum = 0.0
    pos_count = 0
    neg_count = 0
    # ===============================================================

    all_preds = []
    all_labels = []
    all_node_ids = []

    progress_bar = smart_tqdm(val_loader, desc=f"Val Epoch {epoch+1}")

    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
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
                struct_in_embeddings_aug=batch.get("struct_in_embeddings_aug", None).to(device) if "struct_in_embeddings_aug" in batch else None,
                struct_out_embeddings_aug=batch.get("struct_out_embeddings_aug", None).to(device) if "struct_out_embeddings_aug" in batch else None,
                struct_in_mask_aug=batch.get("struct_in_mask_aug", None).to(device) if "struct_in_mask_aug" in batch else None,
                struct_out_mask_aug=batch.get("struct_out_mask_aug", None).to(device) if "struct_out_mask_aug" in batch else None,
                labels=labels,
            )

            loss = outputs["total_loss"]
            align_loss = outputs.get("align_loss", None)
            gen_loss = outputs.get("gen_loss", None)
            cee_loss = outputs.get("cee_loss", None)

            if loss is None:
                continue

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            if align_loss is not None:
                total_align_loss += align_loss.item() * batch_size
            if cee_loss is not None:
                total_cee_loss += cee_loss.item() * batch_size
            total_samples += batch_size

            # ===== 关键改动：不要用 outputs["gen_loss"]（它是 batch mean）
            # 需要 per-sample gen loss 才能按类别统计 =====
            class_logits = outputs["logits_2"]  # [B, 2]

            per_sample_gen = F.cross_entropy(
                class_logits,
                labels.long(),
                reduction="none"
            )  # [B]

            # 总 gen loss 用 per-sample 求和，再除以总样本数
            total_gen_loss += per_sample_gen.sum().item()

            pos_mask = (labels == 1)
            neg_mask = (labels == 0)

            if pos_mask.any():
                pos_gen_loss_sum += per_sample_gen[pos_mask].sum().item()
                pos_count += int(pos_mask.sum().item())

            if neg_mask.any():
                neg_gen_loss_sum += per_sample_gen[neg_mask].sum().item()
                neg_count += int(neg_mask.sum().item())
            # ===== 关键改动结束 =====

            preds = class_logits.argmax(dim=-1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_node_ids.extend(node_ids)

            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_align = total_align_loss / total_samples
                avg_gen = total_gen_loss / total_samples
                avg_cee = total_cee_loss / total_samples
                tmp_acc = (np.array(all_preds, dtype=np.int32) ==
                           np.array(all_labels, dtype=np.int32)).mean()
            else:
                avg_loss = avg_align = avg_gen = avg_cee = 0.0
                tmp_acc = 0.0

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                align=f"{avg_align:.4f}",
                gen=f"{avg_gen:.4f}",
                cee=f"{avg_cee:.4f}",
                acc=f"{tmp_acc:.4f}",
            )

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_align = total_align_loss / total_samples
        avg_gen = total_gen_loss / total_samples
        avg_cee = total_cee_loss / total_samples

        # ===== 新增：按类别平均 =====
        avg_gen_pos = (pos_gen_loss_sum / pos_count) if pos_count > 0 else 0.0
        avg_gen_neg = (neg_gen_loss_sum / neg_count) if neg_count > 0 else 0.0
    else:
        avg_loss = avg_align = avg_gen = avg_cee = 0.0
        avg_gen_pos = 0.0
        avg_gen_neg = 0.0

    if len(all_labels) > 0:
        y_true = np.array(all_labels, dtype=np.int32)
        y_pred = np.array(all_preds, dtype=np.int32)
        acc = (y_true == y_pred).mean()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        acc = 0.0
        tn = fp = fn = tp = 0

    return {
        "val_loss": float(avg_loss),
        "val_align_loss": float(avg_align),
        "val_gen_loss": float(avg_gen),

        # ===== 新增返回字段：按类别 gen loss + 数量 =====
        "val_gen_loss_pos": float(avg_gen_pos),
        "val_gen_loss_neg": float(avg_gen_neg),
        "val_pos_count": int(pos_count),
        "val_neg_count": int(neg_count),
        # ============================================

        "val_cee_loss": float(avg_cee),
        "val_accuracy": float(acc),
        "val_tn": int(tn),
        "val_fp": int(fp),
        "val_fn": int(fn),
        "val_tp": int(tp),
        "val_predictions": all_preds,
        "val_labels": all_labels,
        "val_node_ids": all_node_ids,
    }


def save_epoch_metrics(fold_idx: int, epoch: int, train_metrics: Dict,
                       val_metrics: Dict, output_dir: str):
    """保存每个epoch的训练集和测试集指标到文件，增加 CEE"""

    fold_dir = os.path.join(output_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_dir, exist_ok=True)

    epoch_dir = os.path.join(fold_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)

    train_metrics_path = os.path.join(epoch_dir, "train_metrics.json")
    train_metrics_to_save = {
        'fold': fold_idx + 1,
        'epoch': epoch + 1,
        'loss': train_metrics['train_loss'],
        'gen_loss': train_metrics['train_gen_loss'],
        'align_loss': train_metrics['train_align_loss'],
        'cee_loss': train_metrics.get('train_cee_loss', 0.0),
        'accuracy': train_metrics['train_accuracy'],
        'confusion_matrix': {
            'TN': train_metrics.get('train_tn', 0),
            'FP': train_metrics.get('train_fp', 0),
            'FN': train_metrics.get('train_fn', 0),
            'TP': train_metrics.get('train_tp', 0)
        },
        'predictions': train_metrics.get('train_predictions', []),
        'labels': train_metrics.get('train_labels', []),
        'node_ids': train_metrics.get('train_node_ids', []),
        'timestamp': datetime.datetime.now().isoformat()
    }

    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics_to_save, f, indent=2, ensure_ascii=False)

    val_metrics_path = os.path.join(epoch_dir, "val_metrics.json")
    val_metrics_to_save = {
        'fold': fold_idx + 1,
        'epoch': epoch + 1,
        'loss': val_metrics['val_loss'],
        'gen_loss': val_metrics['val_gen_loss'],
        'gen_loss_pos': val_metrics.get('val_gen_loss_pos', 0.0),
        'gen_loss_neg': val_metrics.get('val_gen_loss_neg', 0.0),
        'pos_count': val_metrics.get('val_pos_count', 0),
        'neg_count': val_metrics.get('val_neg_count', 0),
        'align_loss': val_metrics['val_align_loss'],
        'cee_loss': val_metrics.get('val_cee_loss', 0.0),
        'accuracy': val_metrics['val_accuracy'],
        'confusion_matrix': {
            'TN': val_metrics.get('val_tn', 0),
            'FP': val_metrics.get('val_fp', 0),
            'FN': val_metrics.get('val_fn', 0),
            'TP': val_metrics.get('val_tp', 0)
        },
        'predictions': val_metrics.get('val_predictions', []),
        'labels': val_metrics.get('val_labels', []),
        'node_ids': val_metrics.get('val_node_ids', []),
        'timestamp': datetime.datetime.now().isoformat()

    }

    with open(val_metrics_path, 'w') as f:
        json.dump(val_metrics_to_save, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(epoch_dir, "summary_metrics.json")

    summary = {
        'fold': fold_idx + 1,
        'epoch': epoch + 1,
        'train': {
            'loss': train_metrics['train_loss'],
            'gen_loss': train_metrics['train_gen_loss'],
            'align_loss': train_metrics['train_align_loss'],
            'cee_loss': train_metrics.get('train_cee_loss', 0.0),
            'accuracy': train_metrics['train_accuracy'],
            'confusion_matrix': {
                'TN': train_metrics.get('train_tn', 0),
                'FP': train_metrics.get('train_fp', 0),
                'FN': train_metrics.get('train_fn', 0),
                'TP': train_metrics.get('train_tp', 0)
            }
        },
        'validation': {
            'loss': val_metrics['val_loss'],
            'gen_loss': val_metrics['val_gen_loss'],
            'gen_loss_pos': val_metrics.get('val_gen_loss_pos', 0.0),
            'gen_loss_neg': val_metrics.get('val_gen_loss_neg', 0.0),
            'pos_count': val_metrics.get('val_pos_count', 0),
            'neg_count': val_metrics.get('val_neg_count', 0),
            'align_loss': val_metrics['val_align_loss'],
            'cee_loss': val_metrics.get('val_cee_loss', 0.0),
            'accuracy': val_metrics['val_accuracy'],
            'confusion_matrix': {
                'TN': val_metrics.get('val_tn', 0),
                'FP': val_metrics.get('val_fp', 0),
                'FN': val_metrics.get('val_fn', 0),
                'TP': val_metrics.get('val_tp', 0)
            }
        },
        'timestamp': datetime.datetime.now().isoformat()
    }

    train_tn = train_metrics.get('train_tn', 0)
    train_fp = train_metrics.get('train_fp', 0)
    train_fn = train_metrics.get('train_fn', 0)
    train_tp = train_metrics.get('train_tp', 0)

    if train_tp + train_fp > 0:
        summary['train']['precision'] = train_tp / (train_tp + train_fp)
    if train_tp + train_fn > 0:
        summary['train']['recall'] = train_tp / (train_tp + train_fn)
    if train_tn + train_fp > 0:
        summary['train']['specificity'] = train_tn / (train_tn + train_fp)

    val_tn = val_metrics.get('val_tn', 0)
    val_fp = val_metrics.get('val_fp', 0)
    val_fn = val_metrics.get('val_fn', 0)
    val_tp = val_metrics.get('val_tp', 0)

    if val_tp + val_fp > 0:
        summary['validation']['precision'] = val_tp / (val_tp + val_fp)
    if val_tp + val_fn > 0:
        summary['validation']['recall'] = val_tp / (val_tp + val_fn)
    if val_tn + val_fp > 0:
        summary['validation']['specificity'] = val_tn / (val_tn + val_fp)

    if 'precision' in summary['train'] and 'recall' in summary['train']:
        train_precision = summary['train']['precision']
        train_recall = summary['train']['recall']
        if train_precision + train_recall > 0:
            summary['train']['f1_score'] = 2 * train_precision * train_recall / (train_precision + train_recall)

    if 'precision' in summary['validation'] and 'recall' in summary['validation']:
        val_precision = summary['validation']['precision']
        val_recall = summary['validation']['recall']
        if val_precision + val_recall > 0:
            summary['validation']['f1_score'] = 2 * val_precision * val_recall / (val_precision + val_recall)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  指标保存到: {epoch_dir}/")
    return train_metrics_path, val_metrics_path, summary_path


def plot_losses(train_history: List[Dict], fold_output_dir: str, fold_idx: int):
    """
    为当前 run 画出：
      - 生成损失 (train / val)
      - 对齐损失 (train / val)
      - CEE 损失 (train / val)
    """
    if not train_history:
        return

    epochs = list(range(1, len(train_history) + 1))

    # 1) 生成损失
    train_gen = [h.get("train_gen_loss", 0.0) for h in train_history]
    val_gen = [h.get("val_gen_loss", 0.0) for h in train_history]

    plt.figure()
    plt.plot(epochs, train_gen, marker="o", label="Train Gen Loss")
    plt.plot(epochs, val_gen, marker="s", label="Val Gen Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Generation Loss")
    plt.title(f"Run {fold_idx+1} Generation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    gen_fig_path = os.path.join(fold_output_dir, f"run_{fold_idx+1}_gen_loss.png")
    plt.savefig(gen_fig_path, dpi=200)
    plt.close()
    print(f"[可视化] Generation Loss 曲线已保存到: {gen_fig_path}")

    # 2) 对齐损失
    train_align = [h.get("train_align_loss", 0.0) for h in train_history]
    val_align = [h.get("val_align_loss", 0.0) for h in train_history]

    plt.figure()
    plt.plot(epochs, train_align, marker="o", label="Train Align Loss")
    plt.plot(epochs, val_align, marker="s", label="Val Align Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Loss")
    plt.title(f"Run {fold_idx+1} Alignment Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    align_fig_path = os.path.join(fold_output_dir, f"run_{fold_idx+1}_align_loss.png")
    plt.savefig(align_fig_path, dpi=200)
    plt.close()
    print(f"[可视化] Alignment Loss 曲线已保存到: {align_fig_path}")

    # 3) CEE 损失
    train_cee = [h.get("train_cee_loss", 0.0) for h in train_history]
    val_cee = [h.get("val_cee_loss", 0.0) for h in train_history]

    plt.figure()
    plt.plot(epochs, train_cee, marker="o", label="Train CEE Loss")
    plt.plot(epochs, val_cee, marker="s", label="Val CEE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CEE Loss")
    plt.title(f"Run {fold_idx+1} CEE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    cee_fig_path = os.path.join(fold_output_dir, f"run_{fold_idx+1}_cee_loss.png")
    plt.savefig(cee_fig_path, dpi=200)
    plt.close()
    print(f"[可视化] CEE Loss 曲线已保存到: {cee_fig_path}")


def split_class_counts(node_ids: List[str], labels: Dict[str, int]) -> Dict[str, float]:
    total = len(node_ids)
    malicious = int(sum(1 for nid in node_ids if int(labels.get(nid, 0)) == 1))
    normal = int(total - malicious)
    return {
        "total": int(total),
        "malicious": malicious,
        "normal": normal,
        "normal_per_malicious": float(normal / malicious) if malicious > 0 else None,
    }


def build_split_payload(
    train_node_ids: List[str],
    val_node_ids: List[str],
    test_node_ids: List[str],
    labels: Dict[str, int],
    config: TrainingConfig,
) -> Dict:
    return {
        "train_node_ids": train_node_ids,
        "val_node_ids": val_node_ids,
        "test_node_ids": test_node_ids,
        "split_seed": getattr(config, "split_seed", 42),
        "val_test_neg_pos_ratio": getattr(config, "val_test_neg_pos_ratio", None),
        "class_counts": {
            "train": split_class_counts(train_node_ids, labels),
            "val": split_class_counts(val_node_ids, labels),
            "test": split_class_counts(test_node_ids, labels),
        },
    }


def assert_split_protocol(payload: Dict, labels: Dict[str, int], expected_ratio) -> None:
    counts = payload.get("class_counts") or {
        "train": split_class_counts(payload.get("train_node_ids", []), labels),
        "val": split_class_counts(payload.get("val_node_ids", []), labels),
        "test": split_class_counts(payload.get("test_node_ids", []), labels),
    }

    train = counts["train"]
    if train["malicious"] != train["normal"]:
        raise RuntimeError(f"训练集不是 1:1: {train}")

    if expected_ratio is not None:
        ratio = int(expected_ratio)
        for split_name in ("val", "test"):
            c = counts[split_name]
            if c["normal"] != c["malicious"] * ratio:
                raise RuntimeError(f"{split_name} 不是恶意:正常=1:{ratio}: {c}")


def can_reuse_split(split_data: Dict, config: TrainingConfig, labels: Dict[str, int], node_data: Dict) -> bool:
    expected_seed = getattr(config, "split_seed", 42)
    expected_ratio = getattr(config, "val_test_neg_pos_ratio", None)
    if split_data.get("split_seed", None) != expected_seed:
        print(f"已有划分 split_seed={split_data.get('split_seed', None)} 与当前配置 {expected_seed} 不一致，将重新划分。")
        return False
    if split_data.get("val_test_neg_pos_ratio", None) != expected_ratio:
        print(f"已有划分 val_test_neg_pos_ratio={split_data.get('val_test_neg_pos_ratio', None)} 与当前配置 {expected_ratio} 不一致，将重新划分。")
        return False

    all_ids = split_data.get("train_node_ids", []) + split_data.get("val_node_ids", []) + split_data.get("test_node_ids", [])
    missing_ids = [nid for nid in all_ids if nid not in node_data or nid not in labels]
    if missing_ids:
        print(f"已有划分中有 {len(missing_ids)} 个节点当前未加载，将重新划分。")
        return False

    try:
        payload = build_split_payload(
            split_data.get("train_node_ids", []),
            split_data.get("val_node_ids", []),
            split_data.get("test_node_ids", []),
            labels,
            config,
        )
        assert_split_protocol(payload, labels, expected_ratio)
    except RuntimeError as exc:
        print(f"已有划分比例不满足当前协议，将重新划分: {exc}")
        return False

    return True


def verify_training_assets(train_node_ids: List[str], node_data: Dict, config: TrainingConfig) -> None:
    if float(getattr(config, "lambda_align", 0.0)) > 0.0 and bool(getattr(config, "require_edge_drop_alignment", True)):
        missing_aug = [nid for nid in train_node_ids if "temporal_node_embeddings_aug" not in node_data.get(nid, {})]
        if missing_aug:
            preview = missing_aug[:10]
            raise RuntimeError(
                "训练集缺少 edge-drop GAT 增强视图，不能启动 L_cont。"
                f" missing={len(missing_aug)}, preview={preview}. "
                "先用 gat_encoder.py 生成 gat_encoded_subgraphs_edge_drop。"
            )

    cee_path = getattr(config, "cee_head_path", "")
    if cee_path and not os.path.isabs(cee_path):
        cee_path = os.path.abspath(cee_path)
    if bool(getattr(config, "require_pretrained_cee", True)) and not os.path.exists(cee_path):
        raise RuntimeError(
            f"缺少冻结 CEE dynamics checkpoint: {cee_path}. "
            "先运行 python pretrain_cee_head.py，再运行 python train.py。"
        )


def train_one_fold(model: UnifiedTemporalGTLLM, train_loader, val_loader,
                   config: TrainingConfig, fold_idx: int, output_dir: str, tokenizer=None) -> Dict[str, float]:
    """
    单次训练 run（保留函数名与原始结构，但不再进行 K 折，只会调用一次）
    """
    print(f"\n{'='*60}")
    print(f"开始单次训练（Run {fold_idx+1}）")
    print(f"{'='*60}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=1e-5)

    total_steps = len(train_loader) * config.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    train_history = []
    best_val_loss = float('inf')
    best_model_path = None

    # ===== 阈值选择：在验证集上用 max-F1 作为 best model 选择标准 =====
    best_val_f1 = -1.0
    best_val_theta = None
    best_val_stats = None
    best_epoch_idx = None
    best_time_to_epoch_sec = None
    # ======================================================

    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    cumulative_time_sec = 0.0

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, config, epoch, fold_idx)
        val_metrics = validate(model, val_loader, config, tokenizer, epoch, fold_idx)

        train_metrics_path, val_metrics_path, summary_path = save_epoch_metrics(
            fold_idx, epoch, train_metrics, val_metrics, output_dir
        )

        epoch_time_sec = float(time.time() - epoch_start_time)
        cumulative_time_sec += epoch_time_sec
        epoch_metrics = {**train_metrics, **val_metrics, "epoch_time_sec": epoch_time_sec, "cumulative_time_sec": cumulative_time_sec}
        train_history.append(epoch_metrics)

        print(f"  训练损失: {train_metrics['train_loss']:.4f} "
              f"(生成: {train_metrics['train_gen_loss']:.4f}, 对齐: {train_metrics['train_align_loss']:.4f}, CEE: {train_metrics['train_cee_loss']:.4f})")
        print(f"  验证损失: {val_metrics['val_loss']:.4f} "
              f"(生成: {val_metrics['val_gen_loss']:.4f}, "
              f"生成-pos: {val_metrics.get('val_gen_loss_pos', 0.0):.4f}, "
              f"生成-neg: {val_metrics.get('val_gen_loss_neg', 0.0):.4f}, "
              f"对齐: {val_metrics['val_align_loss']:.4f}, CEE: {val_metrics['val_cee_loss']:.4f})")

        print(f"  [Val counts] pos={val_metrics.get('val_pos_count', 0)} "
              f"neg={val_metrics.get('val_neg_count', 0)}")

        # 训练集混淆矩阵
        tr_tn = train_metrics.get("train_tn", 0)
        tr_fp = train_metrics.get("train_fp", 0)
        tr_fn = train_metrics.get("train_fn", 0)
        tr_tp = train_metrics.get("train_tp", 0)
        tr_acc = train_metrics.get("train_accuracy", 0.0)

        tr_precision = tr_tp / (tr_tp + tr_fp) if (tr_tp + tr_fp) > 0 else 0.0
        tr_recall = tr_tp / (tr_tp + tr_fn) if (tr_tp + tr_fn) > 0 else 0.0
        tr_f1 = (2 * tr_precision * tr_recall / (tr_precision + tr_recall)) if (tr_precision + tr_recall) > 0 else 0.0

        print("  [训练集混淆矩阵]")
        print("           预测0    预测1")
        print(f"真实0    {tr_tn:8d}{tr_fp:8d}")
        print(f"真实1    {tr_fn:8d}{tr_tp:8d}")
        print(f"    训练集 Accuracy : {tr_acc:.4f}")
        print(f"    训练集 Precision: {tr_precision:.4f}  Recall: {tr_recall:.4f}  F1: {tr_f1:.4f}")

        # 验证集混淆矩阵
        val_tn = val_metrics.get("val_tn", 0)
        val_fp = val_metrics.get("val_fp", 0)
        val_fn = val_metrics.get("val_fn", 0)
        val_tp = val_metrics.get("val_tp", 0)
        val_acc = val_metrics.get("val_accuracy", 0.0)

        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        val_f1 = (2 * val_precision * val_recall / (val_precision + val_recall)) if (val_precision + val_recall) > 0 else 0.0

        print("  [验证集混淆矩阵]")
        print("           预测0    预测1")
        print(f"真实0    {val_tn:8d}{val_fp:8d}")
        print(f"真实1    {val_fn:8d}{val_tp:8d}")
        print(f"    验证集 Accuracy : {val_acc:.4f}")
        print(f"    验证集 Precision: {val_precision:.4f}  Recall: {val_recall:.4f}  F1: {val_f1:.4f}")

        print("  [Val threshold Pareto] 基于 s_gen 扫描 θ，构建 Pareto(F1 vs J vs θ)，并在 Pareto 内选 max-F1 的 θ ...")
        s_gen_val_epoch, y_val_epoch, _ = compute_scores_for_loader(model, val_loader, config)

        best_rec, pareto_frontier, records_all = pareto_search_threshold(
            s_gen_val_epoch, y_val_epoch, num_thresholds=201, tie_break="higher_precision"
        )

        epoch_theta = float(best_rec["theta"])
        epoch_f1 = float(best_rec["f1"])

        # 保持你原 epoch_stats 输出风格（并补充 J / pareto_size）
        epoch_stats = {
            "tn": int(best_rec["TN"]),
            "fp": int(best_rec["FP"]),
            "fn": int(best_rec["FN"]),
            "tp": int(best_rec["TP"]),
            "TPR": float(best_rec["recall"]),
            "FPR": float(best_rec["fpr"]),
            "precision": float(best_rec["precision"]),
            "recall": float(best_rec["recall"]),
            "f1": float(best_rec["f1"]),
            "j": float(best_rec["j"]),
            "pareto_size": int(len(pareto_frontier)),
        }

        print(f"    epoch_pareto_theta = {epoch_theta:.4f}, epoch_pareto_F1 = {epoch_f1:.4f}, "
            f"epoch_pareto_J = {epoch_stats['j']:.4f}, stats = {epoch_stats}")

        # ===== 新增：画 Pareto 曲线（每个 epoch 一套图）=====
        pareto_plot_dir = os.path.join(fold_output_dir, "pareto_plots")
        tag = f"epoch{epoch+1:03d}"
        out2d, out3d = plot_pareto_curves(records_all, pareto_frontier, best_rec, pareto_plot_dir, tag=tag)
        # =====================================================


        # 保存最佳模型（按验证集 max-F1 最大为准）
        if epoch_f1 > best_val_f1:
            best_val_f1 = epoch_f1
            best_val_theta = epoch_theta
            best_val_stats = epoch_stats
            best_epoch_idx = epoch + 1
            best_time_to_epoch_sec = cumulative_time_sec

            # 同步记录该 epoch 的 val_loss（仅用于最终打印展示）
            best_val_loss = val_metrics['val_loss']

            best_model_path = os.path.join(fold_output_dir, f"fold_{fold_idx+1}_best_model.pth")
            model.save_model(best_model_path)
            print(f"  ✅ 保存最佳模型(按 val max-F1): {best_model_path} (best_F1={best_val_f1:.4f}, epoch={best_epoch_idx})")

            best_epoch_dir = os.path.join(fold_output_dir, "best_epoch")
            os.makedirs(best_epoch_dir, exist_ok=True)

            import shutil
            best_epoch_source_dir = os.path.join(output_dir, f"fold_{fold_idx+1}", f"epoch_{epoch+1}")
            if os.path.exists(best_epoch_source_dir):
                for file in os.listdir(best_epoch_source_dir):
                    source_file = os.path.join(best_epoch_source_dir, file)
                    dest_file = os.path.join(best_epoch_dir, file)
                    shutil.copy2(source_file, dest_file)

        # 周期性 checkpoint
        if (epoch + 1) % config.save_frequency == 0:
            checkpoint_path = os.path.join(fold_output_dir, f"epoch_{epoch+1}_checkpoint.pth")
            model.save_model(checkpoint_path)
            print(f"  保存检查点: {checkpoint_path}")

    # 保存 training_history
    history_path = os.path.join(fold_output_dir, "training_history.json")

    simplified_history = []
    for epoch_data in train_history:
        simplified = {
            'epoch': len(simplified_history) + 1,
            'train_loss': epoch_data.get('train_loss', 0),
            'train_gen_loss': epoch_data.get('train_gen_loss', 0.0),
            'train_align_loss': epoch_data.get('train_align_loss', 0.0),
            'train_cee_loss': epoch_data.get('train_cee_loss', 0.0),
            'train_accuracy': epoch_data.get('train_accuracy', 0),
            'train_tn': epoch_data.get('train_tn', 0),
            'train_fp': epoch_data.get('train_fp', 0),
            'train_fn': epoch_data.get('train_fn', 0),
            'train_tp': epoch_data.get('train_tp', 0),
            'val_loss': epoch_data.get('val_loss', 0),
            'val_gen_loss': epoch_data.get('val_gen_loss', 0.0),
            'val_align_loss': epoch_data.get('val_align_loss', 0.0),
            'val_cee_loss': epoch_data.get('val_cee_loss', 0.0),
            'val_accuracy': epoch_data.get('val_accuracy', 0),
            'val_tn': epoch_data.get('val_tn', 0),
            'val_fp': epoch_data.get('val_fp', 0),
            'val_fn': epoch_data.get('val_fn', 0),
            'val_tp': epoch_data.get('val_tp', 0),
            'epoch_time_sec': epoch_data.get('epoch_time_sec', 0.0),
            'cumulative_time_sec': epoch_data.get('cumulative_time_sec', 0.0),
        }
        simplified_history.append(simplified)

    with open(history_path, 'w') as f:
        json.dump(simplified_history, f, indent=2, ensure_ascii=False)

    fold_summary = {
        'fold_idx': fold_idx,
        'best_val_loss': best_val_loss,
        'best_model_path': best_model_path,
        'final_train_accuracy': train_history[-1]['train_accuracy'] if train_history else 0.0,
        'final_val_accuracy': train_history[-1]['val_accuracy'] if train_history else 0.0,
        'num_epochs': len(train_history),
        'fold_summary_path': history_path,
        'metrics_dirs': [os.path.join(fold_output_dir, f"epoch_{i+1}") for i in range(len(train_history))],

        # ===== 记录用于选 best 的 val max-F1 信息 =====
        'best_val_f1': float(best_val_f1),
        'best_val_theta': float(best_val_theta) if best_val_theta is not None else None,
        'best_val_stats': best_val_stats,
        'best_epoch_by_f1': int(best_epoch_idx) if best_epoch_idx is not None else None,
        'time_to_best_epoch_sec': float(best_time_to_epoch_sec) if best_time_to_epoch_sec is not None else None,
        # =====================================================
    }

    # 验证集上做阈值搜索（单 run）
    if best_model_path is not None and os.path.exists(best_model_path):
        print("\n开始基于验证集进行阈值网格搜索（仅生成置信度）...")

        model.load_model(best_model_path)
        model.to(config.device)
        model.eval()

        s_gen_val, y_val, val_node_ids = compute_scores_for_loader(
            model, val_loader, config
        )

        best_theta, best_f1, best_stats = grid_search_gen_threshold(
            s_gen_val, y_val
        )

        print(f"\n🔥 最佳阈值搜索结果（单阈值）:")
        print(f"  θ (theta, val max-F1) = {best_theta:.4f}")
        print(f"  best_F1 = {best_f1:.4f}")
        print(f"  验证集统计: {best_stats}")

        fold_summary["best_thresholds"] = {
            "theta": float(best_theta),
            "best_f1": float(best_f1),
            "val_stats": best_stats
        }

        # ===== 追加：对 best_model 在验证集上重算 Pareto 前沿，并保存前沿与图 =====
        try:
            s_gen_val_best, y_val_best, _ = compute_scores_for_loader(model, val_loader, config)
            pareto_best_rec_b, pareto_frontier_b, records_all_b = pareto_search_threshold(
                s_gen_val_best, y_val_best, num_thresholds=201, tie_break="higher_precision"
            )
            selected_threshold_rec = choose_validation_threshold(records_all_b, pareto_frontier_b, config)

            pareto_plot_dir = os.path.join(fold_output_dir, "pareto_plots")
            tag = f"best_epoch{best_epoch_idx:03d}" if best_epoch_idx is not None else "best_epoch"
            out2d_b, out3d_b = plot_pareto_curves(records_all_b, pareto_frontier_b, selected_threshold_rec, pareto_plot_dir, tag=tag)

            fold_summary["best_thresholds"]["pareto_selection"] = {
                "selection_rule": "pareto_max_f1",
                "theta": float(pareto_best_rec_b["theta"]),
                "f1": float(pareto_best_rec_b["f1"]),
                "j": float(pareto_best_rec_b["j"]),
                "precision": float(pareto_best_rec_b["precision"]),
                "recall": float(pareto_best_rec_b["recall"]),
                "fpr": float(pareto_best_rec_b["fpr"]),
                "TN": int(pareto_best_rec_b["TN"]),
                "FP": int(pareto_best_rec_b["FP"]),
                "FN": int(pareto_best_rec_b["FN"]),
                "TP": int(pareto_best_rec_b["TP"]),
                "pareto_size": int(len(pareto_frontier_b)),
                "pareto_plots": {
                    "f1_vs_j_2d": out2d_b,
                    "theta_f1_j_3d": out3d_b
                }
            }

            threshold_rule = getattr(config, "threshold_selection_rule", "pareto_max_f1")
            fold_summary["test_threshold_selection"] = {
                "selection_rule": threshold_rule,
                "theta": float(selected_threshold_rec["theta"]),
                "f1": float(selected_threshold_rec["f1"]),
                "j": float(selected_threshold_rec["j"]),
                "precision": float(selected_threshold_rec["precision"]),
                "recall": float(selected_threshold_rec["recall"]),
                "fpr": float(selected_threshold_rec["fpr"]),
                "TN": int(selected_threshold_rec["TN"]),
                "FP": int(selected_threshold_rec["FP"]),
                "FN": int(selected_threshold_rec["FN"]),
                "TP": int(selected_threshold_rec["TP"]),
            }
            fold_summary["selected_test_theta"] = float(selected_threshold_rec["theta"])
            fold_summary["selected_test_threshold_rule"] = threshold_rule
            print(
                f"  测试阈值选择规则: {threshold_rule}, "
                f"theta={selected_threshold_rec['theta']:.4f}, "
                f"val_F1={selected_threshold_rec['f1']:.4f}, val_J={selected_threshold_rec['j']:.4f}"
            )

            # 仅保存 Pareto 前沿（规模小，可读性强）
            fold_summary["pareto_frontier"] = [
                {
                    "theta": float(r["theta"]),
                    "f1": float(r["f1"]),
                    "j": float(r["j"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "fpr": float(r["fpr"]),
                    "TN": int(r["TN"]),
                    "FP": int(r["FP"]),
                    "FN": int(r["FN"]),
                    "TP": int(r["TP"]),
                }
                for r in pareto_frontier_b
            ]
        except Exception as e:
            print(f"[Warn] Pareto frontier plot/save failed: {e}")
        # ===== 追加结束 =====


    fold_summary_path = os.path.join(fold_output_dir, "fold_summary.json")
    with open(fold_summary_path, 'w') as f:
        json.dump(fold_summary, f, indent=2, ensure_ascii=False)

    print("\n单次训练完成!")
    print(f"  最佳验证损失(对应 best_epoch_by_f1): {best_val_loss:.4f}")
    print(f"  最终验证准确率: {fold_summary['final_val_accuracy']:.4f}")
    print(f"  best val F1: {fold_summary.get('best_val_f1', 0.0):.4f} (epoch={fold_summary.get('best_epoch_by_f1', None)})")
    print(f"  详细指标保存到: {fold_output_dir}/")

    # 画损失曲线
    plot_losses(train_history, fold_output_dir, fold_idx)

    return fold_summary


def main():
    """主函数：支持 TempGT-LLM / TempGT-Base 多次运行，并输出 mean ± std。"""
    print("=" * 80)
    print("统一LoRA微调训练 - 问题一对照实验")
    print("=" * 80)

    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in config.__dict__.items()}
    config_path = os.path.join(config.output_dir, "config.json")
    save_json(config_path, config_dict)
    print(f"配置保存到: {config_path}")
    print(f"当前实验: {config.experiment_name}")
    print(f"backbone_init_mode: {config.backbone_init_mode}")
    print(f"num_runs: {config.num_runs}, random_seeds: {config.random_seeds}")
    print(f"val_test_neg_pos_ratio: {getattr(config, 'val_test_neg_pos_ratio', None)}")
    print(f"LoRA: r={config.lora_rank}, alpha={config.lora_alpha}, dropout={config.lora_dropout}, target_modules={config.target_modules}")
    print(f"loss weights: lambda_gen={config.lambda_gen}, lambda_align={config.lambda_align}, lambda_cee={config.lambda_cee}")
    print(f"align_max_negatives: {getattr(config, 'align_max_negatives', None)}")
    print(f"threshold_selection_rule: {getattr(config, 'threshold_selection_rule', 'pareto_max_f1')}")

    # 1. 加载全部节点数据与标签
    node_data, labels, graph_data = load_all_data(config)
    if len(node_data) == 0 or len(labels) == 0:
        print("❌ 没有有效数据，退出")
        return

    some_id = next(iter(node_data.keys()))
    print("示例节点字段:", node_data[some_id].keys())

    labeled_node_ids = list(labels.keys())
    print(f"有标签节点: {len(labeled_node_ids)}")

    # 2. 固定一次 split，所有 run 共用，保证公平对照
    split_path = os.path.join(config.output_dir, "dataset_split.json")
    expected_ratio = getattr(config, "val_test_neg_pos_ratio", None)

    if config.reuse_existing_split and os.path.exists(split_path):
        with open(split_path, "r") as f:
            split_data = json.load(f)
        if can_reuse_split(split_data, config, labels, node_data):
            train_node_ids = split_data["train_node_ids"]
            val_node_ids = split_data["val_node_ids"]
            test_node_ids = split_data["test_node_ids"]
            print(f"复用已有数据划分: {split_path}")
        else:
            train_node_ids, val_node_ids, test_node_ids = split_train_balanced_val_test_imbalanced(
                labeled_node_ids,
                labels,
                train_ratio=getattr(config, "train_ratio", 0.6),
                val_ratio=getattr(config, "val_ratio", 0.2),
                test_ratio=getattr(config, "test_ratio", 0.2),
                seed=getattr(config, "split_seed", 42),
                val_test_neg_pos_ratio=expected_ratio,
            )
            split_payload = build_split_payload(train_node_ids, val_node_ids, test_node_ids, labels, config)
            assert_split_protocol(split_payload, labels, expected_ratio)
            save_json(split_path, split_payload)
            print(f"数据集划分结果已保存到: {split_path}")
    else:
        train_node_ids, val_node_ids, test_node_ids = split_train_balanced_val_test_imbalanced(
            labeled_node_ids,
            labels,
            train_ratio=getattr(config, "train_ratio", 0.6),
            val_ratio=getattr(config, "val_ratio", 0.2),
            test_ratio=getattr(config, "test_ratio", 0.2),
            seed=getattr(config, "split_seed", 42),
            val_test_neg_pos_ratio=expected_ratio,
        )
        split_payload = build_split_payload(train_node_ids, val_node_ids, test_node_ids, labels, config)
        assert_split_protocol(split_payload, labels, expected_ratio)
        save_json(split_path, split_payload)
        print(f"数据集划分结果已保存到: {split_path}")

    verify_training_assets(train_node_ids, node_data, config)

    def count_stats(node_ids_subset):
        total = len(node_ids_subset)
        mal = sum(1 for nid in node_ids_subset if labels.get(nid, 0) == 1)
        ben = total - mal
        return total, mal, ben

    tr_total, tr_mal, tr_ben = count_stats(train_node_ids)
    val_total, val_mal, val_ben = count_stats(val_node_ids)
    te_total, te_mal, te_ben = count_stats(test_node_ids)

    print("\n数据集划分结果（节点级）:")
    print(f"  Train: {tr_total} (恶意={tr_mal}, 正常={tr_ben})")
    print(f"  Val  : {val_total} (恶意={val_mal}, 正常={val_ben})")
    print(f"  Test : {te_total} (恶意={te_mal}, 正常={te_ben})")

    # 3. 分词器（两种模式都使用同一个 tokenizer，保持 prompt / token 接口一致）
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_results = []

    for run_idx in range(config.num_runs):
        run_seed = int(config.random_seeds[run_idx])
        print("\n" + "=" * 80)
        print(f"开始 Run {run_idx + 1}/{config.num_runs} | seed={run_seed}")
        print("=" * 80)
        set_global_seed(run_seed)
        config.current_run_seed = run_seed

        # 每个 run 重新构造 DataLoader，训练集 shuffle 顺序随 seed 改变
        train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
            train_node_ids, val_node_ids, node_data, labels, config
        )
        _, test_loader, _, test_dataset = create_data_loaders(
            train_node_ids, test_node_ids, node_data, labels, config
        )

        print(f"训练集: {len(train_dataset)} 个样本")
        print(f"验证集: {len(val_dataset)} 个样本")
        print(f"测试集: {len(test_dataset)} 个样本")

        if len(train_dataset) == 0:
            print("⚠️ 训练集为空，退出。")
            return

        print("\n初始化模型...")
        model = UnifiedTemporalGTLLM(config)
        model.set_tokenizer(tokenizer)

        res = train_one_fold(model, train_loader, val_loader, config, run_idx, config.output_dir, tokenizer)
        res["run_seed"] = run_seed

        # 训练结束后，先释放训练阶段模型占用的显存；
        # 否则测试阶段再新建一份大模型并加载 best checkpoint 时，容易出现显存峰值 OOM。
        del model
        gc.collect()
        if getattr(config.device, "type", "") == "cuda":
            torch.cuda.empty_cache()

        threshold_info = res.get("test_threshold_selection", {}) or {}
        selected_theta = threshold_info.get(
            "theta",
            res.get("selected_test_theta", res.get("best_val_theta", 0.5)),
        )
        best_theta = float(selected_theta if selected_theta is not None else 0.5)
        test_metrics = evaluate_best_model_on_test(
            res["best_model_path"], best_theta, test_loader, config, tokenizer
        )
        res["test_metrics"] = test_metrics
        res["test_threshold_used"] = {
            "selection_rule": threshold_info.get("selection_rule", getattr(config, "threshold_selection_rule", "pareto_max_f1")),
            "theta": best_theta,
        }

        fold_output_dir = os.path.join(config.output_dir, f"fold_{run_idx + 1}")
        fold_summary_path = os.path.join(fold_output_dir, "fold_summary.json")
        if os.path.exists(fold_summary_path):
            with open(fold_summary_path, "r") as f:
                fold_summary = json.load(f)
        else:
            fold_summary = {}
        fold_summary["run_seed"] = run_seed
        fold_summary["backbone_init_mode"] = config.backbone_init_mode
        fold_summary["experiment_name"] = config.experiment_name
        fold_summary["test_metrics"] = test_metrics
        fold_summary["test_threshold_used"] = res["test_threshold_used"]
        save_json(fold_summary_path, fold_summary)

        run_results.append(res)

        print("\n[Test metrics]")
        print(f"  F1={test_metrics['f1']:.4f}, AUC={test_metrics['roc_auc_gen']:.4f}, Youden={test_metrics['Youden_J']:.4f}")
        print(f"  R@10={test_metrics['recall_at_k']['R@10']['recall']:.4f}, R@30={test_metrics['recall_at_k']['R@30']['recall']:.4f}, R@50={test_metrics['recall_at_k']['R@50']['recall']:.4f}")

        gc.collect()
        if getattr(config.device, "type", "") == 'cuda':
            torch.cuda.empty_cache()

    # 4. 汇总 mean ± std
    summary = summarize_run_results(run_results)
    summary["config"] = config_dict
    summary["timestamp"] = datetime.datetime.now().isoformat()
    summary["experiment_name"] = config.experiment_name
    summary["backbone_init_mode"] = config.backbone_init_mode

    summary_path = os.path.join(config.output_dir, "training_summary.json")
    summary_csv_path = os.path.join(config.output_dir, "training_summary_mean_std.csv")
    save_json(summary_path, summary)
    write_summary_csv(summary, summary_csv_path)

    readme_content = f"""# 问题一实验结果目录结构（{config.experiment_name}）

## 实验设定
- backbone_init_mode: {config.backbone_init_mode}
- num_runs: {config.num_runs}
- random_seeds: {config.random_seeds}
- split_seed: {config.split_seed}
- LoRA: r={config.lora_rank}, alpha={config.lora_alpha}, dropout={config.lora_dropout}, target_modules={config.target_modules}
- loss weights: lambda_gen={config.lambda_gen}, lambda_align={config.lambda_align}, lambda_cee={config.lambda_cee}
- align_max_negatives: {getattr(config, 'align_max_negatives', None)}
- threshold_selection_rule: {getattr(config, 'threshold_selection_rule', 'pareto_max_f1')}

## 文件说明
- `config.json`: 训练配置
- `dataset_split.json`: 固定的 Train/Val/Test 划分
- `training_summary.json`: 多次运行汇总（含每次 run 与 mean/std）
- `training_summary_mean_std.csv`: 关键指标的 mean ± std 表格

## 关键 test 指标（mean ± std）
- F1: {summary['aggregate']['test_f1']['mean']:.4f} ± {summary['aggregate']['test_f1']['std']:.4f}
- AUC: {summary['aggregate']['test_auc']['mean']:.4f} ± {summary['aggregate']['test_auc']['std']:.4f}
- Youden: {summary['aggregate']['test_youden']['mean']:.4f} ± {summary['aggregate']['test_youden']['std']:.4f}
- R@10: {summary['aggregate']['test_r10']['mean']:.4f} ± {summary['aggregate']['test_r10']['std']:.4f}
- R@30: {summary['aggregate']['test_r30']['mean']:.4f} ± {summary['aggregate']['test_r30']['std']:.4f}
- R@50: {summary['aggregate']['test_r50']['mean']:.4f} ± {summary['aggregate']['test_r50']['std']:.4f}

## 收敛速度（mean ± std）
- best_epoch_by_f1: {summary['aggregate']['best_epoch_by_f1']['mean']:.4f} ± {summary['aggregate']['best_epoch_by_f1']['std']:.4f}
- time_to_best_epoch_sec: {summary['aggregate']['time_to_best_epoch_sec']['mean']:.2f} ± {summary['aggregate']['time_to_best_epoch_sec']['std']:.2f}
"""
    readme_path = os.path.join(config.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    selected_meta = export_best_model_across_runs(run_results, config, summary)

    print("\n" + "=" * 80)
    print("训练完成！汇总结果")
    print("=" * 80)
    print(f"Test F1     : {summary['aggregate']['test_f1']['mean']:.4f} ± {summary['aggregate']['test_f1']['std']:.4f}")
    print(f"Test AUC    : {summary['aggregate']['test_auc']['mean']:.4f} ± {summary['aggregate']['test_auc']['std']:.4f}")
    print(f"Test Youden : {summary['aggregate']['test_youden']['mean']:.4f} ± {summary['aggregate']['test_youden']['std']:.4f}")
    print(f"Test R@10   : {summary['aggregate']['test_r10']['mean']:.4f} ± {summary['aggregate']['test_r10']['std']:.4f}")
    print(f"Test R@30   : {summary['aggregate']['test_r30']['mean']:.4f} ± {summary['aggregate']['test_r30']['std']:.4f}")
    print(f"Test R@50   : {summary['aggregate']['test_r50']['mean']:.4f} ± {summary['aggregate']['test_r50']['std']:.4f}")
    print(f"收敛 epoch   : {summary['aggregate']['best_epoch_by_f1']['mean']:.4f} ± {summary['aggregate']['best_epoch_by_f1']['std']:.4f}")
    print(f"收敛时间(s)  : {summary['aggregate']['time_to_best_epoch_sec']['mean']:.2f} ± {summary['aggregate']['time_to_best_epoch_sec']['std']:.2f}")
    print(f"\n训练总结保存到: {summary_path}")
    print(f"CSV 汇总保存到: {summary_csv_path}")
    print(f"目录说明保存到: {readme_path}")
    if selected_meta:
        print(f"本次完整运行导出的总体最优模型: {selected_meta.get('selected_model_path', '')}")
        print(f"跨运行最佳模型汇总目录: {os.path.abspath(config.selected_best_models_dir)}")
    print(f"\n🎉 训练完成！输出目录: {os.path.abspath(config.output_dir)}")


if __name__ == "__main__":
    main()
