import os
import json
import glob
import gc
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


# ============================================================
# 1. 加载所有节点数据（节点级新格式，和 train.py 保持一致）
# ============================================================
def load_all_data(config: TrainingConfig) -> Tuple[Dict, Dict, Dict]:
    print("=" * 60)
    print("推理阶段：加载数据（全体用户，节点级表示）...")
    print("=" * 60)

    paired_files = find_paired_files(config.gat_encoded_dir, config.text_encoded_dir)
    graph_data = load_original_graph_data(config.original_graph_data_path)

    node_data: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}
    successful_nodes = 0

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


# ============================================================
# 2. 计算 s_gen（与训练/验证阶段保持一致）
# ============================================================
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
        for batch in tqdm(data_loader, desc="计算 s_gen（测试集用户）"):
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

            logits = outputs["logits"]
            batch_size, seq_len, _ = logits.shape

            last_pos = torch.full((batch_size,), seq_len - 1, device=device, dtype=torch.long)
            batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
            last_token_logits = logits[batch_idx, last_pos, :]

            benign_token = model.llm_tokenizer("0", return_tensors="pt", add_special_tokens=False)["input_ids"][0][-1].item()
            malicious_token = model.llm_tokenizer("1", return_tensors="pt", add_special_tokens=False)["input_ids"][0][-1].item()

            pair_logits = torch.stack(
                [last_token_logits[:, benign_token], last_token_logits[:, malicious_token]],
                dim=-1,
            )
            pair_prob = F.softmax(pair_logits, dim=-1)
            p_malicious = pair_prob[:, 1]

            structure_embeddings, text_embeddings, pooled_time_mask = model._pool_from_node_level(
                struct_in,
                struct_out,
                text_nodes,
                struct_in_mask,
                struct_out_mask,
                text_mask,
                time_mask,
            )

            B_cur, T_cur, struct_dim = structure_embeddings.shape
            text_dim = text_embeddings.size(-1)

            proj_text = model.text_to_struct_proj(
                text_embeddings.view(B_cur * T_cur, text_dim)
            ).view(B_cur, T_cur, struct_dim)
            fused_states = torch.cat([structure_embeddings, proj_text], dim=-1)

            batch_cee = []
            for i_b in range(B_cur):
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


# ============================================================
# 3. 单阈值分类 + 指标计算
# ============================================================
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
            if len(y_true) > 0 and y_true[0] == 0:
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




def format_confusion_matrix_from_metrics(metrics: Dict) -> str:
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))
    return (
        f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]"
    )


# ============================================================
# 4. R@K
# ============================================================
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


# ============================================================
# 5. 多模型汇总
# ============================================================
def summarize_selected_model_results(per_model_results: List[Dict]) -> Dict:
    metric_map = {
        "f1": [r["metrics"]["f1"] for r in per_model_results],
        "auc": [r["metrics"]["roc_auc_gen"] for r in per_model_results],
        "youden": [r["metrics"]["Youden_J"] for r in per_model_results],
        "r10": [r["metrics"]["recall_at_k"]["R@10"]["recall"] for r in per_model_results],
        "r30": [r["metrics"]["recall_at_k"]["R@30"]["recall"] for r in per_model_results],
        "r50": [r["metrics"]["recall_at_k"]["R@50"]["recall"] for r in per_model_results],
    }
    aggregate = {}
    for name, values in metric_map.items():
        arr = np.asarray(values, dtype=np.float64)
        aggregate[name] = {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std(ddof=0)) if arr.size else 0.0,
            "values": [float(x) for x in arr.tolist()],
        }
    return {"num_models": len(per_model_results), "per_model_results": per_model_results, "aggregate": aggregate}


def write_inference_summary_csv(summary: Dict, csv_path: str):
    lines = ["metric,mean,std,values\n"]
    for metric, payload in summary.get("aggregate", {}).items():
        values = "|".join(f"{v:.6f}" for v in payload["values"])
        lines.append(f"{metric},{payload['mean']:.6f},{payload['std']:.6f},{values}\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ============================================================
# 6. 主推理逻辑：读取 selected_best_models_dir 中的多个最佳模型
# ============================================================
def main():
    print("=" * 80)
    print("LLM 推理阶段：读取 selected_best_models_dir 中的多个最佳模型并汇总推理结果")
    print("=" * 80)

    config = TrainingConfig()
    os.makedirs(config.selected_best_models_dir, exist_ok=True)
    print(f"使用设备: {config.device}")
    print(f"selected_best_models_dir: {config.selected_best_models_dir}")

    node_data, labels, graph_data = load_all_data(config)
    if len(node_data) == 0 or len(labels) == 0:
        print("❌ 没有有效数据，退出")
        return

    split_path = os.path.join(config.selected_best_models_dir, "dataset_split.json")
    if not os.path.exists(split_path):
        split_path = os.path.join(config.output_dir, "dataset_split.json")

    if not os.path.exists(split_path):
        print("❌ 未找到数据划分文件。")
        print(f"  已检查: {os.path.join(config.selected_best_models_dir, 'dataset_split.json')}")
        print(f"  已检查: {os.path.join(config.output_dir, 'dataset_split.json')}")
        print("  请先运行训练脚本，确保至少一次训练已将 dataset_split.json 保存下来。")
        return

    print(f"从划分文件中载入测试集节点: {split_path}")
    with open(split_path, "r") as f:
        split_info = json.load(f)

    test_ids = split_info.get("test_node_ids", [])
    test_node_data = {nid: node_data[nid] for nid in test_ids if nid in node_data}
    test_labels = {nid: labels[nid] for nid in test_ids if nid in labels}
    print(f"测试集有效用户数: {len(test_node_data)}")

    from torch.utils.data import DataLoader
    struct_node_dim = config.structure_dim // 2
    text_dim = config.text_dim

    test_dataset = TemporalGraphTextDataset(
        node_data=test_node_data,
        labels=test_labels,
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

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    meta_paths = sorted(glob.glob(os.path.join(config.selected_best_models_dir, "*_meta.json")))
    if len(meta_paths) == 0:
        print(f"❌ 在 {config.selected_best_models_dir} 中未找到 *_meta.json")
        print("   请先运行训练脚本，确保每次完整运行结束后已导出总体最优模型。")
        return

    print("\n检测到以下待推理模型：")
    for p in meta_paths:
        print(" ", os.path.basename(p))

    inference_output_dir = os.path.join(config.selected_best_models_dir, "inference_outputs")
    os.makedirs(inference_output_dir, exist_ok=True)

    per_model_results = []

    for meta_path in meta_paths:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        tag = meta.get("outer_experiment_tag", os.path.basename(meta_path).replace("_meta.json", ""))
        best_model_path = meta.get("selected_model_path", os.path.join(config.selected_best_models_dir, f"{tag}_best_model.pth"))
        theta = float(meta.get("best_val_theta", 0.5))

        if not os.path.exists(best_model_path):
            print(f"⚠️ 跳过 {tag}：未找到模型文件 {best_model_path}")
            continue

        print("\n" + "=" * 60)
        print(f"开始推理: {tag}")
        print("=" * 60)
        print(f"  模型: {best_model_path}")
        print(f"  阈值: θ={theta:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = UnifiedTemporalGTLLM(config)
        model.set_tokenizer(tokenizer)
        model.load_model(best_model_path)
        model.to(config.device)
        model.eval()

        s_gen_all, y_all, node_ids_all, cee_all = compute_scores_for_loader(model, test_loader, config)
        metrics = apply_dual_threshold(s_gen_all, y_all, theta)
        recall_at_k = compute_recall_at_k_with_counts(s_gen_all, y_all, k_list=[10, 30, 50])

        result = {
            "outer_experiment_tag": tag,
            "model_path": best_model_path,
            "meta_path": meta_path,
            "theta": theta,
            "metrics": {
                **metrics,
                "confusion_matrix": {
                    "tn": int(metrics["tn"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "tp": int(metrics["tp"]),
                },
                "recall_at_k": recall_at_k,
            },
        }
        per_model_results.append(result)

        per_model_save = {
            "outer_experiment_tag": tag,
            "theta": theta,
            "metrics": {
                **metrics,
                "confusion_matrix": {
                    "tn": int(metrics["tn"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "tp": int(metrics["tp"]),
                },
            },
            "recall_at_k": recall_at_k,
            "node_ids": node_ids_all,
            "s_gen": s_gen_all.tolist(),
            "y_true": y_all.tolist(),
            "y_pred": metrics["y_pred"],
        }
        out_json = os.path.join(inference_output_dir, f"{tag}_inference.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(per_model_save, f, indent=2, ensure_ascii=False)

        print(f"  F1={metrics['f1']:.4f}, AUC={metrics['roc_auc_gen']:.4f}, Youden={metrics['Youden_J']:.4f}")
        print(f"  R@10={recall_at_k['R@10']['recall']:.4f}, R@30={recall_at_k['R@30']['recall']:.4f}, R@50={recall_at_k['R@50']['recall']:.4f}")
        print("  Confusion Matrix:")
        print(f"    [[TN={metrics['tn']}, FP={metrics['fp']}],")
        print(f"     [FN={metrics['fn']}, TP={metrics['tp']}]]")
        print(f"  推理结果已保存到: {out_json}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(per_model_results) == 0:
        print("❌ 没有成功完成的模型推理结果，退出。")
        return

    summary = summarize_selected_model_results(per_model_results)
    summary["selected_best_models_dir"] = config.selected_best_models_dir
    summary["experiment_name"] = config.experiment_name
    summary["backbone_init_mode"] = config.backbone_init_mode

    summary_json_path = os.path.join(config.selected_best_models_dir, "multi_model_inference_summary.json")
    summary_csv_path = os.path.join(config.selected_best_models_dir, "multi_model_inference_summary.csv")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_inference_summary_csv(summary, summary_csv_path)

    print("\n" + "=" * 80)
    print("推理完成！多模型汇总结果")
    print("=" * 80)
    print(f"F1     : {summary['aggregate']['f1']['mean']:.4f} ± {summary['aggregate']['f1']['std']:.4f}")
    print(f"AUC    : {summary['aggregate']['auc']['mean']:.4f} ± {summary['aggregate']['auc']['std']:.4f}")
    print(f"Youden : {summary['aggregate']['youden']['mean']:.4f} ± {summary['aggregate']['youden']['std']:.4f}")
    print(f"R@10   : {summary['aggregate']['r10']['mean']:.4f} ± {summary['aggregate']['r10']['std']:.4f}")
    print(f"R@30   : {summary['aggregate']['r30']['mean']:.4f} ± {summary['aggregate']['r30']['std']:.4f}")
    print(f"R@50   : {summary['aggregate']['r50']['mean']:.4f} ± {summary['aggregate']['r50']['std']:.4f}")
    print(f"\nJSON 汇总保存到: {summary_json_path}")
    print(f"CSV 汇总保存到: {summary_csv_path}")


if __name__ == "__main__":
    main()
