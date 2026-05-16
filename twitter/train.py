import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import os
import json
import datetime
from typing import Dict, List, Tuple
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from config import TrainingConfig
from utils import (
    find_paired_files, load_paired_embeddings, load_original_graph_data,
    get_user_label, balance_dataset, stratified_train_val_test_split
)
from data_loader import create_data_loaders
from unified_model import UnifiedTemporalGTLLM
from transformers import AutoTokenizer
import pickle
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_scores_for_loader(model, data_loader, config):
    """在验证 / 阈值搜索阶段，根据模型输出的生成置信度 s_gen 评估节点"""
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

            # 正常前向：包含 soft token + prompt + LLM
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

            # 取 LLM 原始 logits，形状 [B, L, vocab]
            logits = outputs["logits"]
            batch_size, seq_len, vocab_size = logits.shape

            # 取最后一个 token 的 logits 作为决策位置
            last_pos = torch.full(
                (batch_size,),
                seq_len - 1,
                device=device,
                dtype=torch.long,
            )
            batch_idx = torch.arange(batch_size, device=device, dtype=torch.long)
            last_token_logits = logits[batch_idx, last_pos, :]  # [B, vocab]

            # 使用 '0' 和 '1' 的 token id 作为 Benign / Malicious
            benign_token = model.llm_tokenizer(
                "0", return_tensors="pt", add_special_tokens=False
            )["input_ids"][0][-1].item()
            malicious_token = model.llm_tokenizer(
                "1", return_tensors="pt", add_special_tokens=False
            )["input_ids"][0][-1].item()

            # 只在 {'0','1'} 两个 token 上做 softmax，取 "1" 的概率作为 s_gen
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
    """基于验证集的生成置信度分数 s_gen，进行单阈值网格搜索"""
    theta_grid = np.linspace(0.0, 1.0, 101)

    best_J = -999.0
    best_theta = None
    best_stats = None

    for th in theta_grid:
        pred = (s_gen > th).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        J = TPR - FPR

        if J > best_J:
            best_J = J
            best_theta = th
            best_stats = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "TPR": float(TPR),
                "FPR": float(FPR),
            }

    return best_theta, best_J, best_stats


def load_all_data(config: TrainingConfig) -> Tuple[Dict, Dict, Dict]:
    """加载所有数据（节点级表示）"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    # 主数据集 paired files
    paired_files = find_paired_files(config.gat_encoded_dir, config.text_encoded_dir)

    # 额外 normal_6000 paired files（可选）
    normal_gat_dir = getattr(config, "normal_gat_encoded_dir", "gat_encoded_subgraphs_normal_6000")
    normal_text_dir = getattr(config, "normal_text_encoded_dir", "sbert_encoded_texts_normal_6000")

    paired_files_normal = []
    if os.path.exists(normal_gat_dir) and os.path.exists(normal_text_dir):
        paired_files_normal = find_paired_files(normal_gat_dir, normal_text_dir)

    # 合并两组数据
    all_pairs = list(paired_files) + list(paired_files_normal)

    print(f"[主数据] paired_files: {len(paired_files)}")
    print(f"[normal_6000] paired_files_normal: {len(paired_files_normal)}")
    print(f"[总计] all_pairs: {len(all_pairs)}")

    # 加载原始图数据（用于标签）
    graph_data = load_original_graph_data(config.original_graph_data_path)

    # 加载所有节点的嵌入和标签
    node_data: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}

    successful_nodes = 0
    successful_main = 0
    successful_normal6000 = 0

    # 判定pair是否来自normal_6000
    def _is_from_normal6000(gat_file: str, text_file: str) -> bool:
        return (normal_gat_dir in gat_file) or (normal_text_dir in text_file)

    for gat_file, text_file in tqdm(all_pairs, desc="加载节点数据"):
        embeddings = load_paired_embeddings(gat_file, text_file)
        if embeddings is None:
            continue

        node_id = embeddings.get("node_id", None)
        if node_id is None:
            continue

        if _is_from_normal6000(gat_file, text_file):
            label = 0
            from_normal6000 = True
        else:
            has_label, label = get_user_label(node_id, graph_data)
            if not has_label:
                continue
            from_normal6000 = False

        temporal_node_embeddings = embeddings.get("temporal_node_embeddings", {})
        temporal_text_embeddings = embeddings.get("temporal_text_embeddings", {})
        timesteps = embeddings.get("timesteps", [])

        if isinstance(timesteps, torch.Tensor):
            timesteps_list = timesteps.view(-1).long().tolist()
        else:
            timesteps_list = list(timesteps)

        if len(timesteps_list) == 0:
            ts_keys = set()
            if isinstance(temporal_node_embeddings, dict):
                ts_keys.update(list(temporal_node_embeddings.keys()))
            if isinstance(temporal_text_embeddings, dict):
                ts_keys.update(list(temporal_text_embeddings.keys()))
            timesteps_list = list(ts_keys)

        if len(timesteps_list) == 0:
            continue

        timesteps_sorted = sorted(int(t) for t in timesteps_list)

        if (not isinstance(temporal_node_embeddings, dict) or len(temporal_node_embeddings) == 0) and \
           (not isinstance(temporal_text_embeddings, dict) or len(temporal_text_embeddings) == 0):
            continue

        node_data[node_id] = {
            "temporal_node_embeddings": temporal_node_embeddings,
            "temporal_text_embeddings": temporal_text_embeddings,
            "timesteps": timesteps_sorted,
        }
        labels[node_id] = int(label)

        successful_nodes += 1
        if from_normal6000:
            successful_normal6000 += 1
        else:
            successful_main += 1

    print(f"数据加载完成: {successful_nodes} 个节点")
    print(f"  - 主数据集节点: {successful_main}")
    print(f"  - normal_6000 节点: {successful_normal6000}")
    print(f"标签分布: 恶意={sum(labels.values())}, 正常={len(labels) - sum(labels.values())}")

    if len(paired_files_normal) > 0 and successful_normal6000 == 0:
        print("检测到 normal_6000 配对文件存在，但全部未成功加载。")

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

    progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")

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

    all_preds = []
    all_labels = []
    all_node_ids = []

    progress_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}")

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
            if gen_loss is not None:
                total_gen_loss += gen_loss.item() * batch_size
            if cee_loss is not None:
                total_cee_loss += cee_loss.item() * batch_size
            total_samples += batch_size

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
        "val_loss": float(avg_loss),
        "val_align_loss": float(avg_align),
        "val_gen_loss": float(avg_gen),
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
    """绘制损失图"""
    if not train_history:
        return

    epochs = list(range(1, len(train_history) + 1))

    # 生成损失
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
    print(f" Generation Loss 曲线已保存到: {gen_fig_path}")

    # 对齐损失
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
    print(f" Alignment Loss 曲线已保存到: {align_fig_path}")

    # CEE 损失
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
    print(f" CEE Loss 曲线已保存到: {cee_fig_path}")


def train_one_fold(model: UnifiedTemporalGTLLM, train_loader, val_loader,
                   config: TrainingConfig, fold_idx: int, output_dir: str, tokenizer=None) -> Dict[str, float]:
    """单次训练 run"""

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

    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, config, epoch, fold_idx)
        val_metrics = validate(model, val_loader, config, tokenizer, epoch, fold_idx)

        train_metrics_path, val_metrics_path, summary_path = save_epoch_metrics(
            fold_idx, epoch, train_metrics, val_metrics, output_dir
        )

        epoch_metrics = {**train_metrics, **val_metrics}
        train_history.append(epoch_metrics)

        print(f"  训练损失: {train_metrics['train_loss']:.4f} "
              f"(生成: {train_metrics['train_gen_loss']:.4f}, 对齐: {train_metrics['train_align_loss']:.4f}, CEE: {train_metrics['train_cee_loss']:.4f})")
        print(f"  验证损失: {val_metrics['val_loss']:.4f} "
              f"(生成: {val_metrics['val_gen_loss']:.4f}, 对齐: {val_metrics['val_align_loss']:.4f}, CEE: {val_metrics['val_cee_loss']:.4f})")

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

        # 保存最佳模型
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_path = os.path.join(fold_output_dir, f"fold_{fold_idx+1}_best_model.pth")
            model.save_model(best_model_path)
            print(f" 保存最佳模型: {best_model_path}")

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
        'metrics_dirs': [os.path.join(fold_output_dir, f"epoch_{i+1}") for i in range(len(train_history))]
    }

    # 验证集上做阈值搜索
    if best_model_path is not None and os.path.exists(best_model_path):
        print("\n开始基于验证集进行阈值网格搜索（仅生成置信度）...")

        model.load_model(best_model_path)
        model.to(config.device)
        model.eval()

        s_gen_val, y_val, val_node_ids = compute_scores_for_loader(
            model, val_loader, config
        )

        best_theta, best_J, best_stats = grid_search_gen_threshold(
            s_gen_val, y_val
        )

        print(f"\n最佳阈值搜索结果（单阈值）:")
        print(f"  θ (theta) = {best_theta:.4f}")
        print(f"  Youden's J = {best_J:.4f}")
        print(f"  验证集统计: {best_stats}")

        fold_summary["best_thresholds"] = {
            "theta": float(best_theta),
            "youden_J": float(best_J),
            "val_stats": best_stats
        }

    fold_summary_path = os.path.join(fold_output_dir, "fold_summary.json")
    with open(fold_summary_path, 'w') as f:
        json.dump(fold_summary, f, indent=2, ensure_ascii=False)

    print("\n单次训练完成!")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  最终验证准确率: {fold_summary['final_val_accuracy']:.4f}")
    print(f"  详细指标保存到: {fold_output_dir}/")

    # 绘制损失曲线
    plot_losses(train_history, fold_output_dir, fold_idx)

    return fold_summary


def main():
    """主函数：单次 6:2:2 分层划分 + 微调"""
    print("=" * 80)
    print("统一LoRA微调训练 - 端到端联合训练所有参数")
    print("=" * 80)

    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in config.__dict__.items()}
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"配置保存到: {config_path}")

    # 加载全部节点数据与标签
    node_data, labels, graph_data = load_all_data(config)

    if len(node_data) == 0 or len(labels) == 0:
        print("没有有效数据，退出")
        return

    some_id = next(iter(node_data.keys()))
    print("示例节点字段:", node_data[some_id].keys())

    labeled_node_ids = list(labels.keys())
    print(f"有标签节点: {len(labeled_node_ids)}")

    # 全局类别平衡（可选）
    if config.malicious_to_normal_ratio != 1.0:
        labeled_node_ids = balance_dataset(
            labeled_node_ids, labels, config.malicious_to_normal_ratio
        )

    # 分层 6:2:2 划分 train / val / test，并保证各子集恶意/正常尽量 1:1
    train_node_ids, val_node_ids, test_node_ids = stratified_train_val_test_split(
        labeled_node_ids,
        labels,
        train_ratio=getattr(config, "train_ratio", 0.6),
        val_ratio=getattr(config, "val_ratio", 0.2),
        test_ratio=getattr(config, "test_ratio", 0.2),
    )


    # 将额外的 normal_6000 正常用户平均分配到 Val / Test（各 3000）
    all_split_ids = set(train_node_ids) | set(val_node_ids) | set(test_node_ids)
    extra_ids = [nid for nid in node_data.keys() if nid not in all_split_ids]
    extra_normal_ids = [nid for nid in extra_ids if labels.get(nid, 0) == 0]

    # 固定随机性，确保可复现
    rng_extra = np.random.RandomState(0)
    rng_extra.shuffle(extra_normal_ids)

    extra_each = 3000
    extra_val_ids = extra_normal_ids[:extra_each]
    extra_test_ids = extra_normal_ids[extra_each: 2 * extra_each]

    val_node_ids = list(val_node_ids) + list(extra_val_ids)
    test_node_ids = list(test_node_ids) + list(extra_test_ids)

    print("\n[Extra normal_6000] 注入 Val/Test：")
    print(f"  额外正常用户候选总数: {len(extra_normal_ids)}")
    print(f"  注入到 Val 的额外正常用户数: {len(extra_val_ids)}")
    print(f"  注入到 Test 的额外正常用户数: {len(extra_test_ids)}")

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

    # 保存划分结果
    split_path = os.path.join(config.output_dir, "dataset_split.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "train_node_ids": train_node_ids,
                "val_node_ids": val_node_ids,
                "test_node_ids": test_node_ids,
                "extra_normal_val_ids": list(extra_val_ids),
                "extra_normal_test_ids": list(extra_test_ids),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"数据集划分结果已保存到: {split_path}")

    # 加载分词器
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_path,
        trust_remote_code=True,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构造 DataLoader（仅 train / val）
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        train_node_ids, val_node_ids, node_data, labels, config
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")

    if len(train_dataset) == 0:
        print(" 训练集为空，退出。")
        return

    # 初始化模型并训练（单次 run）
    print("\n初始化模型...")
    model = UnifiedTemporalGTLLM(config)
    model.set_tokenizer(tokenizer)

    fold_idx = 0  # 单次 run 视作 fold 0
    res = train_one_fold(model, train_loader, val_loader, config, fold_idx, config.output_dir, tokenizer)

    del model
    if getattr(config.device, "type", "") == 'cuda':
        torch.cuda.empty_cache()

    # 保存整体 summary（单 run）
    print("\n" + "=" * 80)
    print("训练完成！汇总结果")
    print("=" * 80)

    avg_val_loss = res['best_val_loss']
    avg_val_accuracy = res['final_val_accuracy']
    avg_train_accuracy = res['final_train_accuracy']

    print(f"验证损失: {avg_val_loss:.4f}")
    print(f"训练准确率: {avg_train_accuracy:.4f}")
    print(f"验证准确率: {avg_val_accuracy:.4f}")

    summary = {
        'num_runs': 1,
        'avg_train_accuracy': avg_train_accuracy,
        'avg_val_loss': avg_val_loss,
        'avg_val_accuracy': avg_val_accuracy,
        'run_results': [res],
        'config': {k: str(v) if isinstance(v, torch.device) else v for k, v in config.__dict__.items()},
        'timestamp': datetime.datetime.now().isoformat(),
        'metrics_structure': {
            'run_dir': 'fold_1/',
            'per_epoch': 'fold_1/epoch_{epoch+1}/',
            'files_per_epoch': ['train_metrics.json', 'val_metrics.json', 'summary_metrics.json']
        }
    }

    summary_path = os.path.join(config.output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # README 说明
    readme_content = f"""# 训练结果目录结构（单次 6:2:2 微调）

## 文件说明
- `config.json`: 训练配置
- `training_summary.json`: 训练汇总结果
- `fold_1_best_model.pth`: 本次训练的最佳模型
- `dataset_split.json`: 节点级 Train/Val/Test 划分结果

## 目录结构
- `fold_1/`: 本次训练结果
  - `fold_summary.json`: 训练汇总
  - `training_history.json`: 训练历史（按 epoch）
  - `best_epoch/`: 最佳 epoch 的指标文件
  - `epoch_1/`: 第 1 个 epoch 的详细指标
    - `train_metrics.json`
    - `val_metrics.json`
    - `summary_metrics.json`
  - `epoch_2/` ...
  - `run_1_gen_loss.png`: 生成损失曲线
  - `run_1_align_loss.png`: 对齐损失曲线
  - `run_1_cee_loss.png`: CEE 损失曲线

## 数据集划分
- 使用分层 6:2:2 划分 (Train:Val:Test)，并尽量保证每个子集中恶意/正常 1:1。
- Train: 用于微调
- Val: 用于验证和阈值网格搜索
- Test: 保留给后续推理 / 评估脚本使用

## 混淆矩阵说明
- TN (True Negative): 真实为正常，预测为正常
- FP (False Positive): 真实为正常，预测为恶意
- FN (False Negative): 真实为恶意，预测为正常
- TP (True Positive): 真实为恶意，预测为恶意

## 本次训练信息
- 训练准确率: {avg_train_accuracy:.4f}
- 验证准确率: {avg_val_accuracy:.4f}
- 验证损失: {avg_val_loss:.4f}
- 训练时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    readme_path = os.path.join(config.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"\n训练总结保存到: {summary_path}")
    print(f"目录说明保存到: {readme_path}")
    print(f"\n 训练完成！输出目录: {os.path.abspath(config.output_dir)}")


if __name__ == "__main__":
    main()
