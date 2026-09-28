import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import os
import argparse
import json
import glob
import random
import datetime
from typing import Dict, List, Tuple
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from config import TrainingConfig
from utils import (
    find_paired_files, load_paired_embeddings, load_original_graph_data,
    get_user_label, balance_dataset, stratified_train_val_test_split,
    load_gat_embeddings, extract_node_id
)
from data_loader import create_data_loaders, FixedPairBatchSampler
from unified_model import UnifiedTemporalGTLLM
from transformers import AutoTokenizer
import pickle
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _compact_temporal_embeddings(values, dtype_name: str):
    """Reduce resident CPU storage; collate_fn restores float32 batches."""
    if dtype_name == "float32":
        return values
    if dtype_name != "float16":
        raise ValueError(f"Unsupported cpu_embedding_dtype: {dtype_name}")
    if isinstance(values, dict):
        return {key: _compact_temporal_embeddings(value, dtype_name) for key, value in values.items()}
    if isinstance(values, torch.Tensor) and values.is_floating_point():
        if values.numel() and not torch.isfinite(values).all():
            raise ValueError("Non-finite embedding before CPU float16 conversion")
        if values.numel() and values.abs().max() > 65504:
            raise ValueError("Embedding magnitude exceeds float16 range")
        return values.to(dtype=torch.float16)
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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

            # forward 已按每个样本的最后一个有效位置提取 {'0','1'} logits。
            pair_logits = outputs["logits_2"]
            pair_prob = F.softmax(pair_logits, dim=-1)        # [B, 2]
            p_malicious = pair_prob[:, 1]                     # 概率 P("1"|prompt)

            s_gen_batch = p_malicious.detach().cpu().tolist()

            all_s_gen.extend(s_gen_batch)
            all_labels.extend(labels.detach().cpu().tolist())
            all_node_ids.extend(node_ids)

    return np.array(all_s_gen), np.array(all_labels), all_node_ids


def grid_search_gen_threshold(s_gen, labels):
    """基于验证集的生成置信度分数 s_gen，进行 F1 最优单阈值网格搜索。
    修改方向.md 四.1：阈值只能在验证集上搜索，严禁在测试集上扫描。"""
    theta_grid = np.linspace(0.0, 1.0, 101)

    best_f1 = -1.0
    best_theta = None
    best_stats = None

    for th in theta_grid:
        pred = (s_gen > th).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = TPR
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        J = TPR - FPR

        if f1 > best_f1:
            best_f1 = f1
            best_theta = th
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
                "Youden_J": float(J),
            }

    return best_theta, best_f1, best_stats


def _load_edge_drop_map(edge_drop_dir: str, map_name: str = "edge-drop", allowed_file_ids=None, storage_dtype="float32") -> Dict[str, Dict]:
    if not edge_drop_dir or not os.path.exists(edge_drop_dir):
        print(f"[EdgeDrop] {map_name} 增强目录不存在: {edge_drop_dir}")
        return {}

    gat_files = glob.glob(os.path.join(edge_drop_dir, "subgraph_*_k2_gat_encoded.pkl"))
    if allowed_file_ids is not None:
        gat_files = [f for f in gat_files if extract_node_id(f) in allowed_file_ids]
    edge_map: Dict[str, Dict] = {}
    duplicate_nodes = 0
    for gat_file in tqdm(gat_files, desc="加载 edge-drop GAT"):
        payload = load_gat_embeddings(gat_file)
        if payload is None:
            continue
        node_id = str(payload.get("node_id", extract_node_id(gat_file)))
        if node_id in edge_map:
            duplicate_nodes += 1
        edge_map[node_id] = _compact_temporal_embeddings(
            payload.get("temporal_node_embeddings", {}), storage_dtype
        )

    print(f"[EdgeDrop] 加载 {map_name} 增强结构视图: {len(edge_map)} 个节点, duplicate_nodes={duplicate_nodes}")
    return edge_map


def load_all_data(config: TrainingConfig, train_only: bool = False) -> Tuple[Dict, Dict, Dict]:
    """加载所有数据（节点级表示，并挂载 edge-drop 增强结构视图）"""
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

    load_extra_normal = bool(getattr(config, "load_extra_normal_embeddings", True))
    # CEE 预训练只需 main/train 的真实张量；normal_6000 可只登记 ID。
    all_pairs = list(paired_files)
    if load_extra_normal:
        all_pairs += list(paired_files_normal)

    # With a fixed, pre-existing split, test tensors need not be resident during
    # training. Read the small text payloads to map file IDs to account IDs;
    # retain stubs for test accounts so split validation remains unchanged.
    use_low_memory = train_only and (
        bool(getattr(config, "low_memory_train", False))
        or os.environ.get("NEW_EXE_LOW_MEMORY_TRAIN") == "1"
    )
    active_ids = None
    expected_ids = None
    main_file_ids = None
    normal_file_ids = None
    train_ids = None
    if use_low_memory:
        split_path = os.path.join(config.output_dir, "dataset_split.json")
        if not os.path.isfile(split_path):
            raise RuntimeError("low_memory_train requires an existing fixed dataset_split.json")
        with open(split_path, encoding="utf-8") as handle:
            split_data = json.load(handle)
        train_ids = set(split_data["train_node_ids"])
        active_ids = train_ids | set(split_data["val_node_ids"])
        expected_ids = active_ids | set(split_data["test_node_ids"])
        main_file_ids, normal_file_ids = set(), set()
        selected_pairs = []
        seen_ids = set()
        for gat_file, text_file in all_pairs:
            with open(text_file, "rb") as handle:
                text_payload = pickle.load(handle)
            raw_id = text_payload.get("center_node") if isinstance(text_payload, dict) else None
            if raw_id is None:
                raise RuntimeError(f"Missing center_node in text payload: {text_file}")
            node_id = str(raw_id)
            if node_id in seen_ids or node_id not in expected_ids:
                continue
            seen_ids.add(node_id)
            selected_pairs.append((gat_file, text_file, node_id))
            if node_id in train_ids:
                file_ids = normal_file_ids if gat_file.startswith(normal_gat_dir + os.sep) else main_file_ids
                file_ids.add(extract_node_id(gat_file))
        missing = expected_ids - seen_ids
        if missing:
            raise RuntimeError(f"Fixed split accounts missing from paired files: {len(missing)}; examples={list(missing)[:5]}")
        all_pairs = selected_pairs

    print(f"[主数据] paired_files: {len(paired_files)}")
    print(f"[normal_6000] paired_files_normal: {len(paired_files_normal)}")
    print(f"[总计] all_pairs: {len(all_pairs)}")

    # A verified account-label cache avoids holding the multi-GiB original graph
    # alongside the LLM and embeddings. It is generated from the original graph,
    # not from the model predictions.
    label_cache = None
    if use_low_memory:
        cache_path = getattr(config, "label_cache_path", "") or os.environ.get("NEW_EXE_LABEL_CACHE_PATH", "")
        if not cache_path or not os.path.isfile(cache_path):
            raise RuntimeError("low_memory_train requires a prepared label_cache_path")
        with open(cache_path, encoding="utf-8") as handle:
            cache_payload = json.load(handle)
        graph_stat = os.stat(config.original_graph_data_path)
        if (cache_payload.get("source_size") != graph_stat.st_size or
                cache_payload.get("source_mtime_ns") != graph_stat.st_mtime_ns):
            raise RuntimeError("Label cache source fingerprint differs from original graph")
        label_cache = cache_payload["labels"]
        if set(label_cache) != expected_ids or any(value not in (0, 1) for value in label_cache.values()):
            raise RuntimeError("Label cache does not exactly match fixed split accounts")
        graph_data = {}
    else:
        graph_data = load_original_graph_data(config.original_graph_data_path)

    # L_cont 的 edge-drop 增强结构视图
    storage_dtype = getattr(config, "cpu_embedding_dtype", "float32")
    edge_drop_map_main = _load_edge_drop_map(
        getattr(config, "edge_drop_gat_encoded_dir", ""), "main edge-drop", main_file_ids, storage_dtype
    )
    edge_drop_map_normal = _load_edge_drop_map(
        getattr(config, "normal_edge_drop_gat_encoded_dir", ""),
        "normal_6000 edge-drop",
        normal_file_ids,
        storage_dtype,
    )

    # 加载所有节点的嵌入和标签
    node_data: Dict[str, Dict] = {}
    labels: Dict[str, int] = {}

    successful_nodes = 0
    successful_main = 0
    successful_normal6000 = 0

    # 判定pair是否来自normal_6000
    def _is_from_normal6000(gat_file: str, text_file: str) -> bool:
        return (normal_gat_dir in gat_file) or (normal_text_dir in text_file)

    for pair in tqdm(all_pairs, desc="加载节点数据"):
        gat_file, text_file = pair[:2]
        mapped_node_id = pair[2] if use_low_memory else None
        if use_low_memory and mapped_node_id not in active_ids:
            from_normal6000 = normal_gat_dir in gat_file
            if from_normal6000:
                label = 0
            else:
                has_label, label = (True, label_cache[mapped_node_id]) if use_low_memory else get_user_label(mapped_node_id, graph_data)
                if not has_label:
                    raise RuntimeError(f"Fixed main test account has no label: {mapped_node_id}")
            node_data[mapped_node_id] = {
                "source": "normal_6000" if from_normal6000 else "main",
                "id_stub_only": True,
            }
            labels[mapped_node_id] = int(label)
            continue
        embeddings = load_paired_embeddings(gat_file, text_file)
        if embeddings is None:
            continue

        node_id = embeddings.get("node_id", None)
        if node_id is None:
            continue
        if use_low_memory and node_id != mapped_node_id:
            raise RuntimeError(f"GAT/text account mismatch for {gat_file}: {node_id} != {mapped_node_id}")

        if _is_from_normal6000(gat_file, text_file):
            label = 0
            from_normal6000 = True
        else:
            has_label, label = (True, label_cache[node_id]) if use_low_memory else get_user_label(node_id, graph_data)
            if not has_label:
                continue
            from_normal6000 = False

        temporal_node_embeddings = _compact_temporal_embeddings(
            embeddings.get("temporal_node_embeddings", {}), storage_dtype
        )
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

        # 三个账号同时存在于主数据与 normal_6000。主数据先加载并具有真实来源，
        # 因此 normal_6000 的重复项必须跳过，不能用另一套 GAT 坐标覆盖它。
        if from_normal6000 and node_id in node_data:
            continue

        node_data[node_id] = {
            "temporal_node_embeddings": temporal_node_embeddings,
            "temporal_text_embeddings": temporal_text_embeddings,
            "timesteps": timesteps_sorted,
            "source": "normal_6000" if from_normal6000 else "main",
        }
        source_edge_drop_map = edge_drop_map_normal if from_normal6000 else edge_drop_map_main
        if node_id in source_edge_drop_map:
            node_data[node_id]["temporal_node_embeddings_aug"] = source_edge_drop_map[node_id]
        elif use_low_memory and node_id in active_ids and node_id not in train_ids:
            edge_dir = (getattr(config, "normal_edge_drop_gat_encoded_dir", "") if from_normal6000
                        else getattr(config, "edge_drop_gat_encoded_dir", ""))
            edge_file = os.path.join(edge_dir, os.path.basename(gat_file))
            if not os.path.isfile(edge_file):
                raise RuntimeError(f"Missing validation edge-drop view: {edge_file}")
            node_data[node_id]["edge_drop_gat_path"] = edge_file
        labels[node_id] = int(label)

        successful_nodes += 1
        if from_normal6000:
            successful_normal6000 += 1
        else:
            successful_main += 1

    if not load_extra_normal:
        for gat_file, _ in paired_files_normal:
            # Use the same canonical identity as load_paired_embeddings.
            # Load one file at a time and discard tensors immediately.
            with open(gat_file, 'rb') as handle:
                identity_payload = pickle.load(handle)
            node_id = str(identity_payload.get('center_node', extract_node_id(gat_file)))
            del identity_payload
            if node_id in node_data:
                continue
            node_data[node_id] = {"source": "normal_6000", "id_stub_only": True}
            labels[node_id] = 0
        print(f"[normal_6000] 仅登记账号 ID，不加载编码张量: {len(paired_files_normal)}")

    print(f"数据加载完成: {successful_nodes} 个节点")
    print(f"  - 主数据集节点: {successful_main}")
    print(f"  - normal_6000 节点: {successful_normal6000}")
    print(f"标签分布: 恶意={sum(labels.values())}, 正常={len(labels) - sum(labels.values())}")
    edge_aug_count = sum(1 for v in node_data.values() if "temporal_node_embeddings_aug" in v)
    print(f"已挂载 edge-drop 增强视图: {edge_aug_count} 个节点")

    if len(paired_files_normal) > 0 and successful_normal6000 == 0:
        print("检测到 normal_6000 配对文件存在，但全部未成功加载。")

    return node_data, labels, graph_data


def train_one_epoch(model, train_loader, contrast_loader, optimizer, scheduler, config, epoch, fold_idx):
    model.train()
    device = config.device

    # 固定配对采样器：每个 epoch 只打乱固定配对的顺序，配对本身不变
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if isinstance(batch_sampler, FixedPairBatchSampler):
        batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_align_loss = 0.0
    total_gen_loss = 0.0
    total_cee_loss = 0.0
    total_ce_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_node_ids = []

    contrast_iter = iter(contrast_loader) if contrast_loader is not None else None
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

        aug_kwargs = {}
        if contrast_iter is None and "struct_in_embeddings_aug" in batch:
            aug_kwargs = {
                "struct_in_embeddings_aug": batch["struct_in_embeddings_aug"].to(device),
                "struct_out_embeddings_aug": batch["struct_out_embeddings_aug"].to(device),
                "struct_in_mask_aug": batch["struct_in_mask_aug"].to(device),
                "struct_out_mask_aug": batch["struct_out_mask_aug"].to(device),
            }

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
            loss_mode="pairwise" if contrast_iter is not None else "all",
            **aug_kwargs,
        )

        loss = outputs["total_loss"]
        align_loss = outputs.get("align_loss", None)

        # L_cont 使用独立随机、无标签 batch；与 PU 固定正负配对完全解耦。
        if contrast_iter is not None:
            try:
                contrast_batch = next(contrast_iter)
            except StopIteration:
                contrast_iter = iter(contrast_loader)
                contrast_batch = next(contrast_iter)

            contrast_aug_kwargs = {}
            if "struct_in_embeddings_aug" in contrast_batch:
                contrast_aug_kwargs = {
                    "struct_in_embeddings_aug": contrast_batch["struct_in_embeddings_aug"].to(device),
                    "struct_out_embeddings_aug": contrast_batch["struct_out_embeddings_aug"].to(device),
                    "struct_in_mask_aug": contrast_batch["struct_in_mask_aug"].to(device),
                    "struct_out_mask_aug": contrast_batch["struct_out_mask_aug"].to(device),
                }
            contrast_outputs = model(
                struct_in_embeddings=contrast_batch["struct_in_embeddings"].to(device),
                struct_out_embeddings=contrast_batch["struct_out_embeddings"].to(device),
                text_node_embeddings=contrast_batch["text_node_embeddings"].to(device),
                struct_in_mask=contrast_batch["struct_in_mask"].to(device),
                struct_out_mask=contrast_batch["struct_out_mask"].to(device),
                text_mask=contrast_batch["text_mask"].to(device),
                timesteps=contrast_batch["timesteps"].to(device),
                attention_mask=contrast_batch["attention_mask"].to(device),
                labels=None,
                loss_mode="contrastive",
                **contrast_aug_kwargs,
            )
            loss = loss + contrast_outputs["total_loss"]
            align_loss = contrast_outputs["align_loss"]
        gen_loss = outputs.get("gen_loss", None)
        cee_loss = outputs.get("cee_loss", None)
        ce_loss = outputs.get("ce_loss", None)

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
        if ce_loss is not None:
            total_ce_loss += ce_loss.item() * batch_size
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
            avg_ce = total_ce_loss / total_samples
            tmp_acc = (np.array(all_preds, dtype=np.int32) ==
                       np.array(all_labels, dtype=np.int32)).mean()
        else:
            avg_loss = avg_align = avg_gen = avg_cee = avg_ce = 0.0
            tmp_acc = 0.0

        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            align=f"{avg_align:.4f}",
            gen=f"{avg_gen:.4f}",
            cee=f"{avg_cee:.4f}",
            ce=f"{avg_ce:.4f}",
            acc=f"{tmp_acc:.4f}",
        )

    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_align = total_align_loss / total_samples
        avg_gen = total_gen_loss / total_samples
        avg_cee = total_cee_loss / total_samples
        avg_ce = total_ce_loss / total_samples
    else:
        avg_loss = avg_align = avg_gen = avg_cee = avg_ce = 0.0

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
        "train_ce_loss": float(avg_ce),
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
    total_ce_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_node_ids = []
    all_scores = []
    all_cee_scores = []

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

            aug_kwargs = {}
            if "struct_in_embeddings_aug" in batch:
                aug_kwargs = {
                    "struct_in_embeddings_aug": batch["struct_in_embeddings_aug"].to(device),
                    "struct_out_embeddings_aug": batch["struct_out_embeddings_aug"].to(device),
                    "struct_in_mask_aug": batch["struct_in_mask_aug"].to(device),
                    "struct_out_mask_aug": batch["struct_out_mask_aug"].to(device),
                }

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
                **aug_kwargs,
            )

            loss = outputs["total_loss"]
            align_loss = outputs.get("align_loss", None)
            gen_loss = outputs.get("gen_loss", None)
            cee_loss = outputs.get("cee_loss", None)
            ce_loss = outputs.get("ce_loss", None)

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
            if ce_loss is not None:
                total_ce_loss += ce_loss.item() * batch_size
            total_samples += batch_size

            class_logits = outputs["logits_2"]  # [B, 2]
            preds = class_logits.argmax(dim=-1)
            class_scores = class_logits[:, 1] - class_logits[:, 0]

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_node_ids.extend(node_ids)
            all_scores.extend(class_scores.detach().cpu().tolist())
            cee_scores = outputs.get("cee_scores")
            if cee_scores is not None:
                all_cee_scores.extend(cee_scores.detach().cpu().tolist())

            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_align = total_align_loss / total_samples
                avg_gen = total_gen_loss / total_samples
                avg_cee = total_cee_loss / total_samples
                avg_ce = total_ce_loss / total_samples
                tmp_acc = (np.array(all_preds, dtype=np.int32) ==
                           np.array(all_labels, dtype=np.int32)).mean()
            else:
                avg_loss = avg_align = avg_gen = avg_cee = avg_ce = 0.0
                tmp_acc = 0.0

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                align=f"{avg_align:.4f}",
                gen=f"{avg_gen:.4f}",
                cee=f"{avg_cee:.4f}",
                ce=f"{avg_ce:.4f}",
                acc=f"{tmp_acc:.4f}",
            )

    if total_samples > 0:
        avg_align = total_align_loss / total_samples
        avg_ce = total_ce_loss / total_samples

        # 验证集保持原生不平衡分布，pairwise loss 必须跨整个验证集计算；
        # 不能按 batch 求均值，否则单类别 batch 会被错误记成零损失。
        score_tensor = torch.tensor(all_scores, dtype=torch.float32)
        label_tensor = torch.tensor(all_labels, dtype=torch.long)
        pos_score = score_tensor[label_tensor == 1]
        neg_score = score_tensor[label_tensor == 0]
        if pos_score.numel() and neg_score.numel():
            avg_gen = float(torch.relu(
                float(config.cls_margin) - (pos_score[:, None] - neg_score[None, :])
            ).mean().item())
        else:
            avg_gen = 0.0

        if float(config.lambda_cee) > 0.0 and len(all_cee_scores) == len(all_labels):
            cee_tensor = torch.tensor(all_cee_scores, dtype=torch.float32)
            cib_cee = cee_tensor[label_tensor == 1]
            auth_cee = cee_tensor[label_tensor == 0]
            if cib_cee.numel() and auth_cee.numel():
                avg_cee = float(torch.relu(
                    cib_cee[:, None] - auth_cee[None, :] + float(config.cee_margin)
                ).mean().item())
            else:
                avg_cee = 0.0
        else:
            avg_cee = 0.0

        avg_loss = (
            float(config.lambda_gen) * avg_gen
            + float(config.lambda_align) * avg_align
            + float(config.lambda_cee) * avg_cee
        )
    else:
        avg_loss = avg_align = avg_gen = avg_cee = avg_ce = 0.0

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
        "val_ce_loss": float(avg_ce),
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


def train_one_fold(model: UnifiedTemporalGTLLM, train_loader, contrast_loader, val_loader,
                   config: TrainingConfig, fold_idx: int, output_dir: str, tokenizer=None) -> Dict[str, float]:
    """单次训练 run"""

    print(f"\n{'='*60}")
    print(f"开始单次训练（Run {fold_idx+1}）")
    print(f"{'='*60}")

    lora_params = []
    frontend_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name.lower():
            lora_params.append(param)
        else:
            frontend_params.append(param)
    optimizer = optim.AdamW(
        [
            {"params": frontend_params, "lr": config.learning_rate},
            {"params": lora_params, "lr": config.lora_learning_rate},
        ],
        weight_decay=config.optimizer_weight_decay,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_eps,
    )

    total_steps = len(train_loader) * config.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.scheduler_eta_min)

    train_history = []
    best_val_loss = float('inf')
    best_model_path = None

    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, contrast_loader, optimizer, scheduler, config, epoch, fold_idx
        )
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

        best_theta, best_f1, best_stats = grid_search_gen_threshold(
            s_gen_val, y_val
        )

        print(f"\n最佳阈值搜索结果（单阈值，仅验证集）:")
        print(f"  θ (theta) = {best_theta:.4f}")
        print(f"  验证集 F1 = {best_f1:.4f}")
        print(f"  验证集统计: {best_stats}")

        fold_summary["best_thresholds"] = {
            "theta": float(best_theta),
            "val_f1": float(best_f1),
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


def split_class_counts(node_ids: List[str], labels: Dict[str, int]) -> Dict[str, int]:
    mal = sum(1 for nid in node_ids if labels.get(nid, 0) == 1)
    return {"total": len(node_ids), "malicious": mal, "normal": len(node_ids) - mal}


def build_split_payload(
    train_node_ids: List[str],
    val_node_ids: List[str],
    test_node_ids: List[str],
    extra_normal_val_ids: List[str],
    extra_normal_test_ids: List[str],
    labels: Dict[str, int],
    config: TrainingConfig,
) -> Dict:
    return {
        "split_protocol_version": 2,
        "train_node_ids": train_node_ids,
        "val_node_ids": val_node_ids,
        "test_node_ids": test_node_ids,
        "extra_normal_val_ids": extra_normal_val_ids,
        "extra_normal_test_ids": extra_normal_test_ids,
        "split_seed": getattr(config, "split_seed", 42),
        "extra_normal_seed": getattr(config, "extra_normal_seed", 0),
        "extra_normal_inject_each": int(getattr(config, "extra_normal_inject_each", 3000)),
        "class_counts": {
            "train": split_class_counts(train_node_ids, labels),
            "val": split_class_counts(val_node_ids, labels),
            "test": split_class_counts(test_node_ids, labels),
        },
    }


def assert_split_protocol(payload: Dict, labels: Dict[str, int]) -> None:
    counts = payload.get("class_counts") or {
        "train": split_class_counts(payload.get("train_node_ids", []), labels),
        "val": split_class_counts(payload.get("val_node_ids", []), labels),
        "test": split_class_counts(payload.get("test_node_ids", []), labels),
    }

    train = counts["train"]
    if train["malicious"] != train["normal"]:
        raise RuntimeError(f"训练集不是 1:1: {train}")

    # 额外注入的 normal 必须全部是正常账号，且不得出现在训练集（防 CEE 预训练泄露）
    train_set = set(payload.get("train_node_ids", []))
    for key in ("extra_normal_val_ids", "extra_normal_test_ids"):
        bad = [nid for nid in payload.get(key, []) if labels.get(nid, 0) != 0 or nid in train_set]
        if bad:
            raise RuntimeError(f"{key} 含训练集账号或非正常账号: {len(bad)} 个")

    split_sets = [
        set(payload.get("train_node_ids", [])),
        set(payload.get("val_node_ids", [])),
        set(payload.get("test_node_ids", [])),
    ]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise RuntimeError("Train/Val/Test 存在账号交叉")


def can_reuse_split(split_data: Dict, config: TrainingConfig, labels: Dict[str, int], node_data: Dict) -> bool:
    if not bool(getattr(config, "reuse_existing_split", True)):
        return False

    if split_data.get("split_protocol_version") != 2:
        print("已有划分不是当前 main-only-v2 协议，将重新划分。")
        return False

    expected_seed = getattr(config, "split_seed", 42)
    expected_extra_seed = getattr(config, "extra_normal_seed", 0)
    if split_data.get("split_seed", None) != expected_seed:
        print(f"已有划分 split_seed={split_data.get('split_seed', None)} 与当前配置 {expected_seed} 不一致，将重新划分。")
        return False
    if split_data.get("extra_normal_seed", None) != expected_extra_seed:
        print(f"已有划分 extra_normal_seed={split_data.get('extra_normal_seed', None)} 与当前配置 {expected_extra_seed} 不一致，将重新划分。")
        return False

    all_ids = split_data.get("train_node_ids", []) + split_data.get("val_node_ids", []) + split_data.get("test_node_ids", [])
    missing_ids = [nid for nid in all_ids if nid not in node_data or nid not in labels]
    if missing_ids:
        print(f"已有划分中有 {len(missing_ids)} 个节点当前未加载，将重新划分。")
        return False

    if any(node_data[nid].get("source", "main") != "main" for nid in split_data.get("train_node_ids", [])):
        print("已有划分的训练集含 normal_6000 账号，将重新划分。")
        return False

    try:
        payload = build_split_payload(
            split_data.get("train_node_ids", []),
            split_data.get("val_node_ids", []),
            split_data.get("test_node_ids", []),
            split_data.get("extra_normal_val_ids", []),
            split_data.get("extra_normal_test_ids", []),
            labels,
            config,
        )
        assert_split_protocol(payload, labels)
    except RuntimeError as exc:
        print(f"已有划分不满足当前协议，将重新划分: {exc}")
        return False

    return True


def get_or_create_split(config: TrainingConfig, node_data: Dict, labels: Dict[str, int]):
    """复用或重建 train/val/test 划分；与 pretrain_cee_head.py 共享 dataset_split.json，
    保证 CEE 预训练只使用训练集账号（修改方向.md 一.1 / 四.2）。"""
    split_path = os.path.join(config.output_dir, "dataset_split.json")
    os.makedirs(config.output_dir, exist_ok=True)

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        if can_reuse_split(split_data, config, labels, node_data):
            print(f"复用已有数据集划分: {split_path}")
            return (
                list(split_data["train_node_ids"]),
                list(split_data["val_node_ids"]),
                list(split_data["test_node_ids"]),
                list(split_data.get("extra_normal_val_ids", [])),
                list(split_data.get("extra_normal_test_ids", [])),
            )
        raise RuntimeError(
            f'Existing split is incompatible: {split_path}. Refusing to overwrite or resample it. '
            'Resolve the identity/configuration mismatch explicitly, or use a new experiment directory.'
        )

    # 6:2:2 只在原始主数据集上划分；normal_6000 是额外不平衡评估池，
    # 不得被类别平衡逻辑抽入训练集。
    labeled_node_ids = [
        nid for nid, item in node_data.items()
        if item.get("source", "main") == "main" and nid in labels
    ]
    print(f"主数据有标签节点: {len(labeled_node_ids)}")

    # 全局类别平衡（可选）
    if config.malicious_to_normal_ratio != 1.0:
        labeled_node_ids = balance_dataset(
            labeled_node_ids, labels, config.malicious_to_normal_ratio
        )

    # 分层 6:2:2 划分 train / val / test，并保证各子集恶意/正常 1:1
    train_node_ids, val_node_ids, test_node_ids = stratified_train_val_test_split(
        labeled_node_ids,
        labels,
        train_ratio=getattr(config, "train_ratio", 0.6),
        val_ratio=getattr(config, "val_ratio", 0.2),
        test_ratio=getattr(config, "test_ratio", 0.2),
        seed=getattr(config, "split_seed", 42),
    )

    # 将额外的 normal_6000 正常用户平均分配到 Val / Test
    all_split_ids = set(train_node_ids) | set(val_node_ids) | set(test_node_ids)
    extra_ids = [
        nid for nid, item in node_data.items()
        if item.get("source") == "normal_6000" and nid not in all_split_ids
    ]
    extra_normal_ids = [nid for nid in extra_ids if labels.get(nid, 0) == 0]

    rng_extra = np.random.RandomState(int(getattr(config, "extra_normal_seed", 0)))
    rng_extra.shuffle(extra_normal_ids)

    extra_each = int(getattr(config, "extra_normal_inject_each", 3000))
    extra_val_ids = extra_normal_ids[:extra_each]
    extra_test_ids = extra_normal_ids[extra_each: 2 * extra_each]

    val_node_ids = list(val_node_ids) + list(extra_val_ids)
    test_node_ids = list(test_node_ids) + list(extra_test_ids)

    payload = build_split_payload(
        train_node_ids, val_node_ids, test_node_ids,
        extra_val_ids, extra_test_ids, labels, config,
    )
    assert_split_protocol(payload, labels)
    save_json(split_path, payload)
    print(f"数据集划分结果已保存到: {split_path}")

    return train_node_ids, val_node_ids, test_node_ids, extra_val_ids, extra_test_ids


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
    if cee_path and os.path.isfile(cee_path):
        checkpoint = torch.load(cee_path, map_location='cpu', weights_only=False)
        recorded_ids = checkpoint.get('train_node_ids')
        if recorded_ids is None:
            print('[CEE reuse warning] Legacy head has no training-account list; membership was checked separately, not by this checkpoint.')
        elif set(recorded_ids) != set(train_node_ids):
            raise RuntimeError('CEE training accounts differ from the fixed main training split; refusing reuse.')
        del checkpoint


def main():
    """主函数：单次 6:2:2 分层划分 + 微调"""
    print("=" * 80)
    print("统一LoRA微调训练 - 端到端联合训练所有参数")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--lambda-cee", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from config import load_config
    config = load_config(args.config)
    if args.lambda_cee is not None:
        config.lambda_cee = float(args.lambda_cee)
    if args.output_dir:
        config.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in config.__dict__.items()}
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"配置保存到: {config_path}")

    # 加载全部节点数据与标签
    node_data, labels, graph_data = load_all_data(config, train_only=True)

    if len(node_data) == 0 or len(labels) == 0:
        raise RuntimeError('No valid training data; refusing to report a successful stage.')

    some_id = next(iter(node_data.keys()))
    print("示例节点字段:", node_data[some_id].keys())

    # 固定全局随机种子（修改方向.md 二.2 可复现性）
    set_seed(int(config.training_seed))

    # 复用或重建 train/val/test 划分（与 pretrain_cee_head.py 共享同一份 dataset_split.json）
    train_node_ids, val_node_ids, test_node_ids, extra_val_ids, extra_test_ids = get_or_create_split(
        config, node_data, labels
    )

    print("\n[Extra normal_6000] 注入 Val/Test：")
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

    # 训练前校验：edge-drop 视图齐全 + 冻结 CEE checkpoint 存在
    verify_training_assets(train_node_ids, node_data, config)

    # 加载分词器
    print("\n加载分词器...")
    from backbone_adapter import load_backbone_tokenizer
    tokenizer = load_backbone_tokenizer(config.llm_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构造 DataLoader（仅 train / val）
    train_loader, contrast_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        train_node_ids, val_node_ids, node_data, labels, config
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")

    if len(train_dataset) == 0:
        raise RuntimeError('Training dataset is empty after validation.')

    # 初始化模型并训练（单次 run）
    print("\n初始化模型...")
    model = UnifiedTemporalGTLLM(config)
    from run_metadata import save_runtime_metadata
    save_runtime_metadata(config, model)
    model.set_tokenizer(tokenizer)

    fold_idx = 0  # 单次 run 视作 fold 0
    res = train_one_fold(
        model, train_loader, contrast_loader, val_loader,
        config, fold_idx, config.output_dir, tokenizer,
    )

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
