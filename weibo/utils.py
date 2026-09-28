import torch
import numpy as np
import pickle
import json
import os
import re
import glob
import sys
import types
from typing import Dict, List, Tuple, Optional, Any


def _ensure_torch_geometric_pickle_compat() -> None:
    """Allow label-only loading of PyG pickles when torch_geometric is absent."""
    try:
        import torch_geometric  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    if 'torch_geometric.data.data' in sys.modules and 'torch_geometric.data.storage' in sys.modules:
        return

    class _DummyPyGObject:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__['_state'] = state

    tg_mod = types.ModuleType('torch_geometric')
    data_mod = types.ModuleType('torch_geometric.data')
    data_data_mod = types.ModuleType('torch_geometric.data.data')
    storage_mod = types.ModuleType('torch_geometric.data.storage')

    for mod in (tg_mod, data_mod, data_data_mod, storage_mod):
        sys.modules.setdefault(mod.__name__, mod)

    for cls_name in ('Data', 'DataEdgeAttr', 'DataTensorAttr'):
        cls = type(cls_name, (_DummyPyGObject,), {'__module__': 'torch_geometric.data.data'})
        setattr(sys.modules['torch_geometric.data.data'], cls_name, cls)
        setattr(sys.modules['torch_geometric.data'], cls_name, cls)

    for cls_name in ('BaseStorage', 'GlobalStorage', 'NodeStorage', 'EdgeStorage'):
        cls = type(cls_name, (_DummyPyGObject,), {'__module__': 'torch_geometric.data.storage'})
        setattr(sys.modules['torch_geometric.data.storage'], cls_name, cls)
        setattr(sys.modules['torch_geometric.data'], cls_name, cls)

    sys.modules['torch_geometric'].data = sys.modules['torch_geometric.data']
    sys.modules['torch_geometric.data'].data = sys.modules['torch_geometric.data.data']
    sys.modules['torch_geometric.data'].storage = sys.modules['torch_geometric.data.storage']


def load_original_graph_data(data_path: str) -> Dict:
    """加载原始图数据"""
    try:
        _ensure_torch_geometric_pickle_compat()
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"加载原始图数据成功: {data_path}")
        return data
    except Exception as e:
        print(f"加载原始图数据失败: {e}")
        return {}



def extract_node_id(filename: str) -> str:
    """从文件名提取节点ID"""
    match = re.search(r'subgraph_([^_]+)_k2_(?:gat_encoded|text_encoded)\.pkl', filename)
    if match:
        return match.group(1)

    match = re.search(r'subgraph_(\d+)_', filename)
    if match:
        return match.group(1)

    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 2:
        return parts[1]
    return "unknown"



def find_paired_files(gat_dir: str, text_dir: str) -> List[Tuple[str, str]]:
    """查找配对的 GAT 和 文本 编码文件"""
    gat_files = glob.glob(os.path.join(gat_dir, "subgraph_*_k2_gat_encoded.pkl"))
    text_files = glob.glob(os.path.join(text_dir, "subgraph_*_k2_text_encoded.pkl"))

    print(f"找到GAT文件: {len(gat_files)}, 文本文件: {len(text_files)}")

    gat_dict = {extract_node_id(f): f for f in gat_files}
    text_dict = {extract_node_id(f): f for f in text_files}

    common_ids = set(gat_dict.keys()) & set(text_dict.keys())
    paired_files = [(gat_dict[nid], text_dict[nid]) for nid in sorted(common_ids)]

    print(f"配对成功: {len(paired_files)} 对")
    return paired_files



def _clean_temporal_embeddings_dict(data: Dict[Any, Any]) -> Dict[int, Any]:
    """清洗时间步 -> 嵌入字典"""
    if not isinstance(data, dict):
        return {}

    cleaned: Dict[int, Any] = {}

    for k, v in data.items():
        try:
            t = int(k)
        except Exception:
            continue

        if isinstance(v, dict):
            sub_dict = {}
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    tensor = torch.nan_to_num(sub_v.detach().cpu(), nan=0.0)
                else:
                    arr = np.array(sub_v, dtype=np.float32)
                    tensor = torch.nan_to_num(torch.from_numpy(arr), nan=0.0)
                sub_dict[sub_k] = tensor
            cleaned[t] = sub_dict
        else:
            if isinstance(v, torch.Tensor):
                tensor = torch.nan_to_num(v.detach().cpu(), nan=0.0)
            else:
                arr = np.array(v, dtype=np.float32)
                tensor = torch.nan_to_num(torch.from_numpy(arr), nan=0.0)
            cleaned[t] = tensor

    return cleaned



def load_paired_embeddings(gat_file: str, text_file: str) -> Optional[Dict[str, Any]]:
    """加载 GAT / SBERT 的节点级时序嵌入"""
    try:
        with open(gat_file, 'rb') as f:
            gat_data = pickle.load(f)
        with open(text_file, 'rb') as f:
            text_data = pickle.load(f)
    except Exception as e:
        print(f"加载 GAT / 文本 编码文件失败: {gat_file}, {text_file}, 错误: {e}")
        return None

    node_id = None
    if isinstance(gat_data, dict) and 'center_node' in gat_data:
        node_id = str(gat_data['center_node'])
    elif isinstance(text_data, dict) and 'center_node' in text_data:
        node_id = str(text_data['center_node'])
    if node_id is None:
        node_id = extract_node_id(gat_file)

    temporal_node_embeddings_raw = gat_data.get('temporal_node_embeddings', {})
    temporal_text_embeddings_raw = text_data.get('temporal_text_embeddings', {})

    if not temporal_node_embeddings_raw and not temporal_text_embeddings_raw:
        print(f"{node_id}: 结构与文本的时间步嵌入均为空，跳过。")
        return None

    temporal_node_embeddings = _clean_temporal_embeddings_dict(temporal_node_embeddings_raw)
    temporal_text_embeddings = _clean_temporal_embeddings_dict(temporal_text_embeddings_raw)

    timestep_set = set()

    for src in (gat_data, text_data):
        if isinstance(src, dict) and 'timesteps' in src:
            ts = src['timesteps']
            if isinstance(ts, torch.Tensor):
                ts_list = ts.detach().cpu().long().tolist()
            else:
                ts_list = list(ts)
            ts_list = [int(x) for x in ts_list]
            timestep_set.update(ts_list)

    timestep_set.update(temporal_node_embeddings.keys())
    timestep_set.update(temporal_text_embeddings.keys())

    timesteps = sorted(int(t) for t in timestep_set)

    if len(timesteps) == 0:
        print(f"{node_id}: 无有效时间步")
        return None

    return {
        "node_id": node_id,
        "temporal_node_embeddings": temporal_node_embeddings,
        "temporal_text_embeddings": temporal_text_embeddings,
        "timesteps": timesteps,
        "seq_len": len(timesteps),
    }


def load_gat_embeddings(gat_file: str) -> Optional[Dict[str, Any]]:
    """加载单独的 GAT 编码文件，用于 edge-drop 结构增强视图。"""
    try:
        with open(gat_file, 'rb') as f:
            gat_data = pickle.load(f)
    except Exception as e:
        print(f"加载 GAT 编码文件失败: {gat_file}, 错误: {e}")
        return None

    node_id = str(gat_data.get('center_node', extract_node_id(gat_file))) if isinstance(gat_data, dict) else extract_node_id(gat_file)
    temporal_node_embeddings_raw = gat_data.get('temporal_node_embeddings', {}) if isinstance(gat_data, dict) else {}
    temporal_node_embeddings = _clean_temporal_embeddings_dict(temporal_node_embeddings_raw)

    timestep_set = set(temporal_node_embeddings.keys())
    if isinstance(gat_data, dict) and 'timesteps' in gat_data:
        ts = gat_data['timesteps']
        if isinstance(ts, torch.Tensor):
            timestep_set.update(int(x) for x in ts.detach().cpu().view(-1).long().tolist())
        else:
            try:
                timestep_set.update(int(x) for x in list(ts))
            except Exception:
                pass

    if not temporal_node_embeddings or len(timestep_set) == 0:
        return None

    return {
        "node_id": node_id,
        "temporal_node_embeddings": temporal_node_embeddings,
        "timesteps": sorted(int(t) for t in timestep_set),
    }



def get_user_label(node_id: str, graph_data: Dict) -> Tuple[bool, int]:
    """获取用户标签（恶意=1 正常=0）"""
    malicious_users = graph_data.get('malicious_users', set())

    id_to_user = graph_data.get('id_to_user', {})
    user_id_to_names = graph_data.get('user_id_to_names', {})
    screen_name_to_user_id = graph_data.get('screen_name_to_user_id', {})

    if node_id in malicious_users:
        return True, 1

    try:
        node_id_int = int(node_id)
        if node_id_int in id_to_user:
            name = id_to_user[node_id_int]
            return True, 1 if name in malicious_users else 0
    except Exception:
        pass

    if node_id in user_id_to_names:
        for name in user_id_to_names[node_id]:
            if name in malicious_users:
                return True, 1
        return True, 0

    if node_id in screen_name_to_user_id:
        uid = screen_name_to_user_id[node_id]
        return True, 1 if uid in malicious_users else 0

    return True, 0



def stratified_train_val_test_split(
    node_ids: List[str],
    labels: Dict[str, int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """分层 6:2:2 划分，保证恶意与正常尽量 1:1"""
    malicious = [nid for nid in node_ids if labels.get(nid, 0) == 1]
    benign = [nid for nid in node_ids if labels.get(nid, 0) == 0]

    rng = np.random.RandomState(seed)

    if len(malicious) == 0 or len(benign) == 0:
        print("单一类别，退化为普通随机划分。")
        shuffled = rng.permutation(node_ids)
        N = len(shuffled)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        return (
            shuffled[:n_train].tolist(),
            shuffled[n_train:n_train + n_val].tolist(),
            shuffled[n_train + n_val:].tolist()
        )

    n_per_class = min(len(malicious), len(benign))

    malicious = rng.permutation(malicious)[:n_per_class]
    benign = rng.permutation(benign)[:n_per_class]

    c_train = int(n_per_class * train_ratio)
    c_val = int(n_per_class * val_ratio)

    mal_train = malicious[:c_train]
    mal_val = malicious[c_train:c_train + c_val]
    mal_test = malicious[c_train + c_val:]
    ben_train = benign[:c_train]
    ben_val = benign[c_train:c_train + c_val]
    ben_test = benign[c_train + c_val:]

    train_nodes = rng.permutation(list(mal_train) + list(ben_train)).tolist()
    val_nodes = rng.permutation(list(mal_val) + list(ben_val)).tolist()
    test_nodes = rng.permutation(list(mal_test) + list(ben_test)).tolist()

    print(
        f"6:2:2 分层划分完成:\n"
        f"  train={len(train_nodes)}\n"
        f"  val={len(val_nodes)}\n"
        f"  test={len(test_nodes)}"
    )
    return train_nodes, val_nodes, test_nodes



def _sample_benign_for_val_test(
    benign_remaining: List[str],
    n_mal_val: int,
    n_mal_test: int,
    rng: np.random.RandomState,
    val_test_neg_pos_ratio: Optional[int] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    为 val/test 采样正常样本。

    参数:
        val_test_neg_pos_ratio:
            - None: 沿用当前策略，即把剩余正常样本平均分给 val/test
            - 正整数 r: 令 val/test 分别尽量满足 恶意:正常 = 1:r
    """
    info: Dict[str, Any] = {
        "strategy": "current" if val_test_neg_pos_ratio is None else f"1:{val_test_neg_pos_ratio}",
        "requested_ratio": None if val_test_neg_pos_ratio is None else val_test_neg_pos_ratio,
    }

    if val_test_neg_pos_ratio is None:
        half = len(benign_remaining) // 2
        ben_val = benign_remaining[:half]
        ben_test = benign_remaining[half:]
        info.update({
            "ben_val_count": len(ben_val),
            "ben_test_count": len(ben_test),
            "actual_val_ratio": (len(ben_val) / max(n_mal_val, 1)),
            "actual_test_ratio": (len(ben_test) / max(n_mal_test, 1)),
            "remaining_after_sampling": 0,
        })
        return ben_val, ben_test, info

    if not isinstance(val_test_neg_pos_ratio, int) or val_test_neg_pos_ratio <= 0:
        raise ValueError("val_test_neg_pos_ratio 必须是 None 或正整数，例如 10/20/50。")

    need_ben_val = n_mal_val * val_test_neg_pos_ratio
    need_ben_test = n_mal_test * val_test_neg_pos_ratio

    ben_val = benign_remaining[:min(need_ben_val, len(benign_remaining))]
    remaining_after_val = benign_remaining[len(ben_val):]
    ben_test = remaining_after_val[:min(need_ben_test, len(remaining_after_val))]
    leftover = remaining_after_val[len(ben_test):]

    info.update({
        "need_ben_val": need_ben_val,
        "need_ben_test": need_ben_test,
        "ben_val_count": len(ben_val),
        "ben_test_count": len(ben_test),
        "actual_val_ratio": (len(ben_val) / max(n_mal_val, 1)),
        "actual_test_ratio": (len(ben_test) / max(n_mal_test, 1)),
        "remaining_after_sampling": len(leftover),
    })

    return ben_val, ben_test, info



def split_train_balanced_val_test_imbalanced(
    node_ids: List[str],
    labels: Dict[str, int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    val_test_neg_pos_ratio: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    1) 恶意节点始终按 6:2:2 分到 train/val/test。
    2) 训练集：在正常节点池中随机采样，使 train 中 正常数量 == train 中 恶意数量（1:1）。
    3) 验证/测试集：
       - 若 val_test_neg_pos_ratio is None，则沿用当前策略：把训练后剩余的正常节点平均分给 val/test。
       - 若 val_test_neg_pos_ratio = r，则让 val/test 尽量满足 恶意:正常 = 1:r。
    4) 返回的三个列表都会再次随机打乱。
    """
    rng = np.random.RandomState(seed)

    malicious = [nid for nid in node_ids if labels.get(nid, 0) == 1]
    benign = [nid for nid in node_ids if labels.get(nid, 0) == 0]

    if len(malicious) == 0 or len(benign) == 0:
        print("单一类别，退化为普通随机划分。")
        shuffled = rng.permutation(node_ids)
        N = len(shuffled)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        return (
            shuffled[:n_train].tolist(),
            shuffled[n_train:n_train + n_val].tolist(),
            shuffled[n_train + n_val:].tolist(),
        )

    # 恶意节点：严格 6:2:2 划分
    mal_shuf = rng.permutation(malicious).tolist()
    N_mal = len(mal_shuf)

    m_train = int(N_mal * train_ratio)
    m_val = int(N_mal * val_ratio)

    mal_train = mal_shuf[:m_train]
    mal_val = mal_shuf[m_train:m_train + m_val]
    mal_test = mal_shuf[m_train + m_val:]

    # 训练集正常：按 train 恶意数量 1:1 采样
    ben_shuf = rng.permutation(benign).tolist()

    need_ben_train = len(mal_train)
    if need_ben_train > len(ben_shuf):
        print(f"正常节点不足以实现训练集 1:1：需要 {need_ben_train}，仅有 {len(ben_shuf)}。将使用全部正常节点。")
        ben_train = ben_shuf
        ben_remaining = []
    else:
        ben_train = ben_shuf[:need_ben_train]
        ben_remaining = ben_shuf[need_ben_train:]

    ben_val, ben_test, ratio_info = _sample_benign_for_val_test(
        benign_remaining=ben_remaining,
        n_mal_val=len(mal_val),
        n_mal_test=len(mal_test),
        rng=rng,
        val_test_neg_pos_ratio=val_test_neg_pos_ratio,
    )

    train_nodes = rng.permutation(mal_train + ben_train).tolist()
    val_nodes = rng.permutation(mal_val + ben_val).tolist()
    test_nodes = rng.permutation(mal_test + ben_test).tolist()

    print(
        "数据集划分完成：\n"
        f"  恶意: 总={len(malicious)} | train={len(mal_train)} val={len(mal_val)} test={len(mal_test)}\n"
        f"  正常: 总={len(benign)} | train采样={len(ben_train)} 剩余={len(ben_remaining)} -> val={len(ben_val)} test={len(ben_test)}\n"
        f"  val/test 策略: {ratio_info['strategy']} | actual val=1:{ratio_info['actual_val_ratio']:.2f} test=1:{ratio_info['actual_test_ratio']:.2f}\n"
        f"  结果: train={len(train_nodes)} val={len(val_nodes)} test={len(test_nodes)}"
    )

    if val_test_neg_pos_ratio is not None:
        need_val = ratio_info.get('need_ben_val', 0)
        need_test = ratio_info.get('need_ben_test', 0)
        if len(ben_val) < need_val or len(ben_test) < need_test:
            print(
                "[警告] 剩余正常样本不足，无法完全满足目标 val/test 比例。"
                f"  目标: val 1:{val_test_neg_pos_ratio}, test 1:{val_test_neg_pos_ratio}\n"
                f"  实际: val 1:{ratio_info['actual_val_ratio']:.2f}, test 1:{ratio_info['actual_test_ratio']:.2f}"
            )

    return train_nodes, val_nodes, test_nodes
