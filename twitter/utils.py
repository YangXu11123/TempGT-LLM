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
    """从文件名中提取节点ID"""
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
    except:
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


def balance_dataset(node_ids: List[str], labels: Dict, ratio: float = 1.0) -> List[str]:
    """全局平衡节点集合，使恶意:正常 ≈ ratio:1"""
    malicious_nodes = [nid for nid in node_ids if labels.get(nid, 0) == 1]
    benign_nodes = [nid for nid in node_ids if labels.get(nid, 0) == 0]

    if len(malicious_nodes) == 0 or len(benign_nodes) == 0:
        print("无法平衡（仅单一类别）")
        return node_ids

    if ratio < 1.0:
        num_benign = int(len(malicious_nodes) / ratio)
        benign_nodes = np.random.choice(benign_nodes, min(num_benign, len(benign_nodes)), replace=False).tolist()
    elif ratio > 1.0:
        num_malicious = int(len(benign_nodes) * ratio)
        malicious_nodes = np.random.choice(malicious_nodes, min(num_malicious, len(malicious_nodes)), replace=False).tolist()

    balanced = malicious_nodes + benign_nodes
    print(f"平衡后: 恶意={len(malicious_nodes)}, 正常={len(benign_nodes)}, 总计={len(balanced)}")
    return balanced


def stratified_train_val_test_split(
    node_ids: List[str],
    labels: Dict[str, int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """分层 6:2:2 划分"""
    malicious = [nid for nid in node_ids if labels.get(nid, 0) == 1]
    benign = [nid for nid in node_ids if labels.get(nid, 0) == 0]

    rng = np.random.RandomState(seed)

    if len(malicious) == 0 or len(benign) == 0:
        print("单一类别，进行普通随机划分。")
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
    c_test = n_per_class - c_train - c_val

    mal_train, mal_val, mal_test = malicious[:c_train], malicious[c_train:c_train + c_val], malicious[c_train + c_val:]
    ben_train, ben_val, ben_test = benign[:c_train], benign[c_train:c_train + c_val], benign[c_train + c_val:]

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
