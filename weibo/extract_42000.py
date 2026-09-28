import pickle
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import Tuple, Set, Dict, Optional, List
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd


class TemporalKHopSubgraphExtractor:
    def __init__(self, temporal_graphs_path: str):
        """初始化时序子图提取器"""
        self.load_temporal_graphs(temporal_graphs_path)

    def load_temporal_graphs(self, load_path: str):
        """加载时序图数据"""
        print(f"正在从 {load_path} 加载时序图数据...")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.temporal_graphs = data['temporal_graphs']

        # 兼容新旧数据格式
        if 'user_to_primary_id' in data:
            self.user_to_id = data['user_to_primary_id']
            self.id_to_user = data.get('id_to_user', {})
        else:
            self.user_to_id = data.get('user_to_id', {})
            self.id_to_user = data.get('id_to_user', {})

        # 用户文本数据
        self.user_texts = data.get('user_texts', {})

        self.malicious_users = data.get('malicious_users', set())

        # 处理时序图格式
        if isinstance(self.temporal_graphs, list):
            temporal_graphs_dict = {}
            for i, graph in enumerate(self.temporal_graphs):
                temporal_graphs_dict[i] = graph
            self.temporal_graphs = temporal_graphs_dict

        self.time_steps = sorted(self.temporal_graphs.keys())
        print(f"已加载 {len(self.temporal_graphs)} 个时间步的图数据")
        print(f"用户映射: {len(self.user_to_id)} 个用户")

    def load_malicious_users_from_file(self, malicious_users_path: str) -> Set[str]:
        """从文件加载恶意用户列表（保留函数，但本次不使用）"""
        print(f"正在从 {malicious_users_path} 加载恶意用户列表...")

        with open(malicious_users_path, 'r', encoding='gbk') as f:
            content = f.read()

        import ast
        set_start = content.find('{')
        set_end = content.rfind('}') + 1

        if set_start != -1 and set_end > set_start:
            set_content = content[set_start:set_end]
            try:
                malicious_set = ast.literal_eval(set_content)
            except (ValueError, SyntaxError):
                import re
                pattern = r"'([^']+)'"
                matches = re.findall(pattern, set_content)
                malicious_set = set(matches)
        else:
            malicious_set = set()

        print(f"已加载 {len(malicious_set)} 个恶意用户")
        return malicious_set

    def sample_normal_users_fixed(self, malicious_users: Set[str], num_normal: int = 12000) -> Set[str]:
        """
        固定数量采样正常用户（策略A：不足则全取，不重复采样）
        """
        all_users = set(self.user_to_id.keys())
        normal_users = all_users - malicious_users

        if len(normal_users) == 0:
            print("警告: 没有可用的正常用户可采样")
            return set()

        num_to_sample = min(num_normal, len(normal_users))
        sampled_normal_users = set(random.sample(list(normal_users), num_to_sample))

        print(f"从 {len(normal_users)} 个正常用户中采样了 {len(sampled_normal_users)} 个（目标={num_normal}，策略A：不足则全取）")
        return sampled_normal_users

    def find_user_id(self, user_identifier):
        """查找用户ID"""
        if isinstance(user_identifier, str):
            return self.user_to_id.get(user_identifier)
        elif isinstance(user_identifier, (int, np.integer)):
            return int(user_identifier)
        return None

    def check_node_existence(self, node_id: int) -> Dict[int, bool]:
        """检查节点在每个时间步的存在性"""
        existence = {}

        for time_step in self.time_steps:
            full_graph = self.temporal_graphs[time_step]
            exists = False

            try:
                if hasattr(full_graph, 'original_node_ids') and full_graph.original_node_ids.numel() > 0:
                    original_ids = full_graph.original_node_ids.cpu().numpy()
                    exists = node_id in original_ids
                else:
                    exists = False

            except Exception as e:
                print(f"检查时间步 {time_step} 时出错: {e}")
                exists = False

            existence[time_step] = exists

        return existence

    def get_original_to_idx_mapping(self, full_graph: Data) -> Dict[int, int]:
        """获取原始节点ID到PyG索引的映射"""
        original_to_idx = {}

        try:
            if hasattr(full_graph, 'original_node_ids') and full_graph.original_node_ids.numel() > 0:
                original_ids = full_graph.original_node_ids.cpu().numpy()
                for idx, original_id in enumerate(original_ids):
                    original_to_idx[int(original_id)] = idx
        except Exception as e:
            print(f"获取节点映射时出错: {e}")

        return original_to_idx

    def build_networkx_graph_with_original_ids(self, full_graph: Data) -> Optional[nx.DiGraph]:
        """使用原始节点ID构建NetworkX图"""
        try:
            if not hasattr(full_graph, 'edge_index') or full_graph.edge_index.numel() == 0:
                return None

            original_to_idx = self.get_original_to_idx_mapping(full_graph)
            if not original_to_idx:
                return None

            idx_to_original = {v: k for k, v in original_to_idx.items()}

            edge_index = full_graph.edge_index.cpu().numpy()
            G = nx.DiGraph()

            for original_id in original_to_idx.keys():
                G.add_node(original_id)

            for i in range(edge_index.shape[1]):
                source_idx, target_idx = edge_index[:, i]
                if source_idx in idx_to_original and target_idx in idx_to_original:
                    source_id = idx_to_original[source_idx]
                    target_id = idx_to_original[target_idx]
                    if source_id not in G:
                        G.add_node(source_id)
                    if target_id not in G:
                        G.add_node(target_id)
                    G.add_edge(source_id, target_id)

            return G

        except Exception as e:
            print(f"构建NetworkX图时出错: {e}")
            return None

    def extract_temporal_k_hop_subgraph(self, center_node_identifier, k: int = 2) -> Dict[int, Tuple[Optional[Data], Optional[Data]]]:
        """提取时序累积k跳子图 - 使用原始节点ID系统"""
        center_node_id = self.find_user_id(center_node_identifier)

        if center_node_id is None:
            print(f"错误: 无法找到节点 {center_node_identifier}")
            return {}

        existence = self.check_node_existence(center_node_id)
        existing_steps = [step for step, exists in existence.items() if exists]

        if not existing_steps:
            print(f"节点 {center_node_identifier} 在任何时间步都不存在!")
            return {}

        first_appearance = min(existing_steps)
        temporal_subgraphs = {}

        cumulative_out_nodes = set()
        cumulative_in_nodes = set()

        for time_step in self.time_steps:
            if time_step < first_appearance:
                temporal_subgraphs[time_step] = (None, None)
                continue

            full_graph = self.temporal_graphs[time_step]

            if not hasattr(full_graph, 'edge_index') or full_graph.edge_index.numel() == 0:
                temporal_subgraphs[time_step] = (None, None)
                continue

            G = self.build_networkx_graph_with_original_ids(full_graph)
            if G is None:
                temporal_subgraphs[time_step] = (None, None)
                continue

            if center_node_id not in G:
                G.add_node(center_node_id)

            if time_step == first_appearance:
                out_nodes = self._extract_k_hop_nodes(G, center_node_id, k, direction='out')
                in_nodes = self._extract_k_hop_nodes(G, center_node_id, k, direction='in')
                cumulative_out_nodes = out_nodes
                cumulative_in_nodes = in_nodes
            else:
                out_nodes = self._expand_k_hop_nodes(G, cumulative_out_nodes, k, direction='out')
                in_nodes = self._expand_k_hop_nodes(G, cumulative_in_nodes, k, direction='in')
                cumulative_out_nodes = out_nodes
                cumulative_in_nodes = in_nodes

            out_subgraph = self._create_subgraph_data_with_original_ids(G, full_graph, out_nodes)
            in_subgraph = self._create_subgraph_data_with_original_ids(G, full_graph, in_nodes)

            temporal_subgraphs[time_step] = (out_subgraph, in_subgraph)

        return temporal_subgraphs

    def _extract_k_hop_nodes(self, G: nx.DiGraph, center_node: int, k: int, direction: str) -> Set[int]:
        """提取k跳节点（使用原始节点ID）"""
        if center_node not in G:
            print(f"警告: 中心节点 {center_node} 不在图中")
            return {center_node}

        nodes = {center_node}
        current_nodes = {center_node}

        for _ in range(k):
            next_nodes = set()
            for node in current_nodes:
                neighbors = set(G.successors(node)) if direction == 'out' else set(G.predecessors(node))
                next_nodes.update(neighbors)

            next_nodes = next_nodes - nodes
            nodes.update(next_nodes)
            current_nodes = next_nodes

            if not next_nodes:
                break

        return nodes

    def _expand_k_hop_nodes(self, G: nx.DiGraph, existing_nodes: Set[int], k: int, direction: str) -> Set[int]:
        """基于现有节点集扩展k跳（使用原始节点ID）"""
        if not existing_nodes:
            return set()

        nodes = set(existing_nodes)
        current_nodes = set(existing_nodes)

        for _ in range(k):
            next_nodes = set()
            for node in current_nodes:
                if node in G:
                    neighbors = set(G.successors(node)) if direction == 'out' else set(G.predecessors(node))
                    next_nodes.update(neighbors)

            next_nodes = next_nodes - nodes
            nodes.update(next_nodes)
            current_nodes = next_nodes

            if not next_nodes:
                break

        return nodes

    def _create_subgraph_data_with_original_ids(self, G: nx.DiGraph, full_graph: Data, subgraph_nodes: Set[int]) -> Optional[Data]:
        """创建子图数据（使用原始节点ID）"""
        if len(subgraph_nodes) == 0:
            return None

        original_to_idx = self.get_original_to_idx_mapping(full_graph)
        if not original_to_idx:
            return None

        sorted_nodes = sorted(subgraph_nodes)
        node_mapping = {old_id: new_idx for new_idx, old_id in enumerate(sorted_nodes)}

        subgraph_edges = []
        for u, v in G.edges():
            if u in subgraph_nodes and v in subgraph_nodes:
                subgraph_edges.append([node_mapping[u], node_mapping[v]])

        node_features = []
        node_labels = []
        original_node_ids = []

        feature_dim = full_graph.x.shape[1] if hasattr(full_graph, 'x') and full_graph.x.numel() > 0 else 15

        for original_id in sorted_nodes:
            try:
                if original_id in original_to_idx:
                    pyg_idx = original_to_idx[original_id]

                    if pyg_idx < len(full_graph.x):
                        feature = full_graph.x[pyg_idx].cpu().numpy()
                        node_features.append(feature)
                    else:
                        node_features.append(np.zeros(feature_dim))

                    if hasattr(full_graph, 'y') and pyg_idx < len(full_graph.y):
                        label = full_graph.y[pyg_idx]
                        if isinstance(label, torch.Tensor):
                            if label.dim() == 0:
                                node_labels.append(int(label.item()))
                            else:
                                node_labels.append(int(label[0].item()))
                        elif isinstance(label, (int, float, np.integer, np.floating)):
                            node_labels.append(int(label))
                        else:
                            node_labels.append(0)
                    else:
                        node_labels.append(0)
                else:
                    node_features.append(np.zeros(feature_dim))
                    node_labels.append(0)

                original_node_ids.append(original_id)

            except Exception as e:
                print(f"处理节点 {original_id} 时出错: {e}")
                node_features.append(np.zeros(feature_dim))
                node_labels.append(0)
                original_node_ids.append(original_id)

        try:
            edge_index = (
                torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()
                if subgraph_edges else torch.empty((2, 0), dtype=torch.long)
            )
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.long)

            subgraph_data = Data(x=x, edge_index=edge_index, y=y)
            subgraph_data.original_node_ids = torch.tensor(original_node_ids, dtype=torch.long)

            return subgraph_data

        except Exception as e:
            print(f"创建子图数据时出错: {e}")
            return None

    def save_temporal_subgraphs(
        self,
        temporal_subgraphs: Dict[int, Tuple[Optional[Data], Optional[Data]]],
        center_node_identifier,
        k: int,
        output_dir: str
    ):
        """保存时序子图（包括文本数据）"""
        os.makedirs(output_dir, exist_ok=True)

        node_id = self.find_user_id(center_node_identifier)
        if node_id is None:
            node_id = str(center_node_identifier).replace('/', '_').replace('\\', '_')

        save_path = os.path.join(output_dir, f'subgraph_{node_id}_k{k}.pkl')

        node_texts = {}
        for time_step in self.time_steps:
            text_key = f"{time_step}_{node_id}"
            if text_key in self.user_texts:
                node_texts[time_step] = self.user_texts[text_key]

        save_data = {
            'center_node': center_node_identifier,
            'center_node_id': node_id,
            'k': k,
            'temporal_subgraphs': temporal_subgraphs,
            'time_steps': self.time_steps,
            'node_texts': node_texts
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        return save_path

    def batch_extract_normal_only(
        self,
        malicious_users_path: str,
        k: int = 2,
        output_dir: str = 'new_subgraphs',
        normal_count: int = 42000
    ):
        """
        只提取正常用户子图：
          - 从文件加载恶意用户列表
          - 正常用户 = all_users - malicious_users
          - 固定采样 normal_count（策略A：不足则全取）
          - 存储方式不变（subgraph_{node_id}_k{k}.pkl）
        """
        print("=== 开始批量子图提取（仅正常用户） ===")

        malicious_users = self.load_malicious_users_from_file(malicious_users_path)

        normal_users = self.sample_normal_users_fixed(malicious_users, num_normal=normal_count)
        all_target_users = list(normal_users)
        random.shuffle(all_target_users)

        print(f"总共要处理的用户: {len(all_target_users)} (正常: {len(normal_users)}，目标采样={normal_count})")

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        failed_count = 0

        for user in tqdm(all_target_users, desc="提取子图中"):
            try:
                temporal_subgraphs = self.extract_temporal_k_hop_subgraph(user, k)

                if temporal_subgraphs:
                    self.save_temporal_subgraphs(temporal_subgraphs, user, k, output_dir)
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"处理用户 {user} 时出错: {e}")
                failed_count += 1

        stats_path = os.path.join(output_dir, 'extraction_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'total_users': len(all_target_users),
                'malicious_users': len(malicious_users),   # 仅用于统计记录
                'normal_users': len(normal_users),
                'success_count': success_count,
                'failed_count': failed_count,
                'k': k,
                'normal_count_target': normal_count,
                'mode': 'normal_only'
            }, f)

        print(f"\n=== 批量提取完成（仅正常用户） ===")
        print(f"成功: {success_count}, 失败: {failed_count}")
        print(f"子图文件保存至: {output_dir}")
        print(f"统计信息保存至: {stats_path}")

        return success_count, failed_count


def main():
    temporal_graphs_path = 'whole_temporal_graphs.pkl'
    malicious_users_path = 'cnt-hongma.txt'
    k = 2

    # ========= 你的新需求 =========
    output_dir = 'new_subgraphs'   # 新文件夹
    normal_count = 42000           # 只提取 42000 个正常用户（不足则全取）
    # =============================

    random.seed(24)
    np.random.seed(24)

    extractor = TemporalKHopSubgraphExtractor(temporal_graphs_path)

    success_count, failed_count = extractor.batch_extract_normal_only(
        malicious_users_path=malicious_users_path,
        k=k,
        output_dir=output_dir,
        normal_count=normal_count
    )

    print(f"\n批量处理结果: 成功 {success_count}, 失败 {failed_count}")


if __name__ == "__main__":
    main()
