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
import ast


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

        if 'user_id_to_screen_names' in data:
            
            self.user_id_to_screen_names = defaultdict(set, data.get('user_id_to_screen_names', {}))
            self.screen_name_to_user_id = data.get('screen_name_to_user_id', {})
            self.user_id_to_internal_id = data.get('user_id_to_internal_id', {})
            self.internal_id_to_user_id = data.get('internal_id_to_user_id', {})
            
            # 创建 user_to_id 映射（screen_name -> internal_id）
            self.user_to_id = {}
            for screen_name, user_id_str in self.screen_name_to_user_id.items():
                if user_id_str in self.user_id_to_internal_id:
                    self.user_to_id[screen_name] = self.user_id_to_internal_id[user_id_str]
            
            # 创建 id_to_user 映射（internal_id -> screen_name）
            self.id_to_user = {}
            for internal_id, user_id_str in self.internal_id_to_user_id.items():
                screen_names = self.user_id_to_screen_names.get(user_id_str, set())
                if screen_names:
                    self.id_to_user[internal_id] = list(screen_names)[0]
        elif 'user_to_primary_id' in data:
            self.user_to_id = data['user_to_primary_id']
            self.id_to_user = data.get('id_to_user', {})
            self.user_id_to_screen_names = defaultdict(set)
            self.screen_name_to_user_id = {}
            self.user_id_to_internal_id = {}
            self.internal_id_to_user_id = {}
        else:
            self.user_to_id = data.get('user_to_id', {})
            self.id_to_user = data.get('id_to_user', {})
            self.user_id_to_screen_names = defaultdict(set)
            self.screen_name_to_user_id = {}
            self.user_id_to_internal_id = {}
            self.internal_id_to_user_id = {}

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
        print(f"恶意用户标签: {len(self.malicious_users)} 个")

    def load_malicious_users_from_file(self, malicious_users_path: str) -> Set[str]:
        """从文件加载恶意用户列表"""
        print(f"正在从 {malicious_users_path} 加载恶意用户列表...")

        with open(malicious_users_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if content.startswith('cnt'):
            content = content[3:].strip()

        set_start = content.find('{')
        set_end = content.rfind('}') + 1

        if set_start != -1 and set_end > set_start:
            set_content = content[set_start:set_end]
            try:
                malicious_set = ast.literal_eval(set_content)
            except (ValueError, SyntaxError):
                # 如果解析失败，使用正则表达式提取用户名
                import re
                pattern = r"'([^']+)'"
                matches = re.findall(pattern, set_content)
                malicious_set = set(matches)
        else:
            malicious_set = set()

        print(f"已加载 {len(malicious_set)} 个恶意用户")
        return malicious_set

    def is_malicious_user(self, screen_name: str) -> bool:
        """判断用户是否为恶意用户"""
        return screen_name in self.malicious_users

    def sample_normal_users(self, malicious_users: Set[str], ratio: float = 1.0) -> Set[str]:
        """采样正常用户"""
        # 获取所有用户
        all_users = set(self.user_to_id.keys())

        # 获取正常用户
        normal_users = all_users - malicious_users

        # 计算需要采样的正常用户数量
        num_malicious = len(malicious_users & all_users)  
        num_normal_to_sample = int(num_malicious * ratio)
        num_normal_to_sample = min(num_normal_to_sample, len(normal_users))

        # 随机采样
        sampled_normal_users = set(random.sample(list(normal_users), num_normal_to_sample))

        print(f"从 {len(normal_users)} 个正常用户中采样了 {len(sampled_normal_users)} 个")
        return sampled_normal_users

    def find_user_id(self, user_identifier):
        """查找用户的内部ID"""
        if isinstance(user_identifier, str):
            # 如果是screen_name，通过映射获取internal_id
            return self.user_to_id.get(user_identifier)
        elif isinstance(user_identifier, (int, np.integer)):
            # 如果已经是internal_id，直接返回
            return int(user_identifier)
        return None

    def check_node_existence(self, node_id: int) -> Dict[int, bool]:
        """检查节点在每个时间步的存在性"""
        existence = {}

        for time_step in self.time_steps:
            full_graph = self.temporal_graphs[time_step]
            exists = False

            try:
                # 检查 original_node_ids
                if hasattr(full_graph, 'original_node_ids') and full_graph.original_node_ids.numel() > 0:
                    original_ids = full_graph.original_node_ids.cpu().numpy()
                    exists = node_id in original_ids
                else:
                    # 如果没有 original_node_ids，则是空图
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
            # 检查图是否为空
            if not hasattr(full_graph, 'edge_index') or full_graph.edge_index.numel() == 0:
                return None
                
            # 获取原始节点ID到PyG索引的映射
            original_to_idx = self.get_original_to_idx_mapping(full_graph)
            if not original_to_idx:
                return None
                
            # 获取反向映射
            idx_to_original = {v: k for k, v in original_to_idx.items()}
            
            # 构建NetworkX图（使用原始节点ID）
            edge_index = full_graph.edge_index.cpu().numpy()
            G = nx.DiGraph()
            
            # 添加所有原始节点ID
            for original_id in original_to_idx.keys():
                G.add_node(original_id)
                
            # 添加边（使用原始节点ID）
            for i in range(edge_index.shape[1]):
                source_idx, target_idx = edge_index[:, i]
                
                # 将PyG索引转换为原始节点ID
                if source_idx in idx_to_original and target_idx in idx_to_original:
                    source_id = idx_to_original[source_idx]
                    target_id = idx_to_original[target_idx]
                    
                    # 确保两个节点都在图中
                    if source_id in G and target_id in G:
                        G.add_edge(source_id, target_id)
                    else:
                        # 如果节点不在图中，先添加节点再添加边
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
        """提取时序累积k跳子图"""
        # 获取中心节点ID
        center_node_id = self.find_user_id(center_node_identifier)
        
        if center_node_id is None:
            print(f"错误: 无法找到节点 {center_node_identifier}")
            return {}

        # 检查节点存在性，找到首次出现的时间步
        existence = self.check_node_existence(center_node_id)
        existing_steps = [step for step, exists in existence.items() if exists]
        
        if not existing_steps:
            print(f"节点 {center_node_identifier} 在任何时间步都不存在!")
            return {}

        first_appearance = min(existing_steps)
        print(f"节点 {center_node_identifier} 首次出现在时间步 {first_appearance}")

        temporal_subgraphs = {}
        
        # 累积子图节点集合（用于跨时间步扩展） - 使用原始节点ID
        cumulative_out_nodes = set()
        cumulative_in_nodes = set()
        
        for time_step in self.time_steps:
            # 首次出现之前，子图为空
            if time_step < first_appearance:
                temporal_subgraphs[time_step] = (None, None)
                continue
                
            full_graph = self.temporal_graphs[time_step]
            
            # 检查图是否为空
            if not hasattr(full_graph, 'edge_index') or full_graph.edge_index.numel() == 0:
                temporal_subgraphs[time_step] = (None, None)
                continue

            # 使用原始节点ID构建NetworkX图
            G = self.build_networkx_graph_with_original_ids(full_graph)
            if G is None:
                temporal_subgraphs[time_step] = (None, None)
                continue
                
            # 确保中心节点在图中
            if center_node_id not in G:
                G.add_node(center_node_id)

            # 提取累积扩展子图
            if time_step == first_appearance:
                # 首次出现：直接提取k跳子图
                out_nodes = self._extract_k_hop_nodes(G, center_node_id, k, direction='out')
                in_nodes = self._extract_k_hop_nodes(G, center_node_id, k, direction='in')
                
                cumulative_out_nodes = out_nodes
                cumulative_in_nodes = in_nodes
            else:
                # 后续时间步：基于前一时间步扩展k跳
                out_nodes = self._expand_k_hop_nodes(G, cumulative_out_nodes, k, direction='out')
                in_nodes = self._expand_k_hop_nodes(G, cumulative_in_nodes, k, direction='in')
                
                cumulative_out_nodes = out_nodes
                cumulative_in_nodes = in_nodes

            # 创建子图数据
            out_subgraph = self._create_subgraph_data_with_original_ids(G, full_graph, out_nodes)
            in_subgraph = self._create_subgraph_data_with_original_ids(G, full_graph, in_nodes)

            temporal_subgraphs[time_step] = (out_subgraph, in_subgraph)
            
            # 计算边数量
            out_edge_count = out_subgraph.edge_index.shape[1] if out_subgraph is not None else 0
            in_edge_count = in_subgraph.edge_index.shape[1] if in_subgraph is not None else 0
            
            print(f"时间步 {time_step}: 出度子图 {len(out_nodes) if out_nodes else 0} 个节点, {out_edge_count} 条边; 入度子图 {len(in_nodes) if in_nodes else 0} 个节点, {in_edge_count} 条边")

        return temporal_subgraphs

    def _extract_k_hop_nodes(self, G: nx.DiGraph, center_node: int, k: int, direction: str) -> Set[int]:
        """提取k跳节点（使用原始节点ID）"""
        # 确保中心节点在图中
        if center_node not in G:
            print(f"警告: 中心节点 {center_node} 不在图中")
            return {center_node}  
            
        nodes = {center_node}
        current_nodes = {center_node}

        for hop in range(k):
            next_nodes = set()
            for node in current_nodes:
                if direction == 'out':
                    neighbors = set(G.successors(node))
                else:  # direction == 'in'
                    neighbors = set(G.predecessors(node))
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
            
        nodes = set(existing_nodes)  # 保留所有现有节点
        current_nodes = set(existing_nodes)  # 当前扩展边界节点

        for hop in range(k):
            next_nodes = set()
            for node in current_nodes:
                # 确保节点在图中
                if node in G:
                    if direction == 'out':
                        neighbors = set(G.successors(node))
                    else:  # direction == 'in'
                        neighbors = set(G.predecessors(node))
                    next_nodes.update(neighbors)

            next_nodes = next_nodes - nodes  # 只添加新节点
            nodes.update(next_nodes)
            current_nodes = next_nodes

            if not next_nodes:
                break

        return nodes

    def _create_subgraph_data_with_original_ids(self, G: nx.DiGraph, full_graph: Data, subgraph_nodes: Set[int]) -> Optional[Data]:
        """创建子图数据（使用原始节点ID）"""
        if len(subgraph_nodes) == 0:
            return None

        # 获取原始节点ID到PyG索引的映射
        original_to_idx = self.get_original_to_idx_mapping(full_graph)
        if not original_to_idx:
            return None

        # 节点映射
        sorted_nodes = sorted(subgraph_nodes)
        node_mapping = {old_id: new_idx for new_idx, old_id in enumerate(sorted_nodes)}

        # 提取边
        subgraph_edges = []
        for u, v in G.edges():
            if u in subgraph_nodes and v in subgraph_nodes:
                subgraph_edges.append([node_mapping[u], node_mapping[v]])

        # 提取节点特征和标签
        node_features = []
        node_labels = []
        original_node_ids = []

        # 动态特征维度
        feature_dim = full_graph.x.shape[1] if hasattr(full_graph, 'x') and full_graph.x.numel() > 0 else 15

        for original_id in sorted_nodes:
            try:
                # 通过原始节点ID获取PyG索引，然后访问特征和标签
                if original_id in original_to_idx:
                    pyg_idx = original_to_idx[original_id]
                    
                    # 提取特征
                    if pyg_idx < len(full_graph.x):
                        feature = full_graph.x[pyg_idx].cpu().numpy()
                        node_features.append(feature)
                    else:
                        node_features.append(np.zeros(feature_dim))

                    # 提取标签
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
                    # 默认值
                    node_features.append(np.zeros(feature_dim))
                    node_labels.append(0)

                # 保存原始节点ID
                original_node_ids.append(original_id)

            except Exception as e:
                print(f"处理节点 {original_id} 时出错: {e}")
                node_features.append(np.zeros(feature_dim))
                node_labels.append(0)
                original_node_ids.append(original_id)

        # 创建PyTorch Geometric数据
        try:
            edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous() if subgraph_edges else torch.empty((2, 0), dtype=torch.long)
            x = torch.tensor(np.array(node_features), dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.long)

            subgraph_data = Data(x=x, edge_index=edge_index, y=y)
            subgraph_data.original_node_ids = torch.tensor(original_node_ids, dtype=torch.long)

            return subgraph_data

        except Exception as e:
            print(f"创建子图数据时出错: {e}")
            return None

    def save_temporal_subgraphs(self, temporal_subgraphs: Dict[int, Tuple[Optional[Data], Optional[Data]]],
                                center_node_identifier, k: int, output_dir: str,
                                is_malicious: bool = False):
        """保存时序子图（包括文本数据）"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 使用节点ID作为文件名
        node_id = self.find_user_id(center_node_identifier)
        if node_id is None:
            node_id = str(center_node_identifier).replace('/', '_').replace('\\', '_')

        save_path = os.path.join(output_dir, f'subgraph_{node_id}_k{k}.pkl')

        # 提取该节点的文本数据
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
            'node_texts': node_texts,
            'is_malicious': is_malicious  
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        return save_path

    
    def batch_extract_normal_subgraphs(self, num_normal: int = 6000, k: int = 2,
                                       output_dir: str = 'normal_subgraphs',
                                       malicious_users_path: Optional[str] = None):
        """仅针对正常用户批量提取子图"""
        print("=== 开始批量提取正常用户子图 ===")

        # 所有用户（screen_name）
        all_users = set(self.user_to_id.keys())

        # 合并来自文件和图数据本身的恶意用户
        malicious_users_from_file = set()
        if malicious_users_path is not None:
            malicious_users_from_file = self.load_malicious_users_from_file(malicious_users_path)

        malicious_users = (self.malicious_users | malicious_users_from_file) & all_users

        # 正常用户候选
        normal_users = list(all_users - malicious_users)
        if not normal_users:
            print("警告: 没有正常用户可供采样")
            return 0, 0

        num_normal_to_sample = min(num_normal, len(normal_users))
        sampled_normal_users = set(random.sample(normal_users, num_normal_to_sample))

        print(f"总用户数: {len(all_users)}, 恶意用户数: {len(malicious_users)}, 正常用户候选: {len(normal_users)}")
        print(f"实际采样正常用户: {len(sampled_normal_users)} 个 (目标 {num_normal} 个)")

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        failed_count = 0

        for user in tqdm(sampled_normal_users, desc="提取正常用户子图中"):
            try:
                # 提取时序子图
                temporal_subgraphs = self.extract_temporal_k_hop_subgraph(user, k)

                if temporal_subgraphs:
                    # 正常用户 
                    self.save_temporal_subgraphs(temporal_subgraphs, user, k, output_dir, is_malicious=False)
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"处理正常用户 {user} 时出错: {e}")
                failed_count += 1

        # 保存统计信息
        stats_path = os.path.join(output_dir, 'normal_extraction_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'total_normal_target': num_normal,
                'sampled_normal_users': len(sampled_normal_users),
                'success_count': success_count,
                'failed_count': failed_count,
                'k': k
            }, f)

        print(f"\n=== 正常用户子图批量提取完成 ===")
        print(f"成功: {success_count}, 失败: {failed_count}")
        print(f"正常用户子图文件保存至: {output_dir}")
        print(f"统计信息保存至: {stats_path}")

        return success_count, failed_count

def main():
    
    temporal_graphs_path = 'twitter_temporal_graphs.pkl'
    malicious_users_path = 'tweets_users.txt'
    k = 2

    normal_output_dir = 'twitter_subgraphs_normal_105000'
    num_normal_users = 105000

    random.seed(24)
    np.random.seed(24)

    # 创建提取器
    extractor = TemporalKHopSubgraphExtractor(temporal_graphs_path)

    # 单独提取固定数量的正常用户子图
    normal_success_count, normal_failed_count = extractor.batch_extract_normal_subgraphs(
        num_normal=num_normal_users,
        k=k,
        output_dir=normal_output_dir,
        malicious_users_path=malicious_users_path
    )

    print(f"\n正常用户批量处理结果: 成功 {normal_success_count}, 失败 {normal_failed_count}")



if __name__ == "__main__":
    main()