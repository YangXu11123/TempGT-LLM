import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from collections import defaultdict
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple
import warnings
import re

warnings.filterwarnings('ignore')


class ImprovedTemporalGraphBuilder:
    def __init__(self, csv_path: str, malicious_users_path: str, time_step_hours: int = 6, node_embed_dim: int = 10):
        """
        初始化时序图构建器

        参数：
            csv_path: CSV数据文件路径
            malicious_users_path: 恶意用户标签文件路径
            time_step_hours: 时间步长度，本数据集为6小时
            node_embed_dim: 节点ID嵌入维度，默认10维
        """
        self.csv_path = csv_path
        self.malicious_users_path = malicious_users_path
        self.time_step_hours = time_step_hours
        self.node_embed_dim = node_embed_dim
        self.df = None
        self.malicious_users = set()
        self.user_to_primary_id = {}
        self.user_to_all_ids = {}
        self.id_to_user = {}
        self.user_to_id = {}
        self.user_texts = defaultdict(list)
        self.node_id_embeddings = {}
        self.load_data()

    def load_data(self):
        """加载数据并进行预处理"""
        print(f"从 {self.csv_path} 加载数据...")
        self.df = pd.read_csv(self.csv_path)
        print(f"加载了 {len(self.df)} 条记录")

        if 'publish_time' in self.df.columns:
            self.df['publish_time'] = pd.to_datetime(self.df['publish_time'])

        self.load_malicious_users()
        self._create_improved_user_mapping()

        print(f"总用户数: {len(self.user_to_primary_id)}")
        print(f"总ID数: {len(self.id_to_user)}")

    def load_malicious_users(self):
        """从文件加载恶意用户标签"""
        print(f"从 {self.malicious_users_path} 加载恶意用户标签...")

        try:
            with open(self.malicious_users_path, 'r', encoding='gbk') as f:
                content = f.read()

            user_pattern = r"'([^']+)'"
            matches = re.findall(user_pattern, content)

            self.malicious_users = set(matches)
            print(f"加载了 {len(self.malicious_users)} 个恶意用户标签")

        except Exception as e:
            print(f"加载恶意用户标签时出错: {e}")
            self.malicious_users = set()

    def _create_improved_user_mapping(self):
        """创建改进的用户 ID 映射"""
        print("创建改进的用户ID映射...")

        user_id_pairs = []
        for _, row in self.df.iterrows():
            for user_col, id_col in [('source', 'source_id'), ('target', 'target_id'), ('root', 'root_id')]:
                if pd.notna(row.get(user_col)) and pd.notna(row.get(id_col)):
                    try:
                        user_id = int(float(row[id_col]))
                        user_id_pairs.append((str(row[user_col]).strip(), user_id))
                    except (ValueError, TypeError):
                        pass

        user_ids_count = defaultdict(lambda: defaultdict(int))
        for user, user_id in user_id_pairs:
            user_ids_count[user][user_id] += 1

        used_ids = set()
        for _, id_counts in user_ids_count.items():
            for id_val in id_counts.keys():
                if id_val != 0:
                    used_ids.add(id_val)

        next_new_id = 1
        while next_new_id in used_ids:
            next_new_id += 1

        for user, id_counts in user_ids_count.items():
            sorted_ids = sorted(id_counts.items(), key=lambda x: x[1], reverse=True)

            primary_id = None
            all_ids = [id_val for id_val, _ in sorted_ids]

            for id_val, _ in sorted_ids:
                if id_val != 0:
                    primary_id = id_val
                    break

            if primary_id is None:
                while next_new_id in used_ids or next_new_id in self.id_to_user:
                    next_new_id += 1
                primary_id = next_new_id
                used_ids.add(primary_id)
                all_ids = [primary_id] + [id_val for id_val in all_ids if id_val != primary_id]
                next_new_id += 1

            self.user_to_primary_id[user] = primary_id
            self.user_to_all_ids[user] = all_ids
            self.id_to_user[primary_id] = user
            self.user_to_id[user] = primary_id

        all_users_in_data = set()
        for col in ['source', 'target', 'root']:
            if col in self.df.columns:
                all_users_in_data.update(self.df[col].dropna().astype(str).str.strip())

        users_without_ids = all_users_in_data - set(self.user_to_primary_id.keys())
        for user in users_without_ids:
            while next_new_id in used_ids or next_new_id in self.id_to_user:
                next_new_id += 1

            self.user_to_primary_id[user] = next_new_id
            self.user_to_all_ids[user] = [next_new_id]
            self.id_to_user[next_new_id] = user
            self.user_to_id[user] = next_new_id
            used_ids.add(next_new_id)
            next_new_id += 1

        print(f"创建了 {len(self.user_to_primary_id)} 个用户映射")
        print(f"ID范围: {min(self.id_to_user.keys())} 到 {max(self.id_to_user.keys())}")

    def _get_node_id_embedding(self, node_id: int) -> np.ndarray:
        if node_id not in self.node_id_embeddings:
            seed = int(node_id) % (2**32)
            rng = np.random.RandomState(seed=seed)
            embedding = rng.normal(0, 0.1, self.node_embed_dim)
            self.node_id_embeddings[node_id] = embedding
        return self.node_id_embeddings[node_id]

    def create_time_windows(self) -> List[Tuple[datetime, datetime]]:
        """
        创建时间窗口序列

        返回：
            时间窗口列表，每个窗口为(开始时间, 结束时间)
        """
        start_time = self.df['publish_time'].min()
        end_time = self.df['publish_time'].max()

        total_minutes = (end_time - start_time).total_seconds() / 60.0
        window_minutes = self.time_step_hours * 60

        print("开始构建时序快照图...")
        print(
            f"数据时间跨度: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 "
            f"{end_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(总计 {total_minutes:.1f} 分钟)"
        )

        time_windows = []
        current_time = start_time

        while current_time < end_time:
            window_end = current_time + timedelta(hours=self.time_step_hours)
            time_windows.append((current_time, window_end))
            current_time = window_end

        print(f"创建了 {len(time_windows)} 个时间窗口，每个窗口 {window_minutes} 分钟")
        return time_windows

    def extract_behavioral_features(self, user_interactions: Dict, time_window: Tuple[datetime, datetime]) -> np.ndarray:
        """
        提取用户行为特征

        参数：
            user_interactions: 用户交互数据
            time_window: 时间窗口

        返回：
            5 维行为特征向量：
            [发帖频率, 出度, 入度, 响应延迟, 发帖间隔方差]
        """
        interactions = user_interactions['interactions']

        if not interactions:
            return np.zeros(5)

        window_hours = (time_window[1] - time_window[0]).total_seconds() / 3600
        post_frequency = len(interactions) / window_hours

        out_degree = sum(1 for interaction in interactions if interaction['type'] == 'source')
        in_degree = sum(1 for interaction in interactions if interaction['type'] == 'target')

        reply_delays = []
        for interaction in interactions:
            if interaction['type'] == 'source' and 'target' in interaction:
                target_id = interaction['target']
                target_msgs = [
                    i for i in interactions
                    if i['type'] == 'target'
                    and i.get('source') == target_id
                    and i['time'] < interaction['time']
                ]
                if target_msgs:
                    last_msg_time = max(msg['time'] for msg in target_msgs)
                    delay = (interaction['time'] - last_msg_time).total_seconds() / 3600
                    reply_delays.append(delay)

        avg_delay = np.mean(reply_delays) if reply_delays else 0

        if len(interactions) > 1:
            sorted_interactions = sorted(interactions, key=lambda x: x['time'])
            time_intervals = []
            for i in range(1, len(sorted_interactions)):
                interval = (sorted_interactions[i]['time'] - sorted_interactions[i - 1]['time']).total_seconds() / 3600
                time_intervals.append(interval)

            if time_intervals:
                mean_interval = np.mean(time_intervals)
                std_interval = np.std(time_intervals)
                activity_variance = std_interval / mean_interval if mean_interval > 0 else 0
            else:
                activity_variance = 0
        else:
            activity_variance = 0

        features = np.array([post_frequency, out_degree, in_degree, avg_delay, activity_variance])

        if np.any(features > 0):
            max_vals = np.array([10, 100, 100, 48, 5])
            features = np.clip(features, 0, max_vals)
            features = features / max_vals
        else:
            features = np.zeros(5)

        return features

    def build_temporal_snapshots(self):
        """
        构建时序快照图

        返回：
            时序图列表，每个元素为一个时间步的 PyTorch Geometric 图数据
        """
        time_windows = self.create_time_windows()
        temporal_graphs = []

        for t, (start_time, end_time) in enumerate(tqdm(time_windows, desc="构建时序")):
            window_data = self.df[
                (self.df['publish_time'] >= start_time) &
                (self.df['publish_time'] < end_time)
            ].copy()

            if len(window_data) == 0:
                empty_data = Data(
                    x=torch.zeros((0, 15), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 2), dtype=torch.float),
                    y=torch.zeros((0,), dtype=torch.long),
                    time_step=t,
                    num_nodes=0
                )
                temporal_graphs.append(empty_data)
                continue

            G = nx.DiGraph()
            user_interactions = defaultdict(lambda: {'user_id': None, 'interactions': []})

            for _, row in window_data.iterrows():
                source_user = str(row['source']).strip() if pd.notna(row['source']) else None
                target_user = str(row['target']).strip() if pd.notna(row['target']) else None
                root_user = str(row['root']).strip() if pd.notna(row['root']) else None
                publish_time = row['publish_time']

                for user in [source_user, target_user, root_user]:
                    if user and user in self.user_to_primary_id:
                        user_id = self.user_to_primary_id[user]
                        if not G.has_node(user_id):
                            G.add_node(
                                user_id,
                                user_name=user,
                                is_malicious=1 if user in self.malicious_users else 0
                            )

                        if user_interactions[user_id]['user_id'] is None:
                            user_interactions[user_id]['user_id'] = user_id

                if source_user and target_user:
                    source_id = self.user_to_primary_id.get(source_user)
                    target_id = self.user_to_primary_id.get(target_user)

                    if source_id is not None and target_id is not None:
                        timestamp_float = (
                            publish_time.timestamp()
                            if hasattr(publish_time, 'timestamp')
                            else float(publish_time.value)
                        )

                        G.add_edge(
                            source_id,
                            target_id,
                            timestamp=timestamp_float,
                            cascade_depth=float(row.get('cascade_depth', 1))
                        )

                        user_interactions[source_id]['interactions'].append({
                            'type': 'source',
                            'time': publish_time,
                            'target': target_id
                        })
                        user_interactions[target_id]['interactions'].append({
                            'type': 'target',
                            'time': publish_time,
                            'source': source_id
                        })

                for text_col in ['original_text', 'retweet_text']:
                    if pd.notna(row.get(text_col)) and source_user:
                        user_id = self.user_to_primary_id.get(source_user)
                        if user_id is not None:
                            text_content = str(row[text_col]).strip()
                            if text_content:
                                self.user_texts[f"{t}_{user_id}"].append({
                                    'text': text_content,
                                    'timestamp': publish_time,
                                    'type': text_col
                                })

            for user_id in user_interactions.keys():
                text_key = f"{t}_{user_id}"
                if text_key in self.user_texts:
                    texts = self.user_texts[text_key]
                    if len(texts) > 100:
                        sorted_texts = sorted(texts, key=lambda x: x['timestamp'], reverse=True)
                        self.user_texts[text_key] = sorted_texts[:100]

            if len(G.nodes()) == 0:
                empty_data = Data(
                    x=torch.zeros((0, 15), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 2), dtype=torch.float),
                    y=torch.zeros((0,), dtype=torch.long),
                    time_step=t,
                    num_nodes=0
                )
                temporal_graphs.append(empty_data)
                continue

            node_features = []
            node_labels = []

            for node_id in G.nodes():
                behavioral_features = self.extract_behavioral_features(
                    user_interactions[node_id], (start_time, end_time)
                )

                node_id_embedding = self._get_node_id_embedding(node_id)
                full_features = np.concatenate([behavioral_features, node_id_embedding])
                node_features.append(full_features)

                user_name = G.nodes[node_id].get('user_name', '未知用户')
                is_malicious = 1 if user_name in self.malicious_users else 0
                node_labels.append(is_malicious)

            if len(node_features) > 0:
                node_list = list(G.nodes())
                node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}

                G_mapped = nx.DiGraph()
                for old_id in node_list:
                    new_id = node_mapping[old_id]
                    G_mapped.add_node(new_id, **G.nodes[old_id])

                for source, target, data in G.edges(data=True):
                    G_mapped.add_edge(node_mapping[source], node_mapping[target], **data)

                pyg_data = from_networkx(G_mapped)
                pyg_data.x = torch.tensor(node_features, dtype=torch.float)
                pyg_data.y = torch.tensor(node_labels, dtype=torch.long)
                pyg_data.time_step = t
                pyg_data.original_node_ids = torch.tensor(node_list, dtype=torch.long)

                pyg_data.node_mapping = node_mapping
                pyg_data.reverse_node_mapping = {v: k for k, v in node_mapping.items()}

                temporal_graphs.append(pyg_data)

            else:
                empty_data = Data(
                    x=torch.zeros((0, 15), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 2), dtype=torch.float),
                    y=torch.zeros((0,), dtype=torch.long),
                    time_step=t,
                    num_nodes=0
                )
                temporal_graphs.append(empty_data)

        return temporal_graphs

    def save_temporal_graphs(self, save_path: str):
        """
        保存时序图数据到文件

        参数：
            save_path: 保存路径
        """
        print("开始构建时序图...")

        temporal_graphs = self.build_temporal_snapshots()

        print(f"保存时序图到 {save_path}...")

        save_data = {
            'temporal_graphs': temporal_graphs,
            'user_to_primary_id': self.user_to_primary_id,
            'user_to_all_ids': self.user_to_all_ids,
            'id_to_user': self.id_to_user,
            'user_to_id': self.user_to_id,
            'malicious_users': self.malicious_users,
            'user_texts': dict(self.user_texts),
            'time_step_hours': self.time_step_hours,
            'node_id_embeddings': self.node_id_embeddings
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"成功保存 {len(temporal_graphs)} 个时间步的图数据")

    def load_temporal_graphs(self, load_path: str):
        """
        从文件加载时序图数据

        参数：
            load_path: 文件路径

        返回：
            时序图列表
        """
        print(f"从 {load_path} 加载时序图数据...")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        temporal_graphs = data['temporal_graphs']
        self.user_to_primary_id = data.get('user_to_primary_id', {})
        self.user_to_all_ids = data.get('user_to_all_ids', {})
        self.id_to_user = data.get('id_to_user', {})
        self.user_to_id = data.get('user_to_id', self.user_to_primary_id)
        self.malicious_users = data.get('malicious_users', set())
        self.user_texts = defaultdict(list, data.get('user_texts', {}))
        self.time_step_hours = data.get('time_step_hours', 6)
        self.node_id_embeddings = data.get('node_id_embeddings', {})

        print(f"加载了 {len(temporal_graphs)} 个时间步的图数据")
        return temporal_graphs

    def get_statistics(self):
        malicious_user_ids = sum(
            1 for user in self.malicious_users if user in self.user_to_primary_id
        )
        multi_name_users = sum(
            1 for _, ids in self.user_to_all_ids.items() if len(set(ids)) > 1
        )

        return {
            'total_user_ids': len(self.user_to_primary_id),
            'total_tweets': len(self.df) if self.df is not None else 0,
            'malicious_user_labels': len(self.malicious_users),
            'malicious_user_ids': malicious_user_ids,
            'time_range': (
                self.df['publish_time'].min(),
                self.df['publish_time'].max()
            ) if self.df is not None else None,
            'multi_name_users': multi_name_users
        }

    def verify_user_mapping(self, test_user_names: List[str]):
        """验证用户映射"""
        print("\n=== 用户映射验证 ===")
        for user in test_user_names:
            if user in self.user_to_primary_id:
                primary_id = self.user_to_primary_id[user]
                all_ids = self.user_to_all_ids.get(user, [primary_id])
                is_malicious = "恶意" if user in self.malicious_users else "正常"
                print(f"  {user} -> 主要ID: {primary_id}, 所有ID: {all_ids}, 类型: {is_malicious}")
            else:
                print(f"  {user} -> 未找到映射")


def main():
    """主函数"""
    csv_path = "hongma_cascade.csv"
    malicious_users_path = "cnt-hongma.txt"
    save_path = "whole_temporal_graphs.pkl"

    builder = ImprovedTemporalGraphBuilder(
        csv_path=csv_path,
        malicious_users_path=malicious_users_path,
        time_step_hours=6
    )

    
    #test_users = ["般若寺扫地僧", "CCTV焦点访谈", "河南那几家银行还我83万存款"]
    #builder.verify_user_mapping(test_users)

    builder.save_temporal_graphs(save_path)

    stats = builder.get_statistics()
    print("\n" + "=" * 68)
    print("=== Weibo数据统计 ===")
    print("=" * 68)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 68)
    print("\nWeibo时序图构建完成！")


if __name__ == "__main__":
    main()