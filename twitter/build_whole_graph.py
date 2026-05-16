import json
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
import ast

warnings.filterwarnings('ignore')

class ImprovedTemporalGraphBuilder:
    def __init__(self, json_path: str, malicious_users_path: str, time_step_minutes: int = 20, node_embed_dim: int = 10):
        """初始化时序图构建器"""
        self.json_path = json_path
        self.malicious_users_path = malicious_users_path
        self.time_step_minutes = time_step_minutes
        self.node_embed_dim = node_embed_dim

        # 数据存储
        self.tweets_data = []
        self.malicious_users = set()
        
        # 用户映射 - 以用户ID为主键
        self.user_id_to_screen_names = defaultdict(set)
        self.screen_name_to_user_id = {}
        self.user_id_to_internal_id = {}
        self.internal_id_to_user_id = {}
        
        # 推文ID映射，用于构建回复链
        self.tweet_id_to_data = {}
        
        # 文本存储 - key: "时间步_内部用户ID", value: 文本列表
        self.user_texts = defaultdict(list)
        
        # 确定性节点ID嵌入缓存
        self.node_id_embeddings = {}

        self.load_data()

    def load_data(self):
        """加载Twitter JSON数据并进行预处理"""
        print(f"从 {self.json_path} 加载Twitter数据...")
        
        line_count = 0
        with open(self.json_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    tweet = json.loads(line.strip())
                    created_at = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    tweet['parsed_created_at'] = created_at
                    self.tweets_data.append(tweet)
                    
                    self.tweet_id_to_data[tweet['id_str']] = tweet
                    
                    line_count += 1
                    if line_count % 100000 == 0:
                        print(f"已加载 {line_count} 条推文...")
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    if line_num < 10:
                        print(f"解析第{line_num+1}行时出错: {e}")
                    continue
        
        print(f"成功加载 {len(self.tweets_data)} 条推文")

        self.load_malicious_users()
        self._create_user_mapping()
        
        print(f"总用户ID数: {len(self.user_id_to_internal_id)}")

    def load_malicious_users(self):
        """从文件加载恶意用户标签"""
        print(f"从 {self.malicious_users_path} 加载恶意用户标签...")

        try:
            with open(self.malicious_users_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content.startswith('cnt'):
                content = content[3:].strip()
            
            self.malicious_users = ast.literal_eval(content)
            print(f"加载了 {len(self.malicious_users)} 个恶意用户标签")

        except Exception as e:
            print(f"加载恶意用户标签时出错: {e}")
            self.malicious_users = set()

    def _create_user_mapping(self):
        """创建用户ID映射"""
        print("创建用户ID映射...")
        
        for tweet in self.tweets_data:
            user_info = tweet.get('user', {})
            user_id_str = user_info.get('id_str')
            screen_name = user_info.get('screen_name')
            
            if user_id_str and screen_name:
                self.user_id_to_screen_names[user_id_str].add(screen_name)
                self.screen_name_to_user_id[screen_name] = user_id_str
        
        internal_id_counter = 1
        for user_id_str in self.user_id_to_screen_names.keys():
            self.user_id_to_internal_id[user_id_str] = internal_id_counter
            self.internal_id_to_user_id[internal_id_counter] = user_id_str
            internal_id_counter += 1
        
        multi_name_count = sum(1 for names in self.user_id_to_screen_names.values() if len(names) > 1)
        print(f"创建了 {len(self.user_id_to_internal_id)} 个用户ID映射")
        print(f"有多个用户名的用户ID: {multi_name_count}")

    def _get_node_id_embedding(self, node_id: int) -> np.ndarray:
        """获取确定性的节点ID嵌入"""
        if node_id not in self.node_id_embeddings:
            seed = int(node_id) % (2**32)
            rng = np.random.RandomState(seed=seed)
            embedding = rng.normal(0, 0.1, self.node_embed_dim)
            self.node_id_embeddings[node_id] = embedding
        return self.node_id_embeddings[node_id]

    def create_time_windows(self) -> List[Tuple[datetime, datetime]]:
        """创建时间窗口序列"""
        all_times = [tweet['parsed_created_at'] for tweet in self.tweets_data]
        start_time = min(all_times)
        end_time = max(all_times)
        
        total_minutes = (end_time - start_time).total_seconds() / 60
        print(f"数据时间跨度: {start_time} 到 {end_time} (总计 {total_minutes:.1f} 分钟)")

        time_windows = []
        current_time = start_time

        while current_time < end_time:
            window_end = current_time + timedelta(minutes=self.time_step_minutes)
            time_windows.append((current_time, window_end))
            current_time = window_end

        print(f"创建了 {len(time_windows)} 个时间窗口，每个窗口 {self.time_step_minutes} 分钟")
        return time_windows

    def is_malicious_user(self, user_id_str: str) -> bool:
        """判断用户是否为恶意用户"""
        if user_id_str not in self.user_id_to_screen_names:
            return False
        
        user_screen_names = self.user_id_to_screen_names[user_id_str]
        return any(screen_name in self.malicious_users for screen_name in user_screen_names)

    def extract_behavioral_features(self, user_interactions: Dict, time_window: Tuple[datetime, datetime]) -> np.ndarray:
        """提取用户行为特征"""
        interactions = user_interactions.get('interactions', [])

        if not interactions:
            return np.zeros(5)

        # 1. 发帖频率（帖子数 / 时间窗口分钟数）
        window_minutes = (time_window[1] - time_window[0]).total_seconds() / 60
        post_frequency = len(interactions) / window_minutes if window_minutes > 0 else 0

        # 2. 出度（发起的互动数）
        out_degree = sum(1 for interaction in interactions 
                        if interaction['type'] in ['reply_to', 'retweet'])
        
        # 3. 入度（接收的互动数）
        in_degree = sum(1 for interaction in interactions 
                       if interaction['type'] in ['replied_by', 'retweeted_by'])

        # 4. 响应延迟（回复时间间隔的平均值）
        reply_delays = []
        for interaction in interactions:
            if interaction['type'] == 'reply_to' and 'original_time' in interaction and interaction['original_time']:
                delay = (interaction['time'] - interaction['original_time']).total_seconds() / 60
                if delay > 0:
                    reply_delays.append(delay)
        
        avg_delay = np.mean(reply_delays) if reply_delays else 0

        # 5. 发帖时间间隔的方差（使用变异系数）
        if len(interactions) > 1:
            sorted_interactions = sorted(interactions, key=lambda x: x['time'])
            time_intervals = []
            for i in range(1, len(sorted_interactions)):
                interval = (sorted_interactions[i]['time'] - sorted_interactions[i-1]['time']).total_seconds() / 60
                time_intervals.append(interval)
            
            if time_intervals:
                mean_interval = np.mean(time_intervals)
                std_interval = np.std(time_intervals)
                activity_variance = std_interval / mean_interval if mean_interval > 0 else 0
            else:
                activity_variance = 0
        else:
            activity_variance = 0

        # 特征归一化
        features = np.array([post_frequency, out_degree, in_degree, avg_delay, activity_variance])
        
        if np.any(features > 0):
            max_vals = np.array([1, 50, 50, 120, 5])  
            features = np.clip(features, 0, max_vals)
            features = features / max_vals
        else:
            features = np.zeros(5)

        return features

    def build_temporal_snapshots(self):
        """构建时序快照图"""
        print("开始构建时序快照图...")

        time_windows = self.create_time_windows()
        temporal_graphs = []

        for t, (start_time, end_time) in enumerate(tqdm(time_windows, desc="构建时序图")):
            # 过滤当前时间窗口的推文
            window_tweets = [
                tweet for tweet in self.tweets_data
                if start_time <= tweet['parsed_created_at'] < end_time
            ]

            # 如果当前窗口没有数据，创建空图
            if len(window_tweets) == 0:
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

            # 构建有向图 - 以内部用户ID为节点
            G = nx.DiGraph()
            user_interactions = defaultdict(lambda: {'user_id': None, 'interactions': []})

            # 处理每条推文
            for tweet in window_tweets:
                user_info = tweet.get('user', {})
                user_id_str = user_info.get('id_str')
                screen_name = user_info.get('screen_name')
                created_at = tweet['parsed_created_at']
                text = tweet.get('text', '')

                if not user_id_str or user_id_str not in self.user_id_to_internal_id:
                    continue

                internal_user_id = self.user_id_to_internal_id[user_id_str]

                # 添加节点
                if not G.has_node(internal_user_id):
                    is_malicious = 1 if self.is_malicious_user(user_id_str) else 0
                    all_screen_names = list(self.user_id_to_screen_names[user_id_str])
                    primary_screen_name = screen_name if screen_name else (all_screen_names[0] if all_screen_names else '')
                    
                    G.add_node(internal_user_id,
                             user_id_str=user_id_str,
                             user_name=primary_screen_name,
                             all_screen_names=all_screen_names,
                             is_malicious=is_malicious)

                if user_interactions[internal_user_id]['user_id'] is None:
                    user_interactions[internal_user_id]['user_id'] = internal_user_id

                # 记录推文行为
                user_interactions[internal_user_id]['interactions'].append({
                    'type': 'tweet',
                    'time': created_at
                })

                # 收集用户文本
                if text.strip():
                    self.user_texts[f"{t}_{internal_user_id}"].append({
                        'text': text.strip(),
                        'timestamp': created_at,
                        'type': 'tweet',
                        'screen_name': screen_name
                    })

                # 处理回复关系
                in_reply_to_status_id = tweet.get('in_reply_to_status_id_str')
                in_reply_to_user_id = tweet.get('in_reply_to_user_id_str')

                if in_reply_to_status_id and in_reply_to_user_id:
                    if in_reply_to_user_id in self.user_id_to_internal_id:
                        target_internal_id = self.user_id_to_internal_id[in_reply_to_user_id]
                        
                        # 添加被回复的用户节点
                        if not G.has_node(target_internal_id):
                            target_is_malicious = 1 if self.is_malicious_user(in_reply_to_user_id) else 0
                            target_screen_names = list(self.user_id_to_screen_names[in_reply_to_user_id])
                            target_primary_name = target_screen_names[0] if target_screen_names else ''
                            
                            G.add_node(target_internal_id,
                                     user_id_str=in_reply_to_user_id,
                                     user_name=target_primary_name,
                                     all_screen_names=target_screen_names,
                                     is_malicious=target_is_malicious)

                        timestamp_float = created_at.timestamp()
                        
                        # 添加边（回复者 -> 被回复者）
                        if G.has_edge(internal_user_id, target_internal_id):
                            G[internal_user_id][target_internal_id]['weight'] += 1.0
                        else:
                            G.add_edge(internal_user_id, target_internal_id,
                                     timestamp=timestamp_float,
                                     edge_type=1.0,  # 1表示回复
                                     weight=1.0)

                        # 获取原始推文时间（用于计算响应延迟）
                        original_time = None
                        if in_reply_to_status_id in self.tweet_id_to_data:
                            original_tweet = self.tweet_id_to_data[in_reply_to_status_id]
                            original_time = original_tweet.get('parsed_created_at')

                        # 记录交互信息
                        user_interactions[internal_user_id]['interactions'].append({
                            'type': 'reply_to',
                            'time': created_at,
                            'target': target_internal_id,
                            'original_time': original_time
                        })

                        user_interactions[target_internal_id]['interactions'].append({
                            'type': 'replied_by',
                            'time': created_at,
                            'source': internal_user_id,
                            'original_time': original_time
                        })

                # 处理转发关系
                retweeted_status = tweet.get('retweeted_status')
                if retweeted_status:
                    original_user = retweeted_status.get('user', {})
                    original_user_id_str = original_user.get('id_str')
                    
                    if original_user_id_str and original_user_id_str in self.user_id_to_internal_id:
                        original_internal_id = self.user_id_to_internal_id[original_user_id_str]
                        
                        # 添加原始推文作者节点
                        if not G.has_node(original_internal_id):
                            original_is_malicious = 1 if self.is_malicious_user(original_user_id_str) else 0
                            original_screen_names = list(self.user_id_to_screen_names[original_user_id_str])
                            original_primary_name = original_user.get('screen_name', '') or (original_screen_names[0] if original_screen_names else '')
                            
                            G.add_node(original_internal_id,
                                     user_id_str=original_user_id_str,
                                     user_name=original_primary_name,
                                     all_screen_names=original_screen_names,
                                     is_malicious=original_is_malicious)

                        timestamp_float = created_at.timestamp()
                        
                        # 添加边（转发者 -> 原作者）
                        if G.has_edge(internal_user_id, original_internal_id):
                            G[internal_user_id][original_internal_id]['weight'] += 1.0
                        else:
                            G.add_edge(internal_user_id, original_internal_id,
                                     timestamp=timestamp_float,
                                     edge_type=2.0,  # 2表示转发
                                     weight=1.0)

                        # 记录交互信息
                        user_interactions[internal_user_id]['interactions'].append({
                            'type': 'retweet',
                            'time': created_at,
                            'target': original_internal_id
                        })

                        user_interactions[original_internal_id]['interactions'].append({
                            'type': 'retweeted_by',
                            'time': created_at,
                            'source': internal_user_id
                        })

            # 限制每个节点每个时间步的文本数量为M=100
            for user_id in user_interactions.keys():
                text_key = f"{t}_{user_id}"
                if text_key in self.user_texts:
                    texts = self.user_texts[text_key]
                    if len(texts) > 100:
                        sorted_texts = sorted(texts, key=lambda x: x['timestamp'], reverse=True)
                        self.user_texts[text_key] = sorted_texts[:100]

            # 统计当前时间步的总文本数量
            total_text_count = 0
            for user_id in user_interactions.keys():
                text_key = f"{t}_{user_id}"
                if text_key in self.user_texts:
                    total_text_count += len(self.user_texts[text_key])

            # 如果图中没有节点，创建空图
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

            # 构建节点特征和标签
            node_features = []
            node_labels = []

            for node_id in G.nodes():
                # 提取行为特征（5维）
                behavioral_features = self.extract_behavioral_features(
                    user_interactions[node_id], (start_time, end_time)
                )

                # 获取确定性节点ID嵌入（10维）
                node_id_embedding = self._get_node_id_embedding(node_id)
                
                # 拼接特征：行为特征(5维) + 节点ID嵌入(10维) = 15维
                full_features = np.concatenate([behavioral_features, node_id_embedding])
                node_features.append(full_features)

                # 节点标签
                is_malicious = G.nodes[node_id].get('is_malicious', 0)
                node_labels.append(is_malicious)

            # 转换为PyTorch Geometric格式
            if len(node_features) > 0:
                # 重新映射节点ID（从0开始）
                node_list = list(G.nodes())
                node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}

                G_mapped = nx.DiGraph()
                for old_id in node_list:
                    new_id = node_mapping[old_id]
                    G_mapped.add_node(new_id, **G.nodes[old_id])

                for source, target, data in G.edges(data=True):
                    G_mapped.add_edge(node_mapping[source], node_mapping[target], **data)

                # 创建PyTorch Geometric数据
                pyg_data = from_networkx(G_mapped)
                pyg_data.x = torch.tensor(node_features, dtype=torch.float)
                pyg_data.y = torch.tensor(node_labels, dtype=torch.long)
                pyg_data.time_step = t
                pyg_data.original_node_ids = torch.tensor(node_list, dtype=torch.long)

                # 存储映射关系
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
        """保存时序图数据到文件"""
        print(f"构建时序图...")
        temporal_graphs = self.build_temporal_snapshots()

        print(f"保存时序图到 {save_path}...")

        save_data = {
            'temporal_graphs': temporal_graphs,
            'user_id_to_screen_names': dict(self.user_id_to_screen_names),
            'screen_name_to_user_id': self.screen_name_to_user_id,
            'user_id_to_internal_id': self.user_id_to_internal_id,
            'internal_id_to_user_id': self.internal_id_to_user_id,
            'malicious_users': self.malicious_users,
            'user_texts': dict(self.user_texts),
            'time_step_minutes': self.time_step_minutes,
            'node_id_embeddings': self.node_id_embeddings,
            'tweet_id_to_data': self.tweet_id_to_data
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"成功保存 {len(temporal_graphs)} 个时间步的图数据")

    def load_temporal_graphs(self, load_path: str):
        """从文件加载时序图数据"""
        print(f"从 {load_path} 加载时序图数据...")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        temporal_graphs = data['temporal_graphs']
        self.user_id_to_screen_names = defaultdict(set, data.get('user_id_to_screen_names', {}))
        self.screen_name_to_user_id = data.get('screen_name_to_user_id', {})
        self.user_id_to_internal_id = data.get('user_id_to_internal_id', {})
        self.internal_id_to_user_id = data.get('internal_id_to_user_id', {})
        self.malicious_users = data.get('malicious_users', set())
        self.user_texts = defaultdict(list, data.get('user_texts', {}))
        self.time_step_minutes = data.get('time_step_minutes', 20)
        self.node_id_embeddings = data.get('node_id_embeddings', {})
        self.tweet_id_to_data = data.get('tweet_id_to_data', {})

        print(f"加载了 {len(temporal_graphs)} 个时间步的图数据")
        return temporal_graphs

    def get_statistics(self):
        """获取数据统计信息"""
        if self.tweets_data:
            all_times = [tweet['parsed_created_at'] for tweet in self.tweets_data]
            time_range = (min(all_times), max(all_times))
        else:
            time_range = None

        malicious_user_ids = sum(1 for user_id in self.user_id_to_screen_names.keys() 
                                if self.is_malicious_user(user_id))

        return {
            'total_user_ids': len(self.user_id_to_internal_id),
            'total_tweets': len(self.tweets_data),
            'malicious_user_labels': len(self.malicious_users),
            'malicious_user_ids': malicious_user_ids,
            'time_range': time_range,
            'multi_name_users': sum(1 for names in self.user_id_to_screen_names.values() if len(names) > 1)
        }

    def verify_user_mapping(self, test_screen_names: List[str]):
        """验证用户映射"""
        print("\n=== 用户映射验证 ===")
        for screen_name in test_screen_names:
            if screen_name in self.screen_name_to_user_id:
                user_id_str = self.screen_name_to_user_id[screen_name]
                internal_id = self.user_id_to_internal_id.get(user_id_str)
                all_names = list(self.user_id_to_screen_names.get(user_id_str, []))
                is_malicious = "恶意" if screen_name in self.malicious_users else "正常"
                print(f"  {screen_name} -> user_id: {user_id_str}, internal_id: {internal_id}, 所有用户名: {all_names}, 类型: {is_malicious}")
            else:
                is_malicious = "恶意(仅标签)" if screen_name in self.malicious_users else "未找到"
                print(f"  {screen_name} -> {is_malicious}")


def main():
    """主函数"""
    json_path = "tweets.json"
    malicious_users_path = "tweets_users.txt"
    save_path = "twitter_temporal_graphs_1.pkl"

    # 创建时序图构建器
    builder = ImprovedTemporalGraphBuilder(
        json_path=json_path,
        malicious_users_path=malicious_users_path,
        time_step_minutes=20  
    )

    # 验证用户映射（使用恶意用户样例）
    test_users = ["TheWrightWingv2", "bowensb8", "spooney35", "HalloweenBlogs", "ollieblog"]
    builder.verify_user_mapping(test_users)

    # 构建并保存时序图
    builder.save_temporal_graphs(save_path)

    # 打印统计信息
    stats = builder.get_statistics()
    print(f"\n{'='*80}")
    print("=== Twitter数据统计 ===")
    print(f"{'='*80}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"{'='*80}")

    print("\n Twitter时序图构建完成！")


if __name__ == "__main__":
    main()