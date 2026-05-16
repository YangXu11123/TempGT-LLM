import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class TrainingConfig:
    """训练配置"""
    # ================== 数据路径 ==================
    gat_encoded_dir: str = "gat_encoded_subgraphs"
    text_encoded_dir: str = "sbert_encoded_texts"
    original_graph_data_path: str = "twitter_temporal_graphs.pkl"  
    
    # 仅在推理阶段使用的 6000 正常用户编码目录
    normal_gat_encoded_dir: str = "gat_encoded_subgraphs_normal_6000"
    normal_text_encoded_dir: str = "sbert_encoded_texts_normal_6000"
  
    # ================== 模型配置 ==================
    structure_dim: int = 256
    text_dim: int = 384
    llm_input_dim: int = 3584  
    max_sequence_length: int = 100
    num_classes: int = 2

    # 注意力池化配置
    attention_hidden_dim: int = 128
    
    # ================== LLM / LoRA 配置 ==================
    llm_model_path: str = "/home/llm/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.3
    target_modules: List[str] = None  # 如果为 None，将自动选择
    
    # ================== 训练参数 ==================
    num_epochs: int = 4
    batch_size: int = 2
    learning_rate: float = 2e-5
    lora_learning_rate: float = 5e-6
    max_grad_norm: float = 1.0      # 梯度裁剪阈值

    # ====== 数据集划分比例 ======
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # ====== 调试 / 验证 soft token 是否被使用 ======
    # "normal"  : 正常使用时间序列向量
    # "shuffle" : 打乱每个样本时间步顺序（只打乱，不改变值）
    # "zero"    : 把所有时间步向量置零
    debug_mode: str = "normal"

    # ================== 损失权重 ==================
    lambda_align: float = 0.2   # 图-文对齐损失权重
    lambda_gen: float = 0.7     # 生成式分类损失权重
    lambda_cee: float = 0.1     # CEE 结构时序演化损失权重

    # 对齐损失相关参数
    temperature: float = 0.07   # 对齐损失温度参数
    struct_timestep_dropout: float = 0.2  # 结构时间步 dropout 比例
    align_queue_size: int = 512          # 每类 memory bank 长度
    align_max_negatives: int = 10        # 每步每类使用的最大负样本数

    # CEE 相关参数
    cee_margin: float = 0.3    # CEE ranking 的 margin（auth_cee > cib_cee + margin）
    
    # ================== 数据采样 ==================
    malicious_to_normal_ratio: float = 1.0  # 恶意:正常 = 1:1
    
    # 推理阶段额外正常用户采样
    extra_normal_sample_size: int = 6000
    extra_normal_sample_seed: int = 68

    device: str = 'auto'
    
    # ================== 输出 ==================
    output_dir: str = "joint_lora_output"
    save_frequency: int = 5  
    
    def __post_init__(self):
        # 自动选择 LoRA 注入模块
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        if self.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.device)
