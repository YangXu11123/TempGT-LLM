import torch
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(CODE_DIR)
ARTIFACT_ROOT = os.path.join(WORKSPACE_ROOT, "new_exe_artifacts")


@dataclass
class TrainingConfig:
    """训练配置"""
    # ================== 数据路径 ==================
    # 结构侧由新版 GAT 重建到独立目录；文本侧继续只读复用已有编码。
    gat_encoded_dir: str = os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs")
    text_encoded_dir: str = os.path.join(WORKSPACE_ROOT, "sbert_encoded_texts")
    original_graph_data_path: str = os.path.join(WORKSPACE_ROOT, "twitter_temporal_graphs.pkl")
    
    # 仅在推理阶段使用的 6000 正常用户编码目录
    normal_gat_encoded_dir: str = os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs_normal_6000")
    normal_text_encoded_dir: str = os.path.join(WORKSPACE_ROOT, "sbert_encoded_texts_normal_6000")
    load_extra_normal_embeddings: bool = True

    # L_cont 的 edge-drop 增强结构视图目录（由 gat_encoder.py 生成）
    edge_drop_gat_encoded_dir: str = os.path.join(ARTIFACT_ROOT, "gat_encoded_subgraphs_edge_drop")
    normal_edge_drop_gat_encoded_dir: str = os.path.join(
        ARTIFACT_ROOT, "gat_encoded_subgraphs_normal_6000_edge_drop"
    )
    require_edge_drop_alignment: bool = True

    # ================== 模型配置 ==================
    structure_dim: int = 256
    text_dim: int = 384
    llm_input_dim: int = 3584  
    model_family: str = 'qwen'
    llm_dtype: str = 'bfloat16'
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
    # Shared-server safe defaults: no worker prefetch or page-locked copies.
    data_loader_workers: int = 0
    pin_memory: bool = False
    learning_rate: float = 2e-5
    lora_learning_rate: float = 5e-6
    max_grad_norm: float = 1.0      # 梯度裁剪阈值
    training_seed: int = 42
    contrastive_seed: int = 1051
    optimizer_weight_decay: float = 1e-5
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    scheduler_eta_min: float = 1e-6
    evaluation_ratios: List[int] = None
    evaluation_seeds: List[int] = None

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
    lambda_align: float = 0.2   # L_cont 对比损失权重
    lambda_gen: float = 0.7     # L_cls PU pairwise 排序损失权重
    lambda_cee: float = 0.1     # L_CEE 边际排序正则权重

    # 对齐损失相关参数
    temperature: float = 0.07   # 对齐损失温度参数
    # L_cont 仅使用当前无标签 batch 内的其他账号作为负样本。

    # ================== CEE 冻结头 ==================
    cee_head_path: str = os.path.join(ARTIFACT_ROOT, "cee_dynamics_head.pth")
    require_pretrained_cee: bool = True  # True: 必须先跑 pretrain_cee_head.py
    cee_hidden_dim: int = 512
    cee_sigma: float = 1.0
    cee_score_definition: str = "mean_conditional_log_likelihood_v2"
    cee_include_constant: bool = True
    cee_state_seed: int = 20260911       # CEE 预训练随机种子
    cee_pretrain_epochs: int = 20
    cee_pretrain_lr: float = 1e-3
    cee_margin: float = 0.3              # δ_d：CIB 的 CEE 需低于正常一个 gap

    # ================== L_cls PU pairwise ==================
    cls_margin: float = 0.3              # δ_s：正负得分间隔
    pair_seed: int = 42                  # 训练前一次性预采样配对的种子
    pairwise_train_batches: bool = True  # 训练批按固定 (CIB, normal) 配对组织
    contrastive_batch_size: int = 2      # L_cont 独立无标签 batch，不参与 PU 配对

    # ================== 数据划分复现 ==================
    split_seed: int = 42                 # 分层 6:2:2 划分种子
    extra_normal_seed: int = 0           # normal_6000 注入 Val/Test 的随机种子
    extra_normal_inject_each: int = 3000
    reuse_existing_split: bool = True    # 复用已有 dataset_split.json（保证 CEE 预训练与主训练同划分）

    # ================== 数据采样 ==================
    malicious_to_normal_ratio: float = 1.0  # 恶意:正常 = 1:1
    
    # 推理阶段额外正常用户采样
    extra_normal_sample_size: int = 6000
    extra_normal_sample_seed: int = 68

    device: str = 'auto'
    
    # ================== 输出 ==================
    output_dir: str = os.path.join(ARTIFACT_ROOT, "joint_logp_output")
    save_frequency: int = 5  
    
    def __post_init__(self):
        if self.evaluation_ratios is None:
            self.evaluation_ratios = [10, 20, 30, 40, 50]
        if self.evaluation_seeds is None:
            self.evaluation_seeds = [42, 43, 44, 45, 46]
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


def load_config(path=None):
    import json
    from dataclasses import fields
    if path is None:
        return TrainingConfig()
    with open(path, encoding='utf-8') as handle:
        values = json.load(handle)
    unknown = set(values) - {field.name for field in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    return TrainingConfig(**values)
