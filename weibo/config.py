import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据路径
    gat_encoded_dir: str = "/home/llm/yuanyuan_data/gat_encoded_subgraphs"
    text_encoded_dir: str = "/home/llm/yuanyuan_data/sbert_encoded_texts"
    original_graph_data_path: str = "/home/llm/yuanyuan_data/whole_temporal_graphs.pkl"

    # 模型配置
    structure_dim: int = 256
    text_dim: int = 384
    llm_input_dim: int = 3584
    max_sequence_length: int = 100
    num_classes: int = 2
    attention_hidden_dim: int = 128

    llm_model_path: str = "/home/llm/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
    tokenizer_path: Optional[str] = None

    # 原版 TempGT-LLM
    backbone_init_mode: str = "pretrained"
    experiment_name: str = "tempgt_llm_qwen"

    # 每次完整运行结束后，从内部 5run 中挑 1 个总体最优模型，保存到该目录
    selected_best_models_dir: Optional[str] = "selected_best_models/tempgt_llm_qwen"
    outer_experiment_tag: str = "exp3"

    lora_rank: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None

    # 训练参数
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    lora_learning_rate: float = 2e-5
    max_grad_norm: float = 1.0

    # 多次运行 / 随机种子
    num_runs: int = 1
    random_seeds: List[int] = field(default_factory=lambda: [342, 152, 162, 172, 182])
    split_seed: int = 42
    reuse_existing_split: bool = True

    # 数据集划分比例
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # 验证/测试集负:正比例控制
    # None: 沿用当前策略（训练集采样后剩余正常样本平均分给 val/test）
    # 10/20/30/40: 分别构造 1:10 / 1:20 / 1:30 / 1:40
    val_test_neg_pos_ratio: Optional[int] = 20

    debug_mode: str = "normal"

    # 损失权重
    lambda_align: float = 0.2
    lambda_gen: float = 0.6
    lambda_cee: float = 0.2

    temperature: float = 0.07

    # 对齐损失相关
    struct_timestep_dropout: float = 0.2
    align_queue_size: int = 512
    align_max_negatives: int = 10

    # CEE 相关超参
    cee_margin: float = 0.3

    malicious_to_normal_ratio: float = 1.0

    device: str = "auto"

    # 本次完整运行的输出目录
    output_dir: str = "joint_lora_output/tempgt_llm_qwen_exp3"
    save_frequency: int = 5

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        if self.tokenizer_path is None:
            self.tokenizer_path = self.llm_model_path

        if self.selected_best_models_dir is None:
            self.selected_best_models_dir = os.path.join("selected_best_models", self.experiment_name)

        if self.backbone_init_mode not in {"pretrained", "random_init"}:
            raise ValueError("backbone_init_mode 必须是 'pretrained' 或 'random_init'")

        if self.val_test_neg_pos_ratio is not None:
            if int(self.val_test_neg_pos_ratio) <= 0:
                raise ValueError("val_test_neg_pos_ratio 必须为正整数或 None")
            self.val_test_neg_pos_ratio = int(self.val_test_neg_pos_ratio)

        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)