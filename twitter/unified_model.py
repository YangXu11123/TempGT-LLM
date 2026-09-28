import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, Tuple, List, Optional

from cee_dynamics import CEEDynamicsHead


class LearnableAttentionPooling(nn.Module):
    """
    可学习注意力池化:
        输入:  x: [B, N, D]
               mask: [B, N] (bool, True 表示有效位置)
        输出:  pooled: [B, D]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        B, N, D = x.shape

        # 确保 mask 在同一设备上，且是 bool
        mask = mask.to(x.device).bool()        # [B, N]
        mask_expanded = mask.unsqueeze(-1)     # [B, N, 1]

        # 无效位置置零，仅用于特征
        x_masked = x * mask_expanded           # [B, N, D]

        # 原始打分
        scores = self.attention_net(x_masked)  # [B, N, 1]

        # 将无效位置打分设为 -inf，用于 softmax 掩码
        scores = scores.masked_fill(~mask_expanded, float("-inf"))  # [B, N, 1]

        # valid_any[i] = True 表示第 i 个样本至少有一个有效位置
        valid_any = mask.any(dim=1, keepdim=True)  # [B, 1]

        scores = torch.where(
            valid_any.unsqueeze(-1),   # [B, 1, 1]
            scores,
            torch.zeros_like(scores)   # 对于全 False 的样本，scores 全 0
        )

        # 计算注意力权重
        attn = torch.softmax(scores, dim=1)         # [B, N, 1]

        # 再次应用 mask，防止极端数值
        attn = attn * mask_expanded.float()         # [B, N, 1]

        # 沿 N 维重新归一化，让有效位置的权重和为 1；若没有有效位置则保持 0
        denom = attn.sum(dim=1, keepdim=True)       # [B, 1, 1]
        # 只有 denom>0 的样本才归一化
        attn = torch.where(
            denom > 0,
            attn / (denom + 1e-12),
            attn
        )  # [B, N, 1]

        # 最终加权和
        pooled = torch.sum(attn * x, dim=1)         # [B, D]

        return pooled

class UnifiedTemporalGTLLM(nn.Module):
    """统一的时间感知图-文LLM模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._external_tokenizer = None  

        # 时序图 / 文本的注意力池化 (可训练的 u, w) 
        struct_node_dim = config.structure_dim // 2  # 128
        self.struct_att_pool = LearnableAttentionPooling(
            input_dim=struct_node_dim,
            hidden_dim=config.attention_hidden_dim,
        )
        self.text_att_pool = LearnableAttentionPooling(
            input_dim=config.text_dim,
            hidden_dim=config.attention_hidden_dim,
        )

        # 图文对齐投影层（W_proj）
        self.text_to_struct_proj = nn.Linear(
            config.text_dim, config.structure_dim, bias=False
        )

        # 时序位置编码（P_t）
        self.temporal_position_embedding = nn.Embedding(
            config.max_sequence_length, 2 * config.structure_dim  # 512
        )

        # Frozen CEE dynamics head: mu_theta(x_t) -> x_{t+1}.
        # 必须先由 pretrain_cee_head.py 在训练集上无监督预训练，随后全程冻结。
        self.cee_state_dim = 2 * config.structure_dim
        self.cee_head = CEEDynamicsHead(
            input_dim=self.cee_state_dim,
            hidden_dim=getattr(config, "cee_hidden_dim", self.cee_state_dim),
        )
        self._load_and_freeze_cee_head()

        # 向量到LLM输入的映射层（W_e）
        self.llm_projector = nn.Linear(
            2 * config.structure_dim, config.llm_input_dim, bias=False
        )

        # LLM模型 + LoRA
        print("加载LLM模型...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_path,
            torch_dtype=getattr(torch, getattr(config, 'llm_dtype', 'bfloat16')),
            device_map=None,
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        actual_dim = self.llm.get_input_embeddings().weight.shape[1]
        if actual_dim != config.llm_input_dim:
            raise ValueError(f'LLM embedding dimension {actual_dim} != configured {config.llm_input_dim}')
        self.llm.config.pad_token_id = self.llm.config.eos_token_id

        # 冻结非 LoRA 参数
        for name, param in self.llm.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        self._count_trainable_params()
        self.llm_dtype = torch.bfloat16

        # 前缀缓存
        self._time_prefix_cache = {}
        self._summary_prefix_embedding = None

        self.to(config.device)

    def train(self, mode: bool = True):
        """Keep the pretrained CEE head in eval mode even when the joint model trains."""
        super().train(mode)
        self.cee_head.eval()
        return self

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(path)

    def _load_and_freeze_cee_head(self) -> None:
        cee_path = self._resolve_path(getattr(self.config, "cee_head_path", ""))
        require_pretrained = bool(getattr(self.config, "require_pretrained_cee", True))

        if not cee_path:
            if require_pretrained:
                raise FileNotFoundError("config.cee_head_path is empty, but require_pretrained_cee=True")
            self.cee_head.freeze()
            return

        if not os.path.exists(cee_path):
            if require_pretrained:
                raise FileNotFoundError(
                    f"Frozen CEE dynamics checkpoint not found: {cee_path}. "
                    "Run: python pretrain_cee_head.py before python train.py"
                )
            print(f"[CEE] 未找到预训练 checkpoint，使用随机冻结 CEE head: {cee_path}")
            self.cee_head.freeze()
            return

        checkpoint = torch.load(cee_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        ckpt_input_dim = checkpoint.get("input_dim") if isinstance(checkpoint, dict) else None
        ckpt_hidden_dim = checkpoint.get("hidden_dim") if isinstance(checkpoint, dict) else None
        if ckpt_input_dim is not None and int(ckpt_input_dim) != int(self.cee_state_dim):
            raise ValueError(f"CEE checkpoint input_dim={ckpt_input_dim}, expected {self.cee_state_dim}")
        if ckpt_hidden_dim is not None and int(ckpt_hidden_dim) != int(getattr(self.config, "cee_hidden_dim", self.cee_state_dim)):
            raise ValueError(
                f"CEE checkpoint hidden_dim={ckpt_hidden_dim}, "
                f"expected {getattr(self.config, 'cee_hidden_dim', self.cee_state_dim)}"
            )

        self.cee_head.load_state_dict(state_dict, strict=True)

        builder_state = checkpoint.get("state_builder_state_dict") if isinstance(checkpoint, dict) else None
        if isinstance(builder_state, dict):
            missing = []
            if "struct_att_pool" in builder_state:
                self.struct_att_pool.load_state_dict(builder_state["struct_att_pool"], strict=True)
            else:
                missing.append("struct_att_pool")
            if "text_att_pool" in builder_state:
                self.text_att_pool.load_state_dict(builder_state["text_att_pool"], strict=True)
            else:
                missing.append("text_att_pool")
            if "text_to_struct_proj" in builder_state:
                self.text_to_struct_proj.load_state_dict(builder_state["text_to_struct_proj"], strict=True)
            else:
                missing.append("text_to_struct_proj")
            if missing:
                print(f"[CEE] checkpoint 缺少 state builder 初始权重: {missing}")

        self.cee_head.freeze()
        print(f"[CEE] 已加载并冻结 CEE dynamics head: {cee_path}")

    def set_tokenizer(self, tokenizer):
        self._external_tokenizer = tokenizer
        self.llm_tokenizer = tokenizer  

        # 测试 '0' 和 '1' 的编码结果 
        with torch.no_grad():
            ids_0 = tokenizer("0", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            ids_1 = tokenizer("1", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        print("encode('0'):", ids_0.tolist())
        print("encode('1'):", ids_1.tolist())


    def _get_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            return tokenizer
        if self._external_tokenizer is not None:
            return self._external_tokenizer
        raise ValueError("Tokenizer must be provided either via forward(tokenizer=...) or set_tokenizer().")

    def _count_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        modules = {
            "struct_att_pool": self.struct_att_pool,
            "text_att_pool": self.text_att_pool,
            "text_to_struct_proj": self.text_to_struct_proj,
            "temporal_position_embedding": self.temporal_position_embedding,
            "cee_head_frozen": self.cee_head,
            "llm_projector": self.llm_projector,
            "llm_lora": self.llm,
        }

        for name, module in modules.items():
            if name == "llm_lora":
                lora_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {lora_params:,} (LoRA参数)")
            else:
                module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {module_params:,}")


    # 节点/时间级池化 
    def _pool_from_node_level(
        self,
        struct_in_embeddings: torch.Tensor,    # [B, T, N_in, 128]
        struct_out_embeddings: torch.Tensor,   # [B, T, N_out, 128]
        text_node_embeddings: torch.Tensor,    # [B, T, N_txt, 384]
        struct_in_mask: torch.Tensor,          # [B, T, N_in]
        struct_out_mask: torch.Tensor,         # [B, T, N_out]
        text_mask: torch.Tensor,               # [B, T, N_txt]
        time_mask: torch.Tensor,               # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = struct_in_embeddings.device
        B, T, N_in, D_in = struct_in_embeddings.shape
        _, _, N_out, D_out = struct_out_embeddings.shape
        _, _, N_txt, D_txt = text_node_embeddings.shape

        in_flat = struct_in_embeddings.view(B * T, N_in, D_in)
        out_flat = struct_out_embeddings.view(B * T, N_out, D_out)
        in_mask_flat = struct_in_mask.view(B * T, N_in)
        out_mask_flat = struct_out_mask.view(B * T, N_out)

        h_in_flat = self.struct_att_pool(in_flat, in_mask_flat)     # [B*T, 128]
        h_out_flat = self.struct_att_pool(out_flat, out_mask_flat)  # [B*T, 128]

        h_in = h_in_flat.view(B, T, D_in)
        h_out = h_out_flat.view(B, T, D_out)

        structure_embeddings = torch.cat([h_in, h_out], dim=-1)  # [B, T, 256]

        txt_flat = text_node_embeddings.view(B * T, N_txt, D_txt)
        txt_mask_flat = text_mask.view(B * T, N_txt)

        h_txt_flat = self.text_att_pool(txt_flat, txt_mask_flat)  # [B*T, 384]
        text_embeddings = h_txt_flat.view(B, T, D_txt)            # [B, T, 384]

        attention_mask = time_mask.to(device)
        return structure_embeddings, text_embeddings, attention_mask

    # 节点级全局时序表征（L_cont 输入 x̃_t，带时序位置编码） & edge-drop 对齐

    def _build_global_sequence_repr(
        self,
        structure_embeddings: torch.Tensor,  # [B, T, 256]
        text_embeddings: torch.Tensor,       # [B, T, 384]
        attention_mask: torch.Tensor,        # [B, T]
        timesteps: Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor:
        text_aligned = self.text_to_struct_proj(text_embeddings)  # [B, T, 256]
        fused = torch.cat([structure_embeddings, text_aligned], dim=-1)  # [B, T, 512]

        # x̃_t 带可学习时序位置编码 p_t（仅供 L_cont；CEE 用的 x_t 不带位置编码）
        if timesteps is not None:
            pos_indices = torch.clamp(timesteps.long(), 0, self.config.max_sequence_length - 1)
            fused = fused + self.temporal_position_embedding(pos_indices)

        time_mask = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
        fused_masked = fused * time_mask                 # [B, T, 512]

        lengths = time_mask.sum(dim=1)                   # [B, 1]
        lengths = lengths.clamp(min=1)
        h = fused_masked.sum(dim=1) / lengths            # [B, 512]
        return h

    # 对齐损失（edge-drop 第二视角 + 当前无标签 batch 内负样本）

    def compute_alignment_loss(
        self,
        structure_embeddings: torch.Tensor,                # [B, T, 256]
        structure_embeddings_aug: Optional[torch.Tensor],  # [B, T, 256]
        text_embeddings: torch.Tensor,                     # [B, T, 384]
        attention_mask: torch.Tensor,                      # [B, T]
        timesteps: torch.Tensor,                           # [B, T]
        temperature: float = 0.07,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        device = structure_embeddings.device

        if structure_embeddings_aug is None:
            if self.training and bool(getattr(self.config, "require_edge_drop_alignment", True)):
                raise ValueError("L_cont requires edge-drop GAT embeddings, but augmented structure view is missing")
            return torch.tensor(0.0, device=device)

        # 1) 两种视角的全局表示 h1, h2（输入均为带位置编码的 x̃_t）
        h1 = self._build_global_sequence_repr(
            structure_embeddings, text_embeddings, attention_mask, timesteps
        )  # [B, 512]

        h2 = self._build_global_sequence_repr(
            structure_embeddings_aug, text_embeddings, attention_mask, timesteps
        )  # [B, 512]

        # 2) L2 归一化
        h1_norm = F.normalize(h1, p=2, dim=-1, eps=epsilon)
        h2_norm = F.normalize(h2, p=2, dim=-1, eps=epsilon)

        batch_size = h1_norm.size(0)

        # 3) InfoNCE：正样本 = 同一账号两个视角；负样本 = batch 内其他账号。
        logits = torch.matmul(h1_norm, h2_norm.t()) / temperature
        targets = torch.arange(batch_size, device=device, dtype=torch.long)
        return F.cross_entropy(logits, targets, reduction="mean")
        
    # CEE 计算 
    def _compute_cee_for_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        用冻结的 mu_theta 对单个账号状态序列计算 CEE 值。

        参数:
            states: [T, d]，d = 2 * structure_dim（结构 + 文本投影拼接，无位置编码的 x_t）

        返回:
            cee: mean logp; higher means more predictable (method 2).
        """
        device = states.device
        T = states.size(0)
        if T < 2:
            return torch.tensor(0.0, device=device)
        return self.cee_head.compute_log_likelihood(
            states,
            sigma=getattr(self.config, "cee_sigma", 1.0),
            include_constant=bool(getattr(self.config, "cee_include_constant", True)),
        )

    def compute_cee_loss(
        self,
        structure_embeddings: torch.Tensor,   # [B, T, 256]
        text_embeddings: torch.Tensor,        # [B, T, 384]
        attention_mask: torch.Tensor,         # [B, T]  时间步有效 mask
        labels: torch.Tensor,                 # [B]，0=normal,1=CIB
        return_values: bool = False,
    ):
        
        device = structure_embeddings.device
        labels = labels.to(device)

        B, T, struct_dim = structure_embeddings.shape
        text_dim = text_embeddings.size(-1)

        # 文本先映射到结构空间，然后与结构拼接作为状态向量 s_t
        proj_text = self.text_to_struct_proj(
            text_embeddings.view(B * T, text_dim)
        ).view(B, T, struct_dim)                    # [B, T, 256]

        fused_states = torch.cat(
            [structure_embeddings, proj_text], dim=-1
        )  # [B, T, 512]

        # 逐账号计算 CEE_i
        cee_values = []
        for i in range(B):
            mask_i = attention_mask[i].bool()      # [T]
            states_i = fused_states[i, mask_i]     # [T_i, 512]
            if states_i.size(0) < 2:
                cee_i = torch.tensor(0.0, device=device)
            else:
                cee_i = self._compute_cee_for_states(states_i)
            cee_values.append(cee_i)

        if len(cee_values) == 0:
            empty_values = torch.empty(0, device=device)
            zero_loss = torch.tensor(0.0, device=device)
            return (zero_loss, empty_values) if return_values else zero_loss

        cee_values = torch.stack(cee_values)       # [B]

        # 拆分 CIB / authentic
        is_cib = (labels == 1)
        is_auth = (labels == 0)

        if not (is_cib.any() and is_auth.any()):
            # 本 batch 没有同时包含两类，CEE 正则不生效（保持计算图连通）
            zero_loss = cee_values.sum() * 0.0
            return (zero_loss, cee_values) if return_values else zero_loss

        cib_cee = cee_values[is_cib]               # [N_cib]
        auth_cee = cee_values[is_auth]             # [N_auth]

        margin = getattr(self.config, "cee_margin", 0.1)

        # L_CEE = E_{(v+,v-)} max(0, CEE(v+) - CEE(v-) + δ_d)，v+ = CIB
        # Method 2: logp(CIB) below logp(normal), i.e. larger CIB NLL.
        l_cee = torch.relu(cib_cee[:, None] - auth_cee[None, :] + margin).mean()

        return (l_cee, cee_values) if return_values else l_cee


    # soft token 构造 
    def _build_sequence_tokens(
        self,
        structure_embeddings: torch.Tensor,  # [B, seq, 256]
        text_embeddings: torch.Tensor,       # [B, seq, 384]
        attention_mask: torch.Tensor,        # [B, seq]
        timesteps: torch.Tensor,             # [B, seq]
        debug_mode: str = "normal",
    ) -> Tuple[List[torch.Tensor], None, None]:
        device = structure_embeddings.device
        batch_size, seq_len, _ = structure_embeddings.shape

        text_aligned_seq = self.text_to_struct_proj(text_embeddings)  # [B, seq, 256]

        llm_token_sequences: List[torch.Tensor] = []

        for i in range(batch_size):
            valid_len = int(attention_mask[i].sum().item())
            if valid_len <= 0 or valid_len > seq_len:
                valid_len = seq_len

            struct_valid = structure_embeddings[i, :valid_len]
            text_aligned_valid = text_aligned_seq[i, :valid_len]
            time_valid = timesteps[i, :valid_len].long()

            fused_256 = torch.cat([struct_valid, text_aligned_valid], dim=-1)  # [L, 512]

            pos_indices = torch.clamp(time_valid, 0, self.config.max_sequence_length - 1)
            pos_embed = self.temporal_position_embedding(pos_indices)          # [L, 512]

            fused_with_pos = fused_256 + pos_embed
            if debug_mode == "normal":
                fused_for_llm = fused_with_pos
            elif debug_mode == "shuffle":
                perm = torch.randperm(valid_len, device=device)
                fused_for_llm = fused_with_pos[perm]
            elif debug_mode == "zero":
                fused_for_llm = torch.zeros_like(fused_with_pos)
            else:
                fused_for_llm = fused_with_pos

            # L_cls 的梯度边界止于 W_e：分类损失只更新 W_e 与 LoRA，
            # pooling/W_proj/位置编码由 L_cont 与 L_CEE 更新。
            llm_tokens = self.llm_projector(fused_for_llm.detach())  # [L, D_llm]
            llm_token_sequences.append(llm_tokens)

        return llm_token_sequences, None, None

    # prompt 构造 

    def _get_or_build_time_prefix(self, tokenizer, device):
        if "time_prefix" in self._time_prefix_cache:
            return self._time_prefix_cache["time_prefix"]

        system_prompt = (
            "请你作为一个专门分析社交网络用户行为的安全专家。"
            "你会看到某个用户在不同时间步的结构特征和文本特征总结。"
            "你的任务是根据这些特征判断该用户是否是恶意账号（参与协调不良行为）。"
        )
        behavior_prefix = (
            "接下来是该用户在一段时间内的图结构与文本行为特征的编码表示，"
            "每一行对应一个时间步的特征。"
        )
        question_prefix = (
            "请综合上述时间序列特征判断该用户是否为恶意账号（参与协调不良行为），"
            "只回答 '0' 或 '1'（0 表示 Benign，1 表示 Malicious）。"
        )

        full_prefix = (
            "[INST]<<SYS>>" + system_prompt + "<</SYS>>\n" +
            behavior_prefix + "\n" +
            question_prefix + "[/INST]"
        )

        with torch.no_grad():
            # 伪 token pipeline 不额外引入 BOS/EOS；prompt 中需要的文字标记已显式写入。
            input_ids = tokenizer(
                full_prefix,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            prefix_embeddings = self.llm.get_input_embeddings()(input_ids)[0]  # [L, D_llm]

        self._time_prefix_cache["time_prefix"] = prefix_embeddings
        return prefix_embeddings

    def construct_prompts(
        self,
        tokenizer,
        llm_token_sequences: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(llm_token_sequences)
        time_prefix_embeddings = self._get_or_build_time_prefix(tokenizer, device)
        prefix_len, d_llm = time_prefix_embeddings.shape

        seq_lengths = [tokens.size(0) for tokens in llm_token_sequences]
        max_seq_len = max(seq_lengths)
        total_len = prefix_len + max_seq_len

        inputs_embeds = torch.zeros(batch_size, total_len, d_llm, device=device, dtype=self.llm_dtype)
        attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)

        time_prefix_embeddings = time_prefix_embeddings.to(self.llm_dtype)

        for i in range(batch_size):
            L_i = seq_lengths[i]
            tokens_i = llm_token_sequences[i]

            inputs_embeds[i, :prefix_len] = time_prefix_embeddings
            attention_mask[i, :prefix_len] = 1

            inputs_embeds[i, prefix_len:prefix_len + L_i] = tokens_i.to(self.llm_dtype)
            attention_mask[i, prefix_len:prefix_len + L_i] = 1

        return inputs_embeds, attention_mask

    # forward
    def forward(
        self,
        struct_in_embeddings: torch.Tensor,
        struct_out_embeddings: torch.Tensor,
        text_node_embeddings: torch.Tensor,
        struct_in_mask: torch.Tensor,
        struct_out_mask: torch.Tensor,
        text_mask: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        struct_in_embeddings_aug: Optional[torch.Tensor] = None,
        struct_out_embeddings_aug: Optional[torch.Tensor] = None,
        struct_in_mask_aug: Optional[torch.Tensor] = None,
        struct_out_mask_aug: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tokenizer=None,
        loss_mode: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        注意：这里的 attention_mask 是时间步级的 [B, T]（train.py 里传进来的 time_mask）
        """

        device = self.config.device
        tokenizer = self._get_tokenizer(tokenizer)

        # --- 0. 损失权重（lambda==0 时硬消融：不计算、不更新对应分支） ---
        if loss_mode not in {"all", "pairwise", "contrastive"}:
            raise ValueError(f"unsupported loss_mode={loss_mode!r}")
        lambda_gen = float(getattr(self.config, "lambda_gen", 0.5))
        lambda_align = float(getattr(self.config, "lambda_align", 0.5))
        lambda_cee = float(getattr(self.config, "lambda_cee", 0.1))
        if loss_mode == "pairwise":
            lambda_align = 0.0
        elif loss_mode == "contrastive":
            lambda_gen = 0.0
            lambda_cee = 0.0

        # --- 1. 移到设备 ---
        struct_in_embeddings = struct_in_embeddings.to(device)
        struct_out_embeddings = struct_out_embeddings.to(device)
        text_node_embeddings = text_node_embeddings.to(device)
        struct_in_mask = struct_in_mask.to(device)
        struct_out_mask = struct_out_mask.to(device)
        text_mask = text_mask.to(device)
        attention_mask = attention_mask.to(device)
        timesteps = timesteps.to(device)
        if struct_in_embeddings_aug is not None:
            struct_in_embeddings_aug = struct_in_embeddings_aug.to(device)
        if struct_out_embeddings_aug is not None:
            struct_out_embeddings_aug = struct_out_embeddings_aug.to(device)
        if struct_in_mask_aug is not None:
            struct_in_mask_aug = struct_in_mask_aug.to(device)
        if struct_out_mask_aug is not None:
            struct_out_mask_aug = struct_out_mask_aug.to(device)

        # --- 2. 节点级 -> 时间步级 (attention pooling) ---
        structure_embeddings, text_embeddings, attention_mask = self._pool_from_node_level(
            struct_in_embeddings,
            struct_out_embeddings,
            text_node_embeddings,
            struct_in_mask,
            struct_out_mask,
            text_mask,
            attention_mask,
        )  # [B, T, 256], [B, T, 384], [B, T]

        structure_embeddings_aug = None
        has_aug_view = lambda_align > 0.0 and all(
            x is not None
            for x in (
                struct_in_embeddings_aug,
                struct_out_embeddings_aug,
                struct_in_mask_aug,
                struct_out_mask_aug,
            )
        )
        if has_aug_view:
            structure_embeddings_aug, _, _ = self._pool_from_node_level(
                struct_in_embeddings_aug,
                struct_out_embeddings_aug,
                text_node_embeddings,
                struct_in_mask_aug,
                struct_out_mask_aug,
                text_mask,
                attention_mask,
            )

        # --- 3. L_cont 对齐损失（edge-drop 第二视角 + label-free memory bank） ---
        # 硬消融：当 lambda_align == 0 时，完全跳过计算与队列更新。
        if (lambda_align is not None) and float(lambda_align) > 0.0:
            align_loss = self.compute_alignment_loss(
                structure_embeddings,
                structure_embeddings_aug,
                text_embeddings,
                attention_mask,
                timesteps,
                temperature=getattr(self.config, "temperature", 0.07),
            )
        else:
            align_loss = torch.tensor(0.0, device=device)


        # --- 3.5 L_CEE 正则（仅在有标签且启用时计算；分数来自冻结 CEEHead） ---
        # 硬消融：当 lambda_cee == 0 时，完全跳过计算。
        cee_scores = None
        if (labels is not None) and ((lambda_cee is not None) and float(lambda_cee) > 0.0):
            cee_loss, cee_scores = self.compute_cee_loss(
                structure_embeddings,
                text_embeddings,
                attention_mask,
                labels,
                return_values=True,
            )
        else:
            cee_loss = torch.tensor(0.0, device=device)

        # 对比学习独立于标签配对，也不需要构造伪 token 或运行 7B LLM。
        if loss_mode == "contrastive":
            return {
                "align_loss": align_loss,
                "gen_loss": torch.tensor(0.0, device=device),
                "cee_loss": torch.tensor(0.0, device=device),
                "total_loss": lambda_align * align_loss,
            }

        # --- 4. 构造 LLM soft tokens ---
        llm_token_sequences, _, _ = self._build_sequence_tokens(
            structure_embeddings,
            text_embeddings,
            attention_mask,
            timesteps,
            debug_mode=getattr(self.config, "debug_mode", "normal"),
        )

        # --- 5. 构造 prompt 输入 LLM ---
        inputs_embeds, llm_attention_mask = self.construct_prompts(
            tokenizer, llm_token_sequences, device
        )

        from backbone_adapter import forward_embeddings
        outputs = forward_embeddings(self.llm, inputs_embeds, llm_attention_mask,
                                     getattr(self.config, 'model_family', 'qwen'))
        logits = outputs.logits  # [B, L, vocab]

        # 每个样本必须在自己的最后一个有效输入位置读取 next-token logits。
        # construct_prompts 使用右侧 padding；直接 logits[:, -1, :] 会让短序列
        # 从被 mask 的 padding 位置取分数。
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_valid_positions = llm_attention_mask.long().sum(dim=1).sub(1).clamp(min=0)
        last_logits = logits[batch_indices, last_valid_positions, :]

        benign_token_id = tokenizer(
            "0", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0][-1].item()
        malicious_token_id = tokenizer(
            "1", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0][-1].item()

        benign_logits = last_logits[:, benign_token_id]
        malicious_logits = last_logits[:, malicious_token_id]
        logits_2 = torch.stack([benign_logits, malicious_logits], dim=-1).float()

        # 无标签模式：仅输出 logits（loss 仅用于调试；启用的分支由 lambda_* 控制）
        if labels is None:
            return {
                "logits": logits,
                "logits_2": logits_2,
                "last_valid_positions": last_valid_positions,
                "align_loss": align_loss.detach(),
                "gen_loss": torch.tensor(0.0, device=device),
                "cee_loss": cee_loss.detach(),
                "total_loss": (lambda_align * align_loss + lambda_cee * cee_loss).detach(),
            }

        # --- 6. L_cls PU pairwise 排序损失（Benign / Malicious 二分类） ---
        labels = labels.to(device)

        # L_cls = E_{(u+,u^-)} max(0, δ_s - (score(u+) - score(u^-)))
        # CIB 拉高，正常不直接压到 0，而是保持 δ_s 的 gap
        score = logits_2[:, 1] - logits_2[:, 0]
        pos_score = score[labels == 1]
        neg_score = score[labels == 0]
        if pos_score.numel() > 0 and neg_score.numel() > 0:
            margin = float(getattr(self.config, "cls_margin", 0.3))
            gen_loss = torch.relu(margin - (pos_score[:, None] - neg_score[None, :])).mean()
        else:
            gen_loss = score.sum() * 0.0

        # CE 只保留为日志指标，不进入 total_loss
        ce_loss = F.cross_entropy(logits_2, labels.long(), reduction="mean")

        # --- 7. 总损失 = L_cls + L_cont + L_CEE（权重可调） ---

        total_loss = (
            lambda_gen * gen_loss
            + lambda_align * align_loss
            + lambda_cee * cee_loss
        )

        return {
            "logits": logits,
            "logits_2": logits_2,
            "align_loss": align_loss,
            "gen_loss": gen_loss,
            "ce_loss": ce_loss,
            "cee_loss": cee_loss,
            "cee_scores": cee_scores,
            "total_loss": total_loss,
        }

    # 保存 / 加载 

    def save_model(self, path: str):
        from asset_provenance import capture_assets
        # 只保存可训练参数。冻结的 7B LLM 主干与 CEEHead 均从各自的预训练
        # 路径加载，重复写入会让每个 checkpoint 膨胀到十几 GB。
        trainable_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        full_state = self.state_dict()
        trainable_state = {
            name: tensor.detach().cpu()
            for name, tensor in full_state.items()
            if name in trainable_names
        }
        torch.save(
            {
                "checkpoint_format": "trainable_only_v1",
                "model_state_dict": trainable_state,
                "trainable_parameter_names": sorted(trainable_names),
                "config": self.config,
                "asset_manifest": capture_assets(self.config),
            },
            path,
        )
        print(f"轻量模型保存到: {path} ({len(trainable_state)} 个可训练张量)")

    def load_model(self, path: str):
        """
        安全加载模型：
        1) checkpoint 先加载到 CPU，避免 GPU 上临时展开整份 checkpoint 导致显存峰值过高；
        2) 再把 state_dict 拷贝进当前模型；
        3) 最后确保模型位于目标设备，且 CEE head 保持冻结。
        """
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            from asset_provenance import verify_assets
            verify_assets(checkpoint.get('asset_manifest'), self.config)
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        checkpoint_format = checkpoint.get("checkpoint_format") if isinstance(checkpoint, dict) else None
        if checkpoint_format == "trainable_only_v1":
            expected = set(checkpoint.get("trainable_parameter_names", []))
            missing_saved = expected - set(state_dict.keys())
            if missing_saved:
                raise RuntimeError(f"轻量 checkpoint 缺少声明的参数: {sorted(missing_saved)[:10]}")

        if len(missing_keys) > 0:
            print(f"[load_model] missing_keys: {missing_keys[:10]} ... 共 {len(missing_keys)} 个")
        if len(unexpected_keys) > 0:
            print(f"[load_model] unexpected_keys: {unexpected_keys[:10]} ... 共 {len(unexpected_keys)} 个")

        self.to(self.config.device)
        self.cee_head.freeze()

        del checkpoint
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"模型加载成功: {path}")
