import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, Tuple, List, Optional

from cee_dynamics import CEEDynamicsHead, transform_cee_nll_to_score


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
        """
        x:    [B, N, D]
        mask: [B, N]  bool, True 表示有效位置

        设计目标：
        - 有有效节点时：正常做 masked softmax 注意力池化
        - 若某个样本在该维度所有 mask=False（全无节点）：直接输出 0 向量，避免 NaN
        """
        B, N, D = x.shape

        # 确保 mask 在同一设备上，且是 bool
        mask = mask.to(x.device).bool()        
        mask_expanded = mask.unsqueeze(-1)    

        # 无效位置置零，仅用于特征（不影响打分维度）
        x_masked = x * mask_expanded         

        # 原始打分
        scores = self.attention_net(x_masked) 

        # 将无效位置打分设为 -inf，用于 softmax 掩码
        scores = scores.masked_fill(~mask_expanded, float("-inf")) 

        # 检查每个样本是否至少有一个 True
        # valid_any[i] = True 表示第 i 个样本至少有一个有效位置
        valid_any = mask.any(dim=1, keepdim=True)  # [B, 1]

        # 对于完全没有有效位置的样本，将 scores 整行设为 0，
        # softmax(0,0,...,0) = 均匀分布，之后再乘以 mask=0，结果仍为 0，不会出 NaN
        scores = torch.where(
            valid_any.unsqueeze(-1),   # [B, 1, 1]
            scores,
            torch.zeros_like(scores)   # 对于全 False 的样本，scores 全 0
        )

        # 计算注意力权重
        attn = torch.softmax(scores, dim=1)         # [B, N, 1]

        # 再次应用 mask，防止极端数值（对全 False 样本，此时 attn 本来是均匀分布）
        attn = attn * mask_expanded.float()         # [B, N, 1]

        # 沿 N 维重新归一化，让有效位置的权重和为 1；若没有有效位置则保持 0
        denom = attn.sum(dim=1, keepdim=True)       # [B, 1, 1]
        # 避免 0 除，只有 denom>0 的样本才归一化
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
        self._external_tokenizer = None  # 供 set_tokenizer 使用

        # ===== 0. 时序图 / 文本的注意力池化 (可训练的 u, w) =====
        struct_node_dim = config.structure_dim // 2  # 128
        self.struct_att_pool = LearnableAttentionPooling(
            input_dim=struct_node_dim,
            hidden_dim=config.attention_hidden_dim,
        )
        self.text_att_pool = LearnableAttentionPooling(
            input_dim=config.text_dim,
            hidden_dim=config.attention_hidden_dim,
        )

        # 1. 图文对齐投影层（W_proj）
        self.text_to_struct_proj = nn.Linear(
            config.text_dim, config.structure_dim, bias=False
        )

        # 2. 时序位置编码（P_t）
        self.temporal_position_embedding = nn.Embedding(
            config.max_sequence_length, 2 * config.structure_dim  # 512
        )

        self.fusion_norm = nn.LayerNorm(2 * config.structure_dim)

        # Frozen CEE dynamics head: mu_theta(x_t) -> x_{t+1}.
        # It must be pretrained by pretrain_cee_head.py before main training.
        self.cee_state_dim = 2 * config.structure_dim
        self.cee_head = CEEDynamicsHead(
            input_dim=self.cee_state_dim,
            hidden_dim=getattr(config, "cee_hidden_dim", self.cee_state_dim),
        )
        self._load_and_freeze_cee_head()

        # 3. 向量到LLM输入的映射层（W_e）
        self.llm_projector = nn.Linear(
            2 * config.structure_dim, config.llm_input_dim, bias=False
        )

        # 4. LLM模型 + LoRA
        print("加载LLM模型...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_path,
            torch_dtype=torch.bfloat16,
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
        self.llm.config.pad_token_id = self.llm.config.eos_token_id

        # 冻结非 LoRA 参数
        for name, param in self.llm.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        self._count_trainable_params()
        self.llm_dtype = torch.bfloat16

        # ===== 对齐损失相关（edge-drop view + label-free memory bank） =====
        self.struct_timestep_dropout = getattr(self.config, "struct_timestep_dropout", 0.0)
        self.align_queue_size = getattr(self.config, "align_queue_size", 512)
        self.align_max_negatives = getattr(self.config, "align_max_negatives", 64)

        feat_dim = 2 * self.config.structure_dim  # 512

        self.register_buffer("align_queue", torch.zeros(self.align_queue_size, feat_dim, dtype=torch.float32))
        self.register_buffer("align_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("align_queue_size_used", torch.zeros(1, dtype=torch.long))

        # 前缀缓存
        self._time_prefix_cache = {}
        self._summary_prefix_embedding = None

        self.to(config.device)

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
        self.llm_tokenizer = tokenizer  # 兼容 train.py

        # ===== 调试：看一下 '0' 和 '1' 的编码结果 =====
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

    # ===================== 节点/时间级池化 =====================

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

    # ===================== 节点级全局时序表征 & edge-drop 对齐 =====================

    def _build_global_sequence_repr(
        self,
        structure_embeddings: torch.Tensor,  # [B, T, 256]
        text_embeddings: torch.Tensor,       # [B, T, 384]
        attention_mask: torch.Tensor,        # [B, T]
    ) -> torch.Tensor:
        text_aligned = self.text_to_struct_proj(text_embeddings)  # [B, T, 256]
        fused = torch.cat([structure_embeddings, text_aligned], dim=-1)  # [B, T, 512]

        time_mask = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
        fused_masked = fused * time_mask                 # [B, T, 512]

        lengths = time_mask.sum(dim=1)                   # [B, 1]
        lengths = lengths.clamp(min=1)
        h = fused_masked.sum(dim=1) / lengths            # [B, 512]
        return h

    def _update_alignment_queue(self, h1: torch.Tensor) -> None:
        with torch.no_grad():
            feats = h1.detach().to(self.align_queue.device)
            if feats.numel() == 0:
                return

            K = int(self.align_queue_size)
            if K <= 0:
                return

            num = feats.size(0)
            if num >= K:
                feats = feats[-K:]
                num = K

            ptr = int(self.align_queue_ptr.item())
            used = int(self.align_queue_size_used.item())
            end = ptr + num

            if end <= K:
                self.align_queue[ptr:end] = feats
            else:
                first_len = K - ptr
                self.align_queue[ptr:] = feats[:first_len]
                self.align_queue[: end - K] = feats[first_len:]

            self.align_queue_ptr[0] = (ptr + num) % K
            self.align_queue_size_used[0] = min(K, used + num)

    # ===================== 对齐损失（edge-drop + label-free queue） =====================

    def compute_alignment_loss(
        self,
        structure_embeddings: torch.Tensor,  # [B, T, 256]
        structure_embeddings_aug: Optional[torch.Tensor],  # [B, T, 256]
        text_embeddings: torch.Tensor,       # [B, T, 384]
        attention_mask: torch.Tensor,        # [B, T]
        temperature: float = 0.07,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        device = structure_embeddings.device

        if structure_embeddings_aug is None:
            if self.training and bool(getattr(self.config, "require_edge_drop_alignment", True)):
                raise ValueError("L_cont requires edge-drop GAT embeddings, but augmented structure view is missing")
            return torch.tensor(0.0, device=device)

        # 1) 计算两种视角的全局表示 h1, h2
        h1 = self._build_global_sequence_repr(
            structure_embeddings, text_embeddings, attention_mask
        )  # [B, 512]

        h2 = self._build_global_sequence_repr(
            structure_embeddings_aug, text_embeddings, attention_mask
        )  # [B, 512]

        # 2) L2 归一化
        h1_norm = F.normalize(h1, p=2, dim=-1, eps=epsilon)
        h2_norm = F.normalize(h2, p=2, dim=-1, eps=epsilon)

        batch_size = h1_norm.size(0)

        logits = torch.matmul(h1_norm, h2_norm.t()) / temperature
        targets = torch.arange(batch_size, device=device, dtype=torch.long)

        queue_used = int(self.align_queue_size_used.item())
        if queue_used > 0:
            queue = self.align_queue[:queue_used]
            queue = F.normalize(queue, p=2, dim=-1, eps=epsilon)
            if queue.size(0) > int(self.align_max_negatives):
                idx = torch.randperm(queue.size(0), device=device)[: int(self.align_max_negatives)]
                queue = queue[idx]
            queue_logits = torch.matmul(h1_norm, queue.t()) / temperature
            logits = torch.cat([logits, queue_logits], dim=-1)

        loss = F.cross_entropy(logits, targets, reduction="mean")
        self._update_alignment_queue(h1)
        return loss
        
    # ===================== CEE 计算 =====================
    def _compute_cee_for_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        用冻结的 mu_theta 对单个账号状态序列计算报告/正则用 CEE score。

        参数:
            states: [T, d]，这里 d = 2 * structure_dim (结构 + 文本投影拼接)

        返回:
            scalar。默认 cee_score_mode=log_likelihood，值越高表示轨迹越可预测、越像真实用户模式。
        """
        device = states.device
        T = states.size(0)
        if T < 2:
            return torch.tensor(0.0, device=device)
        nll = self.cee_head.compute_cee(
            states,
            sigma=getattr(self.config, "cee_sigma", 1.0),
            include_constant=bool(getattr(self.config, "cee_include_constant", True)),
        )
        return transform_cee_nll_to_score(nll, getattr(self.config, "cee_score_mode", "log_likelihood"))

    def compute_cee_loss(
        self,
        structure_embeddings: torch.Tensor,   # [B, T, 256]
        text_embeddings: torch.Tensor,        # [B, T, 384]
        attention_mask: torch.Tensor,         # [B, T]  时间步有效 mask
        labels: torch.Tensor,                 # [B]，0=normal,1=CIB
    ) -> torch.Tensor:
        """
        Pairwise L_CEE:
            默认使用 CEE score = log p(x_{t+1}|x_t)。
            E_{v+,v-} max(0, score(v+) - score(v-) + delta_d)

        v+ 是 CIB / malicious，v- 是 normal；目标方向是 normal 的 log-likelihood 更高。
        """
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

        # Step 1: 逐账号计算 CEE_i
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
            return torch.tensor(0.0, device=device)

        cee_values = torch.stack(cee_values)       # [B]

        # Step 2: 拆分 CIB / authentic
        is_cib = (labels == 1)
        is_auth = (labels == 0)

        if not (is_cib.any() and is_auth.any()):
            # 本 batch 没有同时包含两类，CEE 正则不生效
            return cee_values.sum() * 0.0

        cib_cee = cee_values[is_cib]               # [N_cib]
        auth_cee = cee_values[is_auth]             # [N_auth]

        margin = getattr(self.config, "cee_margin", 0.1)

        score_mode = str(getattr(self.config, "cee_score_mode", "log_likelihood")).lower()
        if score_mode in {"nll", "negative_log_likelihood", "raw_nll"}:
            # raw NLL 越低越可预测；正常用户应低于 CIB。
            l_cee = torch.relu(auth_cee[None, :] - cib_cee[:, None] + margin).mean()
        else:
            # log-likelihood 越高越可预测；正常用户应高于 CIB。
            l_cee = torch.relu(cib_cee[:, None] - auth_cee[None, :] + margin).mean()

        return l_cee


    # ===================== soft token 构造 =====================
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
            fused_normed = self.fusion_norm(fused_with_pos)

            if debug_mode == "normal":
                fused_for_llm = fused_normed
            elif debug_mode == "shuffle":
                perm = torch.randperm(valid_len, device=device)
                fused_for_llm = fused_normed[perm]
            elif debug_mode == "zero":
                fused_for_llm = torch.zeros_like(fused_normed)
            else:
                fused_for_llm = fused_normed

            llm_tokens = self.llm_projector(fused_for_llm)  # [L, D_llm]
            llm_token_sequences.append(llm_tokens)

        return llm_token_sequences, None, None

    # ===================== prompt 构造 =====================

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

        """ question_prefix = (
            "请综合上述时间序列特征判断该用户是否为恶意账号（参与协调不良行为），"
            "只回答 'Benign' 或 'Malicious'。"
        ) """


        full_prefix = (
            "[INST]<<SYS>>" + system_prompt + "<</SYS>>\n" +
            behavior_prefix + "\n" +
            question_prefix + "[/INST]"
        )

        with torch.no_grad():
            input_ids = tokenizer(full_prefix, return_tensors="pt").input_ids.to(device)
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

    # ===================== forward =====================
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
    ) -> Dict[str, torch.Tensor]:
        """
        注意：这里的 attention_mask 是时间步级的 [B, T]（train.py 里传进来的 time_mask）
        """
        device = self.config.device
        tokenizer = self._get_tokenizer(tokenizer)

        # --- 0. 损失权重（用于硬消融门控：lambda==0 时不计算、不更新对应分支） ---
        lambda_gen = getattr(self.config, "lambda_gen", 0.5)
        lambda_align = getattr(self.config, "lambda_align", 0.5)
        lambda_cee = getattr(self.config, "lambda_cee", 0.1)


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
            attention_mask,   # 这里传的是时间步 mask
        )  # [B, T, 256], [B, T, 384], [B, T]

        structure_embeddings_aug = None
        has_aug_view = all(
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

        # --- 3. 对齐损失（edge-drop InfoNCE + label-free memory bank） ---
        # 硬消融：当 lambda_align == 0 时，完全跳过计算与队列更新（不调用 compute_alignment_loss）。
        if (lambda_align is not None) and float(lambda_align) > 0.0:
            align_loss = self.compute_alignment_loss(
                structure_embeddings,
                structure_embeddings_aug,
                text_embeddings,
                attention_mask,
                temperature=getattr(self.config, "temperature", 0.07),
            )
        else:
            align_loss = torch.tensor(0.0, device=device)


        # --- 3.5 CEE 正则（仅在有标签且启用时计算） ---
        # 硬消融：当 lambda_cee == 0 时，完全跳过计算（不调用 compute_cee_loss）。
        if (labels is not None) and ((lambda_cee is not None) and float(lambda_cee) > 0.0):
            cee_loss = self.compute_cee_loss(
                structure_embeddings,
                text_embeddings,
                attention_mask,
                labels,
            )
        else:
            cee_loss = torch.tensor(0.0, device=device)

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

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # [B, L, vocab]

        # 无标签模式：仅输出 logits（loss 仅用于调试；启用的分支由 lambda_* 控制）
        if labels is None:
            return {
                "logits": logits,
                "align_loss": align_loss.detach(),
                "gen_loss": torch.tensor(0.0, device=device),
                "cee_loss": cee_loss.detach(),
                "total_loss": (lambda_align * align_loss + lambda_cee * cee_loss).detach(),
            }

        # --- 6. 生成损失（Benign / Malicious 二分类） ---
        labels = labels.to(device)

        # 使用 '0' / '1' 作为分类 token，更适配各种 tokenizer
        benign_token_id = tokenizer(
            "0", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0][-1].item()
        malicious_token_id = tokenizer(
            "1", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0][-1].item()

        # 1) 取最后一个位置的原始 logits（未 softmax）: [B, vocab]
        last_logits = logits[:, -1, :]

        # 2) 直接从 logits 里抽 "0" 和 "1" 两个位置的 logit 值（不是概率）: [B]
        benign_logits = last_logits[:, benign_token_id]
        malicious_logits = last_logits[:, malicious_token_id]

        # 3) 堆叠成 [B, 2]，作为二分类 logits
        logits_2 = torch.stack([benign_logits, malicious_logits], dim=-1).float()  # [B, 2]

        # 4) PU pairwise ranking 分类损失。CE 只保留为日志指标，不进入 total_loss。
        score = logits_2[:, 1] - logits_2[:, 0]
        pos_score = score[labels == 1]
        neg_score = score[labels == 0]
        if pos_score.numel() > 0 and neg_score.numel() > 0:
            margin = float(getattr(self.config, "cls_margin", 0.3))
            gen_loss = torch.relu(margin - (pos_score[:, None] - neg_score[None, :])).mean()
        else:
            gen_loss = score.sum() * 0.0

        ce_loss = F.cross_entropy(logits_2, labels.long(), reduction="mean")

        # --- 7. 总损失 = 生成损失 + 对齐损失 + CEE 正则（权重可调） ---

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
            "total_loss": total_loss,
        }

    # ===================== 保存 / 加载 =====================

    def save_model(self, path: str):
        torch.save(
            {"model_state_dict": self.state_dict(), "config": self.config},
            path,
        )
        print(f"模型保存到: {path}")

    def load_model(self, path: str):
        """
        安全加载模型：
        1) checkpoint 先加载到 CPU，避免 GPU 上临时展开整份 checkpoint 导致显存峰值过高；
        2) 再把 state_dict 拷贝进当前模型；
        3) 最后确保模型位于目标设备。
        """
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        skipped_cee_keys = [key for key in state_dict if key.startswith("cee_head.")]
        if skipped_cee_keys:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("cee_head.")
            }
            print(f"[load_model] skipped checkpoint cee_head.* keys: {len(skipped_cee_keys)}")

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

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
