import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, Tuple, List, Optional


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

class CEEHead(nn.Module):
    """
    CEE Head: 输入 [s_t; s_{t+1}] ∈ R^{2d}，输出 P(s_{t+1} | s_t) ∈ (0, 1)
    不经过 LLM，仅接在编码器输出之后。
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),  
        )

    def forward(self, s_t: torch.Tensor, s_tp1: torch.Tensor) -> torch.Tensor:
        """
        s_t, s_tp1: [..., d]
        返回: [...], 每个位置是 P(s_{t+1} | s_t)
        """
        # 保证维度对齐
        if s_t.dim() == 1:
            s_t = s_t.unsqueeze(0)
        if s_tp1.dim() == 1:
            s_tp1 = s_tp1.unsqueeze(0)

        x = torch.cat([s_t, s_tp1], dim=-1)  # [..., 2d]
        prob = self.mlp(x).squeeze(-1)      # [...]
        return prob


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

        self.fusion_norm = nn.LayerNorm(2 * config.structure_dim)

        # CEE Head：输入维度 = 结构 256 + 文本投影 256 = 512
        self.cee_head = CEEHead(input_dim=2 * config.structure_dim)

        # 向量到LLM输入的映射层（W_e）
        self.llm_projector = nn.Linear(
            2 * config.structure_dim, config.llm_input_dim, bias=False
        )

        # LLM模型 + LoRA
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

        # 对齐损失相关（结构dropout + memory bank） 
        self.struct_timestep_dropout = getattr(self.config, "struct_timestep_dropout", 0.2)
        self.align_queue_size = getattr(self.config, "align_queue_size", 512)
        self.align_max_negatives = getattr(self.config, "align_max_negatives", 64)

        feat_dim = 2 * self.config.structure_dim  # 512

        self.register_buffer(
            "queue_benign",
            torch.zeros(self.align_queue_size, feat_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "queue_cib",
            torch.zeros(self.align_queue_size, feat_dim, dtype=torch.float32),
        )
        self.register_buffer("queue_benign_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_cib_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_benign_size", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_cib_size", torch.zeros(1, dtype=torch.long))

        # 前缀缓存
        self._time_prefix_cache = {}
        self._summary_prefix_embedding = None

        self.to(config.device)

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

    # 节点级全局时序表征 & 结构增强 

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

    def _augment_structure_for_alignment(
        self,
        structure_embeddings: torch.Tensor,  # [B, T, 256]
        attention_mask: torch.Tensor,        # [B, T]
    ) -> torch.Tensor:
        device = structure_embeddings.device
        B, T, D = structure_embeddings.shape
        p_drop = float(self.struct_timestep_dropout)

        if p_drop <= 0.0:
            return structure_embeddings

        aug = structure_embeddings.clone()
        time_mask = attention_mask.bool()

        for i in range(B):
            valid_idx = torch.nonzero(time_mask[i], as_tuple=False).view(-1)
            L = int(valid_idx.numel())
            if L <= 1:
                continue

            drop_count = int(round(p_drop * L))
            if drop_count <= 0:
                continue
            if drop_count >= L:
                drop_count = L - 1

            perm = torch.randperm(L, device=device)
            drop_idx = valid_idx[perm[:drop_count]]
            aug[i, drop_idx, :] = 0.0

        return aug

    def _update_alignment_queues(
        self,
        h1: torch.Tensor,     # [B, 512]
        labels: torch.Tensor  # [B]
    ) -> None:
        with torch.no_grad():
            device = self.queue_benign.device
            h1 = h1.detach().to(device)
            labels = labels.to(device)

            for cls_value, queue_name, ptr_name, size_name in [
                (0, "queue_benign", "queue_benign_ptr", "queue_benign_size"),
                (1, "queue_cib", "queue_cib_ptr", "queue_cib_size"),
            ]:
                mask = (labels == cls_value)
                if not mask.any():
                    continue

                feats = h1[mask]  # [N_cls, 512]
                if feats.numel() == 0:
                    continue

                K = int(self.align_queue_size)
                queue = getattr(self, queue_name)
                ptr_buf = getattr(self, ptr_name)
                size_buf = getattr(self, size_name)

                ptr = int(ptr_buf.item())
                size = int(size_buf.item())

                num = feats.size(0)
                if num >= K:
                    feats = feats[-K:]
                    num = K

                end = ptr + num
                if end <= K:
                    queue[ptr:end] = feats
                else:
                    first_len = K - ptr
                    queue[ptr:] = feats[:first_len]
                    queue[: end - K] = feats[first_len:]

                ptr = (ptr + num) % K
                size = min(K, size + num)

                ptr_buf[0] = ptr
                size_buf[0] = size

    # 对齐损失（Memory Bank）

    def compute_alignment_loss(
        self,
        structure_embeddings: torch.Tensor,  # [B, T, 256]
        text_embeddings: torch.Tensor,       # [B, T, 384]
        attention_mask: torch.Tensor,        # [B, T]
        labels: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        device = structure_embeddings.device

        if labels is None:
            return torch.tensor(0.0, device=device)

        labels = labels.to(device)

        # 计算两种视角的全局表示 h1, h2
        h1 = self._build_global_sequence_repr(
            structure_embeddings, text_embeddings, attention_mask
        )  # [B, 512]

        struct_aug = self._augment_structure_for_alignment(
            structure_embeddings, attention_mask
        )
        h2 = self._build_global_sequence_repr(
            struct_aug, text_embeddings, attention_mask
        )  # [B, 512]

        # L2归一化
        h1_norm = F.normalize(h1, p=2, dim=-1, eps=epsilon)
        h2_norm = F.normalize(h2, p=2, dim=-1, eps=epsilon)

        batch_size = h1_norm.size(0)

        # 构建统一的负样本池
        neg_list: List[torch.Tensor] = []

        # 来自 benign 队列的历史样本
        q_benign_size = int(self.queue_benign_size.item())
        if q_benign_size > 0:
            q_benign = self.queue_benign[:q_benign_size]
            q_benign_norm = F.normalize(q_benign, p=2, dim=-1, eps=epsilon)
            neg_list.append(q_benign_norm)

        # 来自 cib 队列的历史样本
        q_cib_size = int(self.queue_cib_size.item())
        if q_cib_size > 0:
            q_cib = self.queue_cib[:q_cib_size]
            q_cib_norm = F.normalize(q_cib, p=2, dim=-1, eps=epsilon)
            neg_list.append(q_cib_norm)

        # 当前 batch 中所有样本的 h1 也加入统一池
        if batch_size > 0:
            neg_list.append(h1_norm)

        if len(neg_list) > 0:
            neg_all = torch.cat(neg_list, dim=0)  # [N_all, 512]
            N_total = neg_all.size(0)
            if N_total > int(self.align_max_negatives):
                perm = torch.randperm(N_total, device=device)
                idx = perm[: int(self.align_max_negatives)]
                neg_pool = neg_all[idx]              # [K, 512]
            else:
                neg_pool = neg_all                   # [N_total, 512]
        else:
            neg_pool = None

        # per-sample InfoNCE（统一 neg_pool，不再区分 benign / cib）
        total_loss = 0.0
        valid_count = 0

        if (neg_pool is not None) and (neg_pool.size(0) > 0):
            for i in range(batch_size):
                anchor = h1_norm[i].unsqueeze(0)  # [1, 512]
                pos = h2_norm[i].unsqueeze(0)     # [1, 512]

                # 正样本：同一账号两种视角
                sim_pos = torch.sum(anchor * pos, dim=-1, keepdim=True)  # [1, 1]
                # 负样本：统一 memory bank + 当前 batch 的其他样本
                sim_neg = torch.matmul(anchor, neg_pool.t())             # [1, N_neg]

                logits = torch.cat([sim_pos, sim_neg], dim=-1) / temperature
                target = torch.zeros(1, dtype=torch.long, device=device)  # index 0 为正样本

                loss_i = F.cross_entropy(logits, target, reduction="mean")
                total_loss += loss_i
                valid_count += 1

        # 更新队列
        self._update_alignment_queues(h1, labels)

        if valid_count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / valid_count
        
    # CEE 计算 
    def _compute_cee_for_states(self, states: torch.Tensor) -> torch.Tensor:
        
        device = states.device
        T = states.size(0)
        if T < 2:
            return torch.tensor(0.0, device=device)

        log_probs = []
        for t in range(T - 1):
            s_t = states[t]       # [d]
            s_tp1 = states[t + 1] # [d]
            prob = self.cee_head(s_t, s_tp1)               # scalar in (0,1)
            log_prob = torch.log(prob + 1e-8)              # 避免 log(0)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs)                 # [T-1]
        cee = -torch.mean(log_probs)                       # scalar
        return cee

    def compute_cee_loss(
        self,
        structure_embeddings: torch.Tensor,   # [B, T, 256]
        text_embeddings: torch.Tensor,        # [B, T, 384]
        attention_mask: torch.Tensor,         # [B, T]  时间步有效 mask
        labels: torch.Tensor,                 # [B]，0=normal,1=CIB
    ) -> torch.Tensor:
        
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
            return torch.tensor(0.0, device=device)

        cee_values = torch.stack(cee_values)       # [B]

        # 拆分 CIB / authentic
        is_cib = (labels == 1)
        is_auth = (labels == 0)

        if not (is_cib.any() and is_auth.any()):
            return torch.tensor(0.0, device=device)

        cib_cee = cee_values[is_cib]               # [N_cib]
        auth_cee = cee_values[is_auth]             # [N_auth]

        cib_mean = cib_cee.mean()
        auth_mean = auth_cee.mean()

        margin = getattr(self.config, "cee_margin", 0.1)

        l_cee = torch.clamp(margin + cib_mean - auth_mean, min=0.0)

        return l_cee


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
        labels: Optional[torch.Tensor] = None,
        tokenizer=None,
    ) -> Dict[str, torch.Tensor]:

        device = self.config.device
        tokenizer = self._get_tokenizer(tokenizer)

        # 损失权重
        lambda_gen = getattr(self.config, "lambda_gen", 0.5)
        lambda_align = getattr(self.config, "lambda_align", 0.5)
        lambda_cee = getattr(self.config, "lambda_cee", 0.1)

        # 移到设备 
        struct_in_embeddings = struct_in_embeddings.to(device)
        struct_out_embeddings = struct_out_embeddings.to(device)
        text_node_embeddings = text_node_embeddings.to(device)
        struct_in_mask = struct_in_mask.to(device)
        struct_out_mask = struct_out_mask.to(device)
        text_mask = text_mask.to(device)
        attention_mask = attention_mask.to(device)
        timesteps = timesteps.to(device)

        # 节点级 -> 时间步级 (attention pooling) 
        structure_embeddings, text_embeddings, attention_mask = self._pool_from_node_level(
            struct_in_embeddings,
            struct_out_embeddings,
            text_node_embeddings,
            struct_in_mask,
            struct_out_mask,
            text_mask,
            attention_mask,   
        )  # [B, T, 256], [B, T, 384], [B, T]

        # 对齐损失（InfoNCE + memory bank）
        if (lambda_align is not None) and float(lambda_align) > 0.0:
            align_loss = self.compute_alignment_loss(
                structure_embeddings,
                text_embeddings,
                attention_mask,
                labels=labels,
                temperature=getattr(self.config, "temperature", 0.07),
            )
        else:
            align_loss = torch.tensor(0.0, device=device)


        # CEE 正则
        if (labels is not None) and ((lambda_cee is not None) and float(lambda_cee) > 0.0):
            cee_loss = self.compute_cee_loss(
                structure_embeddings,
                text_embeddings,
                attention_mask,
                labels,
            )
        else:
            cee_loss = torch.tensor(0.0, device=device)

        # 构造 LLM soft tokens 
        llm_token_sequences, _, _ = self._build_sequence_tokens(
            structure_embeddings,
            text_embeddings,
            attention_mask,
            timesteps,
            debug_mode=getattr(self.config, "debug_mode", "normal"),
        )

        # 构造 prompt 输入 LLM 
        inputs_embeds, llm_attention_mask = self.construct_prompts(
            tokenizer, llm_token_sequences, device
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # [B, L, vocab]

        # 无标签模式：仅输出 logits
        if labels is None:
            return {
                "logits": logits,
                "align_loss": align_loss.detach(),
                "gen_loss": torch.tensor(0.0, device=device),
                "cee_loss": cee_loss.detach(),
                "total_loss": (lambda_align * align_loss + lambda_cee * cee_loss).detach(),
            }

        # 生成损失（Benign / Malicious 二分类） 
        labels = labels.to(device)

        # 使用 '0' / '1' 作为分类 token
        benign_token_id = tokenizer(
            "0", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0][-1].item()
        malicious_token_id = tokenizer(
            "1", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0][-1].item()

        last_logits = logits[:, -1, :]

        benign_logits = last_logits[:, benign_token_id]
        malicious_logits = last_logits[:, malicious_token_id]

        logits_2 = torch.stack([benign_logits, malicious_logits], dim=-1).float()  # [B, 2]

        gen_loss = F.cross_entropy(
            logits_2,
            labels.long(),
            reduction="mean",
        )

        # 总损失 = 生成损失 + 对齐损失 + CEE 正则（权重可调）

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
            "cee_loss": cee_loss,
            "total_loss": total_loss,
        }

    # 保存 / 加载 

    def save_model(self, path: str):
        torch.save(
            {"model_state_dict": self.state_dict(), "config": self.config},
            path,
        )
        print(f"模型保存到: {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(
            path,
            map_location=self.config.device,
            weights_only=False,
        )
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"模型加载成功: {path}")