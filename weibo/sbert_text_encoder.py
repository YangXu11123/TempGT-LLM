import os
import re
import glob
import pickle
import warnings
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

# 消融模型
MODEL_REGISTRY: Dict[str, str] = {
    "bert_microsoft_Multilingual-MiniLM-L12-H384":
        "/home/llm/xy_tweets_data/ablation_exe/models/bert_microsoft_Multilingual-MiniLM-L12-H384",
    "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2":
        "/home/llm/xy_tweets_data/ablation_exe/models/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2",
    "xlm-roberta-large":
        "/home/llm/xy_tweets_data/ablation_exe/models/xlm-roberta-large",
    "microsoft_deberta-v3-large":
        "/home/llm/xy_tweets_data/ablation_exe/models/microsoft_deberta-v3-large",
}

DEFAULT_ENCODER_KEY = "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"

# 下游固定维度
TARGET_TEXT_DIM = 384

PROJ_SEED = 0
HF_MAX_LENGTH = 256
HF_POOLING = "mean"   


def _normalize_device(device: Any) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _is_sentence_transformer_dir(model_dir: str) -> bool:
    if not model_dir or not os.path.exists(model_dir):
        return False
    return os.path.exists(os.path.join(model_dir, "modules.json"))


class ExperimentalSBERTTextEncoder(nn.Module):
    def __init__(self,
                 encoder_key: str = DEFAULT_ENCODER_KEY,
                 device: object = 'auto',
                 freeze_sbert: bool = True,
                 target_dim: int = TARGET_TEXT_DIM,
                 proj_seed: int = PROJ_SEED,
                 proj_cache_dir: Optional[str] = None,
                 hf_max_length: int = HF_MAX_LENGTH,
                 hf_pooling: str = HF_POOLING):
        super().__init__()

        self.device = _normalize_device(device)

        self.encoder_key = str(encoder_key)
        self.target_dim = int(target_dim)
        self.proj_seed = int(proj_seed)
        self.proj_cache_dir = proj_cache_dir

        self.hf_max_length = int(hf_max_length)
        self.hf_pooling = str(hf_pooling).lower().strip()
        if self.hf_pooling not in ["mean"]:
            raise ValueError(f"不支持的 hf_pooling='{hf_pooling}'，当前仅支持 'mean'。")

        # 解析本地模型路径
        if self.encoder_key not in MODEL_REGISTRY:
            raise RuntimeError(
                f"[TextEncoder] encoder_key '{self.encoder_key}' 不在 MODEL_REGISTRY 中。\n"
                f"可用 keys: {list(MODEL_REGISTRY.keys())}"
            )

        candidate = MODEL_REGISTRY[self.encoder_key]
        if not os.path.exists(candidate):
            raise RuntimeError(f"[TextEncoder] encoder_key '{self.encoder_key}' 对应路径不存在: {candidate}")

        self.resolved_model_path = candidate

        self.encoder_backend = (
            "sentence_transformers"
            if _is_sentence_transformer_dir(self.resolved_model_path)
            else "hf_auto_model"
        )

        # 加载模型
        self.st_model: Optional[SentenceTransformer] = None
        self.hf_tokenizer: Optional[AutoTokenizer] = None
        self.hf_model: Optional[AutoModel] = None

        if self.encoder_backend == "sentence_transformers":
            self.st_model = SentenceTransformer(self.resolved_model_path, device=self.device)
            self.original_dim = int(self.st_model.get_sentence_embedding_dimension())
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.resolved_model_path, use_fast=False)
            self.hf_model = AutoModel.from_pretrained(self.resolved_model_path)
            self.hf_model.to(self.device)
            self.original_dim = int(getattr(self.hf_model.config, "hidden_size"))

        self.embedding_dim = int(self.target_dim)

        # 冻结编码器参数
        if freeze_sbert:
            if self.encoder_backend == "sentence_transformers":
                for p in self.st_model.parameters():
                    p.requires_grad = False
            else:
                for p in self.hf_model.parameters():
                    p.requires_grad = False

        self.to(self.device)

        self._proj_matrix = None
        self._proj_loaded_for_dim = None

    def _get_proj_cache_path(self, in_dim: int) -> str:
        if self.proj_cache_dir is None or len(self.proj_cache_dir) == 0:
            raise RuntimeError("proj_cache_dir 未设置，应在 batch_encode_node_texts_experimental() 中指定。")
        os.makedirs(self.proj_cache_dir, exist_ok=True)
        fname = f"proj_gaussian_D{in_dim}_to_{self.target_dim}_seed{self.proj_seed}.pt"
        return os.path.join(self.proj_cache_dir, fname)

    def _load_or_create_proj(self, in_dim: int) -> torch.Tensor:
        if self._proj_matrix is not None and self._proj_loaded_for_dim == in_dim:
            return self._proj_matrix

        cache_path = self._get_proj_cache_path(in_dim)

        if os.path.exists(cache_path):
            P = torch.load(cache_path, map_location='cpu')
        else:
            g = torch.Generator(device='cpu')
            g.manual_seed(self.proj_seed)
            P = torch.randn(in_dim, self.target_dim, generator=g, dtype=torch.float32)
            P = P / (in_dim ** 0.5)
            torch.save(P, cache_path)

        P = P.to(self.device)
        P.requires_grad_(False)

        self._proj_matrix = P
        self._proj_loaded_for_dim = in_dim
        return P

    def _project_to_target_dim(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings is None or embeddings.numel() == 0:
            return torch.empty((0, self.target_dim), device=self.device, dtype=torch.float)

        in_dim = int(embeddings.shape[-1])
        if in_dim == self.target_dim:
            return embeddings

        P = self._load_or_create_proj(in_dim)
        return embeddings @ P

    def _hf_encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        assert self.hf_tokenizer is not None and self.hf_model is not None

        if not texts:
            return torch.empty((0, self.original_dim), device=self.device, dtype=torch.float)

        all_embeds = []
        n = len(texts)

        for i in range(0, n, batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = self.hf_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.hf_max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.hf_model(**enc)
                token_emb = out.last_hidden_state
                attn_mask = enc.get("attention_mask", None)

                if self.hf_pooling == "mean":
                    if attn_mask is None:
                        sent_emb = token_emb.mean(dim=1)
                    else:
                        mask = attn_mask.unsqueeze(-1).type_as(token_emb)
                        summed = (token_emb * mask).sum(dim=1)
                        denom = mask.sum(dim=1).clamp(min=1e-6)
                        sent_emb = summed / denom
                else:
                    sent_emb = token_emb.mean(dim=1)

                all_embeds.append(sent_emb.detach())

        return torch.cat(all_embeds, dim=0).to(self.device)

    def encode_texts_with_sbert(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.target_dim), device=self.device, dtype=torch.float)

        valid_texts = []
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                cleaned = text.strip()
                if cleaned:
                    valid_texts.append(cleaned)

        if not valid_texts:
            return torch.empty((0, self.target_dim), device=self.device, dtype=torch.float)

        try:
            if self.encoder_backend == "sentence_transformers":
                with torch.no_grad():
                    emb = self.st_model.encode(
                        valid_texts,
                        convert_to_tensor=True,
                        device=self.device,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=False
                    )
                    emb = emb.clone().detach().to(self.device)
            else:
                emb = self._hf_encode(valid_texts, batch_size=batch_size)

            emb = self._project_to_target_dim(emb)
            return emb

        except Exception:
            return torch.empty((0, self.target_dim), device=self.device, dtype=torch.float)

    def encode_timestep_texts_experimental(self,
                                          timestep_texts: List[str],
                                          batch_size: int = 64,
                                          training: bool = False) -> torch.Tensor:
        if not timestep_texts:
            return torch.empty((0, self.target_dim), device=self.device, dtype=torch.float)
        return self.encode_texts_with_sbert(timestep_texts, batch_size=batch_size)

    def encode_node_temporal_texts_experimental(self,
                                                node_texts: Dict[int, List[str]],
                                                all_timesteps: List[int],
                                                batch_size: int = 64,
                                                training: bool = False) -> Dict[int, torch.Tensor]:
        temporal_text_embeddings: Dict[int, torch.Tensor] = {}
        for time_step in all_timesteps:
            if time_step in node_texts and node_texts[time_step]:
                timestep_embeddings = self.encode_timestep_texts_experimental(
                    node_texts[time_step],
                    batch_size=batch_size,
                    training=training
                )
            else:
                timestep_embeddings = torch.empty((0, self.target_dim), device=self.device, dtype=torch.float)

            temporal_text_embeddings[time_step] = timestep_embeddings
        return temporal_text_embeddings


def load_subgraph_data(subgraph_file: str) -> Dict:
    try:
        with open(subgraph_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def extract_text_content(item) -> Optional[str]:
    if isinstance(item, str):
        t = item.strip()
        return t if t else None

    if isinstance(item, dict):
        if 'text' in item:
            t = str(item['text']).strip()
            if t:
                return t

        for key in ['content', 'message', 'comment', 'post', 'body']:
            if key in item and item[key]:
                t = str(item[key]).strip()
                if t:
                    return t

        text_parts = []
        skip_keys = {'timestamp', 'type', 'id', 'user_id', 'node_id'}
        for k, v in item.items():
            if k not in skip_keys and v is not None:
                s = str(v).strip()
                if s and len(s) > 1:
                    text_parts.append(s)

        return ' '.join(text_parts) if text_parts else None

    if item is not None:
        t = str(item).strip()
        return t if t else None

    return None


def extract_node_texts_from_subgraph(subgraph_data: Dict) -> Dict[int, List[str]]:
    if 'node_texts' not in subgraph_data:
        return {}

    node_texts_raw = subgraph_data['node_texts']
    temporal_texts: Dict[int, List[str]] = {}

    for time_step, text_data in node_texts_raw.items():
        cleaned_texts = []
        if isinstance(text_data, list):
            for item in text_data:
                tc = extract_text_content(item)
                if tc:
                    cleaned_texts.append(tc)
        else:
            tc = extract_text_content(text_data)
            if tc:
                cleaned_texts.append(tc)

        if cleaned_texts:
            temporal_texts[int(time_step)] = cleaned_texts

    return temporal_texts


def find_subgraph_files(input_dir: str = "subgraphs") -> List[str]:
    if not os.path.exists(input_dir):
        return []
    pattern = os.path.join(input_dir, "subgraph_*_k2.pkl")
    files = glob.glob(pattern)
    files.sort()
    return files


def _as_int_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []


def get_all_timesteps_from_subgraphs(subgraph_dir: str) -> List[int]:
    all_timesteps = set()
    subgraph_files = find_subgraph_files(subgraph_dir)

    for subgraph_file in subgraph_files[:10]:
        try:
            subgraph_data = load_subgraph_data(subgraph_file)
            if 'time_steps' in subgraph_data:
                all_timesteps.update(_as_int_list(subgraph_data['time_steps']))
            elif 'node_texts' in subgraph_data:
                all_timesteps.update(int(k) for k in subgraph_data['node_texts'].keys())
        except Exception:
            continue

    return sorted(list(all_timesteps))


def extract_node_id_from_filename(filename: str) -> str:
    match = re.search(r'subgraph_(.+?)_k2\.pkl', os.path.basename(filename))
    return match.group(1) if match else "unknown"


def process_node_text_encoding_experimental(subgraph_data: Dict,
                                            encoder: ExperimentalSBERTTextEncoder,
                                            all_timesteps: List[int],
                                            batch_size: int = 64) -> Dict:
    center_node = subgraph_data.get('center_node', 'unknown')
    node_texts = extract_node_texts_from_subgraph(subgraph_data)

    text_counts_per_timestep = {t: 0 for t in all_timesteps}

    if not node_texts:
        temporal_text_embeddings: Dict[int, torch.Tensor] = {}
        total_texts = 0
        valid_timesteps = 0
    else:
        temporal_text_embeddings = encoder.encode_node_temporal_texts_experimental(
            node_texts, all_timesteps, batch_size, training=False
        )
        for t in list(temporal_text_embeddings.keys()):
            temporal_text_embeddings[t] = temporal_text_embeddings[t].detach().cpu()

        total_texts = 0
        valid_timesteps = 0
        for timestep, texts in node_texts.items():
            num_texts = len(texts)
            if timestep in text_counts_per_timestep:
                text_counts_per_timestep[timestep] = num_texts
            total_texts += num_texts
            if num_texts > 0:
                valid_timesteps += 1

    projection_used = (encoder.original_dim != encoder.target_dim)
    proj_cache_file = None
    if projection_used and encoder.proj_cache_dir is not None:
        proj_cache_file = os.path.join(
            encoder.proj_cache_dir,
            f"proj_gaussian_D{encoder.original_dim}_to_{encoder.target_dim}_seed{encoder.proj_seed}.pt"
        )

    result = {
        'center_node': center_node,
        'temporal_text_embeddings': temporal_text_embeddings,
        'timesteps': torch.tensor(all_timesteps, dtype=torch.long),
        'embedding_dim': encoder.embedding_dim,  # 固定为 target_dim
        'num_timesteps': len(all_timesteps),
        'total_texts': total_texts,
        'valid_timesteps': valid_timesteps,
        'text_counts_per_timestep': text_counts_per_timestep,
        'encoding_info': {
            'encoder_key': encoder.encoder_key,
            'resolved_model_path': encoder.resolved_model_path,
            'encoder_backend': encoder.encoder_backend,
            'original_dim': encoder.original_dim,
            'final_dim': encoder.target_dim,
            'projection_used': projection_used,
            'projection_type': 'fixed_gaussian' if projection_used else None,
            'projection_seed': encoder.proj_seed if projection_used else None,
            'projection_cache_file': proj_cache_file if projection_used else None,
            'frozen_sbert': True,
            'trainable_attention_in_this_file': False,
            'need_downstream_attention_pooling': True,
            'experimental_design': True,
            'hf_pooling': encoder.hf_pooling if encoder.encoder_backend == "hf_auto_model" else None,
            'hf_max_length': encoder.hf_max_length if encoder.encoder_backend == "hf_auto_model" else None,
        }
    }
    return result


def batch_encode_node_texts_experimental(input_dir: str = "subgraphs",
                                         output_dir: str = "sbert_encoded_texts",
                                         config: Dict = None) -> Dict:
    if config is None:
        config = {
            'encoder_key': DEFAULT_ENCODER_KEY,
            'device': 'auto',
            'freeze_sbert': True,
            'batch_size': 64,
            'target_dim': TARGET_TEXT_DIM,
            'proj_seed': PROJ_SEED,
            'hf_max_length': HF_MAX_LENGTH,
            'hf_pooling': HF_POOLING,
        }

    os.makedirs(output_dir, exist_ok=True)
    subgraph_files = find_subgraph_files(input_dir)
    if not subgraph_files:
        return {}

    proj_cache_dir = os.path.join(output_dir, "proj_cache")

    encoder_config = {k: v for k, v in config.items() if k not in ['batch_size']}
    encoder_config['proj_cache_dir'] = proj_cache_dir

    encoder = ExperimentalSBERTTextEncoder(**encoder_config)

    all_timesteps = get_all_timesteps_from_subgraphs(input_dir)
    if not all_timesteps:
        return {}

    stats = {
        'total_files': len(subgraph_files),
        'successful': 0,
        'failed': 0,
        'failed_files': [],
        'output_files': [],
        'total_texts_processed': 0,
        'total_size_mb': 0.0,
        'experimental_design': True,
        'encoder_key': encoder.encoder_key,
        'resolved_model_path': encoder.resolved_model_path,
        'encoder_backend': encoder.encoder_backend,
        'original_dim': encoder.original_dim,
        'final_dim': encoder.target_dim,
        'projection_used': (encoder.original_dim != encoder.target_dim),
        'text_counts_per_timestep': {t: 0 for t in all_timesteps}
    }

    print("\n[TextEncoder] 当前配置：")
    print(f"  encoder_key    : {encoder.encoder_key}")
    print(f"  模型路径       : {encoder.resolved_model_path}")
    print(f"  后端           : {encoder.encoder_backend}")
    print(f"  原始维度       : {encoder.original_dim}")
    print(f"  最终维度       : {encoder.target_dim}")
    if encoder.original_dim != encoder.target_dim:
        print(f"  固定投影       : 高斯投影（seed={encoder.proj_seed}）")
        print(f"  投影缓存目录   : {proj_cache_dir}")
    else:
        print("  固定投影       : 不启用（原始维度已为 target_dim）")
    if encoder.encoder_backend == "hf_auto_model":
        print(f"  HF pooling     : {encoder.hf_pooling}")
        print(f"  HF max_length  : {encoder.hf_max_length}")

    for subgraph_file in tqdm(subgraph_files, desc="文本编码"):
        try:
            node_id = extract_node_id_from_filename(subgraph_file)
            subgraph_data = load_subgraph_data(subgraph_file)
            if not subgraph_data:
                stats['failed'] += 1
                stats['failed_files'].append(subgraph_file)
                continue

            result = process_node_text_encoding_experimental(
                subgraph_data, encoder, all_timesteps, config.get('batch_size', 64)
            )

            for timestep, count in result['text_counts_per_timestep'].items():
                stats['text_counts_per_timestep'][timestep] += count

            output_filename = f"subgraph_{node_id}_k2_text_encoded.pkl"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'wb') as f:
                pickle.dump(result, f)

            stats['successful'] += 1
            stats['output_files'].append(output_path)
            stats['total_texts_processed'] += result['total_texts']
            stats['total_size_mb'] += os.path.getsize(output_path) / 1024 / 1024

        except Exception:
            stats['failed'] += 1
            stats['failed_files'].append(subgraph_file)

        # 定期清理显存
        if (stats['successful'] + stats['failed']) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats_file = os.path.join(output_dir, "experimental_text_encoding_stats.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)

    return stats


def main():
    print("=" * 60)
    print("文本编码器（SentenceTransformer + HF AutoModel）- 消融 + 固定 384 投影")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 在这里切换 encoder_key
    config = {
        'encoder_key': "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2",
        # 可用 keys：
        # "bert_microsoft_Multilingual-MiniLM-L12-H384"
        # "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"
        # "xlm-roberta-large"
        # "microsoft_deberta-v3-large"

        'device': device,
        'freeze_sbert': True,
        'batch_size': 64,
        'target_dim': TARGET_TEXT_DIM,
        'proj_seed': PROJ_SEED,
        'hf_max_length': HF_MAX_LENGTH,
        'hf_pooling': HF_POOLING,
    }

    stats = batch_encode_node_texts_experimental(
        input_dir="subgraphs",
        output_dir="sbert_encoded_texts",
        config=config
    )

    if not stats:
        print("批量编码失败")
        return

    print("\n编码统计：")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功编码: {stats['successful']}")
    print(f"编码失败: {stats['failed']}")
    print(f"总文本数: {stats['total_texts_processed']}")
    print(f"后端: {stats['encoder_backend']}")
    print(f"原始维度: {stats['original_dim']}")
    print(f"最终维度: {stats['final_dim']}")
    print(f"是否投影: {stats['projection_used']}")
    print(f"输出大小: {stats['total_size_mb']:.2f} MB")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()