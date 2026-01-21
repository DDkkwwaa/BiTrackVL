import torch
import torch.nn as nn
import os
from typing import List, Dict, Any


try:
    from transformers import RobertaTokenizerFast, RobertaModel
except ImportError as e:
    raise ImportError(
        "需要安装 transformers 库以使用文本编码器，请执行: pip install transformers。"\
        f" 原始错误: {e}"
    )

class SubjectIndexPred(nn.Module):
    """轻量主体词预测器: 两层 MLP + GELU -> Sigmoid。
    不依赖外部项目，仅内部使用。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x).squeeze(-1)
        return torch.sigmoid(x)

class TextEncoderWrapper(nn.Module):
    def __init__(self,
                 model_name: str = 'roberta-base',
                 visual_hidden_dim: int = 768,
                 max_len: int = 256,
                 enable_subject_classifier: bool = True,
                 need_projection: bool = True,
                 dropout: float = 0.1,
                 cache: bool = True):
        super().__init__()
        self.max_len = max_len
        self.visual_hidden_dim = visual_hidden_dim
        # 离线环境支持: 若提供的是本地目录则强制只用本地文件, 避免去 huggingface 下载
        local_files_only = os.path.isdir(model_name)
        try:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, local_files_only=local_files_only)
            self.text_encoder = RobertaModel.from_pretrained(model_name, local_files_only=local_files_only)
        except Exception as e:
            raise RuntimeError(f"加载 RoBERTa 模型/分词器失败: {model_name}. 请确认目录包含 config.json, pytorch_model.bin, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt. 原始错误: {e}")
        self.orig_hidden = self.text_encoder.config.hidden_size
        self.need_projection = need_projection and (self.orig_hidden != visual_hidden_dim)
        self.proj = nn.Identity() if not self.need_projection else nn.Sequential(
            nn.Linear(self.orig_hidden, visual_hidden_dim),
            nn.LayerNorm(visual_hidden_dim),
            nn.Dropout(dropout)
        )
        self.enable_subject_classifier = enable_subject_classifier
        self.subject_pred = SubjectIndexPred(visual_hidden_dim if self.need_projection else self.orig_hidden) if enable_subject_classifier else None
        self.use_cache = cache
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {} if cache else None

    def forward(self, captions: List[str], device=None, fp16: bool = False) -> Dict[str, Any]:
        if device is None:
            device = next(self.parameters()).device
        # 先尝试从缓存命中
        batch_indices = list(range(len(captions)))
        cached_embeddings = {}
        to_encode = []
        to_encode_indices = []
        if self.use_cache:
            for i, cap in enumerate(captions):
                if cap in self._cache:
                    cached_embeddings[i] = self._cache[cap]
                else:
                    to_encode.append(cap)
                    to_encode_indices.append(i)
        else:
            to_encode = captions
            to_encode_indices = batch_indices

        outputs_map = {}
        if to_encode:
            tokenized = self.tokenizer.batch_encode_plus(
                to_encode,
                padding='longest',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.cuda.amp.autocast(enabled=fp16):
                enc = self.text_encoder(**tokenized)
                hidden = enc.last_hidden_state  # [B1,L,Corig]
                hidden = self.proj(hidden)  # [B1,L,Cvis]
            attention_mask = tokenized['attention_mask']  # 1=valid
            pad_mask = attention_mask.eq(0)  # True=padding
            if self.enable_subject_classifier:
                subj_mask_pred = self.subject_pred(hidden)  # [B1,L]
            else:
                subj_mask_pred = torch.ones(hidden.shape[0], hidden.shape[1], device=device)
            # 存入 outputs_map
            for bi_local, bi_global in enumerate(to_encode_indices):
                data_item = {
                    'embeddings': hidden[bi_local],            # [L,C]
                    'pad_mask': pad_mask[bi_local],            # [L]
                    'subject_mask_pred': subj_mask_pred[bi_local]  # [L]
                }
                outputs_map[bi_global] = data_item
                if self.use_cache:
                    self._cache[to_encode[bi_local]] = {
                        'embeddings': data_item['embeddings'].detach().cpu(),
                        'pad_mask': data_item['pad_mask'].detach().cpu(),
                        'subject_mask_pred': data_item['subject_mask_pred'].detach().cpu()
                    }

        # 合并缓存命中的部分
        if cached_embeddings:
            for bi, data_cpu in cached_embeddings.items():
                outputs_map[bi] = {
                    'embeddings': data_cpu['embeddings'].to(device),
                    'pad_mask': data_cpu['pad_mask'].to(device),
                    'subject_mask_pred': data_cpu['subject_mask_pred'].to(device)
                }

        # 重新按原顺序拼接 batch
        max_len_batch = max(item['embeddings'].shape[0] for item in outputs_map.values())
        B = len(captions)
        C = outputs_map[0]['embeddings'].shape[-1]
        emb_batch = torch.zeros(B, max_len_batch, C, device=device)
        pad_mask_batch = torch.ones(B, max_len_batch, dtype=torch.bool, device=device)  # True=pad
        subj_pred_batch = torch.zeros(B, max_len_batch, device=device)
        for i in range(B):
            item = outputs_map[i]
            L = item['embeddings'].shape[0]
            emb_batch[i, :L] = item['embeddings']
            pad_mask_batch[i, :L] = item['pad_mask']
            subj_pred = item['subject_mask_pred']
            subj_pred_batch[i, :L] = subj_pred
        return {
            'embeddings': emb_batch,           # [B,Lmax,C]
            'pad_mask': pad_mask_batch,        # [B,Lmax] True=pad
            'subject_mask_pred': subj_pred_batch  # [B,Lmax]
        }

__all__ = ['TextEncoderWrapper']
