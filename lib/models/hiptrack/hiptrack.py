"""
Basic HIPTrack model.
"""
import math
from lib.utils.box_ops import clip_box
import os
from lib.utils.iou_predictor import CEN,CEN_Dot,CEN_Attn,CEN_Corr,CEN_Cosine
from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from lib.models.layers.head import build_box_head
from lib.models.hiptrack.vit import vit_base_patch16_224
from lib.models.hiptrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce, vit_base_patch16_224_ce_light
from lib.utils.box_ops import box_xyxy_to_cxcywh,box_xywh_to_xyxy,bbox_iou_xywh,bbox_iou_cxcywh
import cv2
import torchvision.transforms.functional as tvisf
from lib.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, visualizeDuringTraining,box_xyxy_to_xywh
from lib.models.hip import HistoricalPromptNetwork
from lib.models.hip.modules import KeyProjection
from lib.models.hip import ResBlock
from collections import deque
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.ce_utils import generate_mask_cond
import torch
import torch.nn as nn
import torch.nn.functional as F


class HIPTrack(nn.Module):
    """ This is the base class for HIPTrack """

    def __init__(self, transformer, box_head, bwd_box_head, aux_loss=False, head_type="CORNER", vis_during_train=False,
                 new_hip=False, memory_max=150, update_interval=10, cfg=None, text_encoder=None, text_cfg: dict = None):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.bwd_box_head = bwd_box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        self.HIP = HistoricalPromptNetwork()
        self.key_proj = KeyProjection(1024, keydim=64)
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.searchRegionFusion = ResBlock(1024, 1024)
        # [Fix] Add LayerNorm to prevent feature explosion from memory accumulation
        # self.prompt_norm = nn.LayerNorm(1024)
        self.prompt_norm = nn.Identity()
        self.new_hip = new_hip
        self.update_interval = update_interval
        if self.new_hip:
            self.upsample = nn.Upsample(scale_factor=2.0, align_corners=True, mode="bilinear")
        self.memorys = []
        self.mem_max = memory_max
        # BackTrack参数初始化
        self.batch_size = None
        self.forward_tracklet = deque(maxlen=update_interval)  # 存储前向轨迹的bounding boxes
        self.device = None
        self.prev_search = None
        self.prev_box = None
        self.iou_list = deque(maxlen=update_interval)
        self.error_index = deque(maxlen=update_interval)
        self.target_to_distractors_vectors = deque(maxlen=update_interval+1)
        self.x = None
        self.feat_sz = None
        self.output_window = None
        self.state = None
        self.backwardflag = False
        self.preprocessor = Preprocessor()
        self.template_factor = None
        self.template_size = None
        self.search_size = None
        self.back_template = None
        self.iou_max = 0
        self.back_state = None
        self.prev_state = None
        self.search_factor = None
        self.state_back_prev = None
        self.update_counts = 0
        self.last_pred_anno_gt = None
        self.Firstidx = None
        self.out_list = deque(maxlen=update_interval)
        self.bwd_out_list = deque(maxlen=update_interval)
        self.pred_anno_gt_list = deque(maxlen=update_interval)
        self.pred_bwd_anno_gt_list = deque(maxlen=update_interval)
        self.pred_bwd_anno_gt = deque(maxlen=update_interval)
        self.pred_anno_gt = deque(maxlen=update_interval)
        self.pred_anno_gt_list = deque(maxlen=update_interval)
        self.pred_bwd_anno_gt_list = deque(maxlen=update_interval)

        self.active_iou_mode = "CEN"
        self.iou_predictor = CEN(feature_dim=1024, roi_size=7)
        self.iou_predictor_Cosine = CEN_Cosine(feature_dim=1024, roi_size=7)
        self.iou_predictor_Dot = CEN_Dot(feature_dim=1024, roi_size=7)
        self.iou_predictor_Attn = CEN_Attn(feature_dim=1024, roi_size=7)
        self.iou_predictor_Corr = CEN_Corr(feature_dim=1024, roi_size=7)
        self.template_roi_feature = None
        self.PrRoIPooling = PrRoIPool2D(pooled_height=7, pooled_width=7, spatial_scale=1.0)

        self.target_to_distractors_vectors_forward =deque(maxlen=update_interval)
        self.cfg = cfg
        self.template_cfg = self.cfg.DATA.TEMPLATE
        self.search_cfg = self.cfg.DATA.SEARCH

        # 文本编码与融合配置
        self.text_encoder = text_encoder
        self.text_cfg = text_cfg or {}
        self.text_fuse_type = self.text_cfg.get('FUSE', 'none')
        self.text_num_heads = int(self.text_cfg.get('NUM_HEADS', 4))
        self.text_gate_mode = self.text_cfg.get('GATE_MODE', 'scalar')
        # 构建 cross-attn 轻量模块 (仅在 FUSE == 'cross_attn')
        if self.text_fuse_type == 'cross_attn':
            embed_dim = 1024  # 与 backbone hidden_dim 保持一致 (Target uses 1024, VLT used 768. I must match Target's backbone dim)
            assert embed_dim % self.text_num_heads == 0, f"embed_dim {embed_dim} 不能被 num_heads {self.text_num_heads} 整除"
            self.text_q = nn.Linear(embed_dim, embed_dim)
            self.text_k = nn.Linear(embed_dim, embed_dim)
            self.text_v = nn.Linear(embed_dim, embed_dim)
            self.text_out = nn.Linear(embed_dim, embed_dim)
            if self.text_gate_mode == 'scalar':
                self.text_gate = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 4),
                    nn.GELU(),
                    nn.Linear(embed_dim // 4, 1)
                )
            elif self.text_gate_mode == 'channel':
                self.text_gate = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim)
                )
            else:
                raise ValueError(f"未知的 GATE_MODE: {self.text_gate_mode}")
            self.text_attn_dropout = nn.Dropout(float(self.text_cfg.get('DROPOUT', 0.1)))
            print(f"[Fusion] 初始化 cross-attn (heads={self.text_num_heads}, gate={self.text_gate_mode})")
        else:
            self.text_q = None
            self.text_k = None
            self.text_v = None
            self.text_out = None
            self.text_gate = None
        # 冻结文本编码器参数（若配置启用 FREEZE 或 FROZEN）
        # 优先读取配置，如果未配置则默认冻结 (True)
        should_freeze = self.text_cfg.get('FREEZE', self.text_cfg.get('FROZEN', True))
        if self.text_encoder is not None and should_freeze:
            # 1. 先冻结所有
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            
            # 2. 解冻投影层 (proj)
            if hasattr(self.text_encoder, 'proj'):
                for p in self.text_encoder.proj.parameters():
                    p.requires_grad = True
            
            # 3. 解冻主体词预测器 (subject_pred)
            if hasattr(self.text_encoder, 'subject_pred') and self.text_encoder.subject_pred is not None:
                for p in self.text_encoder.subject_pred.parameters():
                    p.requires_grad = True
                    
            print(f"[Text] 文本编码器已冻结参数 (should_freeze={should_freeze})，但保留 proj 和 subject_pred 可训练。")

        # 新增：文本-记忆融合策略模块
        self.enable_subject_branch = bool(self.text_cfg.get('ENABLE_SUBJECT_BRANCH', True))
        self.mem_context_align = bool(self.text_cfg.get('MEM_CONTEXT_ALIGN', True))
        self.mem_fusion = bool(self.text_cfg.get('MEM_FUSION', True))
        self.mem_attention_layers = int(self.text_cfg.get('MEM_ATTENTION_LAYERS', 1))
        self.search_attn_layers = int(self.text_cfg.get('SEARCH_ATTN_LAYERS', 1))
        
        if self.text_encoder is not None:
            text_dim = 768 # RoBERTa base
            mem_dim = 512 # HIP memory value dim
            search_dim = 1024 # Backbone output dim
            
            # 1. Memory Projection (Visual Memory -> Text Space)
            self.mem_proj = nn.Linear(mem_dim, text_dim)
            
            # 2. Text-Memory Align (Q=Text, K=Mem, V=Mem)
            self.text_memory_align = nn.MultiheadAttention(embed_dim=text_dim, num_heads=8, batch_first=True)
            
            # 3. Text-Memory Fusion (Q=Concat, K=Mem, V=Mem)
            self.text_memory_fusion = nn.MultiheadAttention(embed_dim=text_dim, num_heads=8, batch_first=True)
            
            # 4. Search-Text Fusion (Q=Search, K=Text, V=Text)
            self.search_text_fusion = nn.MultiheadAttention(embed_dim=search_dim, kdim=text_dim, vdim=text_dim, num_heads=8, batch_first=True)
            
            # 5. Write Weight Computer Parameters
            self.write_weight_a = nn.Parameter(torch.tensor(1.0))
            self.write_weight_b = nn.Parameter(torch.tensor(0.5))
            self.write_weight_c = nn.Parameter(torch.tensor(0.0))
            
            # [New] Text Memory Components
            # 6. Visual Projection for Memory Write
            self.visual_query_proj = nn.Linear(search_dim, text_dim)

            # 7. Visual-Text Cross Attention for Memory Write (Q=Visual(Proj), K=Text, V=Text)
            self.text_visual_cross_attn = nn.MultiheadAttention(embed_dim=text_dim, kdim=text_dim, vdim=text_dim, num_heads=8, batch_first=True)
            
            # 8. Text Memory Fusion (Concat + Linear)
            # Input: Concat(Global Text [768], Specific Text [768]) -> Output: [768]
            self.text_memory_fusion_proj = nn.Linear(text_dim * 2, text_dim)
            self.text_memory_fusion_norm = nn.LayerNorm(text_dim) # Norm for stability

    def _fuse_tokens_with_text(self, search_tokens: torch.Tensor, text_outputs: dict):
        """复用 cross-attn 融合逻辑, 输入视觉 search tokens 与文本编码结果, 返回融合后 tokens"""
        if self.text_fuse_type != 'cross_attn' or text_outputs is None:
            return search_tokens
        txt_emb = text_outputs['embeddings']  # [B,L,C]
        pad_mask = text_outputs['pad_mask']   # [B,L]
        subj_pred = text_outputs.get('subject_mask_pred', torch.zeros_like(pad_mask, dtype=pad_mask.dtype))
        Bq, Nq, Cq = search_tokens.shape
        Bk, Lk, Ck = txt_emb.shape
        if Bq != Bk or Cq != Ck:
            print(f"[Fusion][Warn] Batch 或通道不匹配 Bq={Bq},Bk={Bk},Cq={Cq},Ck={Ck}")
            return search_tokens
        head_dim = Cq // self.text_num_heads
        if head_dim * self.text_num_heads != Cq:
            print(f"[Fusion][Warn] 维度 {Cq} 不能被 num_heads {self.text_num_heads} 整除")
            return search_tokens
        scale = head_dim ** -0.5
        q = self.text_q(search_tokens).view(Bq, Nq, self.text_num_heads, head_dim).permute(0,2,1,3)
        k = self.text_k(txt_emb).view(Bq, Lk, self.text_num_heads, head_dim).permute(0,2,1,3)
        v = self.text_v(txt_emb).view(Bq, Lk, self.text_num_heads, head_dim).permute(0,2,1,3)
        weight = 0.5 + 0.5 * subj_pred.float()
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores + torch.log(weight.unsqueeze(1).unsqueeze(1) + 1e-6)
        attn_scores = attn_scores.masked_fill(pad_mask.float().unsqueeze(1).unsqueeze(1).bool(), float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.text_attn_dropout(attn)
        attn_out = torch.matmul(attn, v).permute(0,2,1,3).contiguous().view(Bq, Nq, Cq)
        attn_out = self.text_out(attn_out)
        pooled = search_tokens.mean(dim=1)
        gate_raw = self.text_gate(pooled)
        if self.text_gate_mode == 'scalar':
            gate = torch.sigmoid(gate_raw).view(Bq,1,1)
        else:
            gate = torch.sigmoid(gate_raw).view(Bq,1,Cq)
        fused = search_tokens + gate * attn_out
        return fused

    # =============== 新策略: 备份旧融合函数，保证可回退 ===============
    def _legacy_fuse(self, search_tokens: torch.Tensor, text_outputs: dict):
        """旧版文本融合备份。保持与 _fuse_tokens_with_text 当前实现一致，便于回退与对比。
        NOTE: 不再直接调用；调试或 ablation 时可切换使用。"""
        return self._fuse_tokens_with_text(search_tokens, text_outputs)

    # =============== 新策略实现 ===============
    def _stage_text_target_context(self, text_outputs: dict):
        """分解文本为主体词强化特征 f_L^T 与原始特征 f_L。
        返回: dict { 'f_L': Tensor[B,L,C], 'f_L_T': Tensor[B,L,C], 'subj_prob': Tensor[B,L] }
        """
        if text_outputs is None:
            return None
        emb = text_outputs['embeddings']      # [B,L,C]
        subj = text_outputs.get('subject_mask_pred', None)
        if subj is None:
            subj = torch.ones(emb.size(0), emb.size(1), device=emb.device)
        
        # f_L^T = embeddings * p_T
        f_L_T = emb * subj.unsqueeze(-1)
        return {'f_L': emb, 'f_L_T': f_L_T, 'subj_prob': subj}

    def _stage_text_memory_align(self, f_L: torch.Tensor, memory_dict: dict):
        """文本与历史记忆第一次对齐 (f_L -> f_L_C)。"""
        if f_L is None:
            return None
        if not self.mem_context_align:
            return f_L
            
        mem_v = memory_dict.get('mem_values', None) # [B, N_mem, 512]
        if mem_v is None or mem_v.shape[1] == 0:
            return f_L
            
        # Project memory to text dim
        mem_proj = self.mem_proj(mem_v) # [B, N_mem, 768]
        
        # Attn(Q=f_L, K=mem_proj, V=mem_proj)
        f_L_C, _ = self.text_memory_align(f_L, mem_proj, mem_proj)
        return f_L_C

    def _stage_text_memory_fusion(self, f_L_T: torch.Tensor, f_L_C: torch.Tensor, memory_dict: dict):
        """第二阶段文本+记忆融合 (concat -> f_XL)。"""
        if f_L_T is None and f_L_C is None:
            return None
        if f_L_T is None: return f_L_C
        if f_L_C is None: return f_L_T
        
        f_concat = torch.cat([f_L_T, f_L_C], dim=1) # [B, L+L, C]
        
        if not self.mem_fusion:
            return f_concat
            
        mem_v = memory_dict.get('mem_values', None)
        if mem_v is None or mem_v.shape[1] == 0:
            return f_concat
            
        mem_proj = self.mem_proj(mem_v)
        
        # Attn(Q=concat, K=mem_proj, V=mem_proj)
        f_XL, _ = self.text_memory_fusion(f_concat, mem_proj, mem_proj)
        return f_XL

    def _compute_write_weight(self, subj_prob: torch.Tensor, attn_entropy: float = 0.0):
        """计算记忆写入权重 w_t。"""
        if subj_prob is None:
            # Return ones of shape [B]
            return torch.ones(self.batch_size, device=self.device)
            
        # Calculate mean per sample in batch [B, L] -> [B]
        mean_p = subj_prob.mean(dim=1)
        entropy = torch.tensor(attn_entropy, device=subj_prob.device)
        
        logit = self.write_weight_a * mean_p - self.write_weight_b * entropy + self.write_weight_c
        w_t = torch.sigmoid(logit)
        
        # Clamp to [0.3, 1.0]
        w_t = torch.clamp(w_t, min=0.3, max=1.0)
        return w_t

    def set_eval(self):
        self.HIP.set_eval(mem_max=self.mem_max)

    def box_back(self, box, resize_factor, H_image, W_image, back=False):
        box = box.view(-1, 4)

        box = (box.mean(
            dim=0) * self.search_size / resize_factor).tolist()
        if not back:
            box = clip_box(self.map_box_forward(box, resize_factor), H_image, W_image, margin=10)
        else:
            box = clip_box(self.map_box_back(box, resize_factor), H_image, W_image, margin=10)
        return box


    def _run_backtrack_validation(self, template, template_boxes, H_image, W_image, update_interval):
        hit_count = 0
        start_iou = 0
        self.update_counts = 0
        
        # Text Fusion Config
        enable_text = (self.text_encoder is not None and self.text_fuse_type == 'cross_attn')
        fuse_memory = bool(self.text_cfg.get('FUSE_MEMORY', False))
        mem_alpha_cfg = float(self.text_cfg.get('MEMORY_ALPHA', 0.5))

        for t in reversed(range(update_interval)):
            if t == update_interval - 1:
                # 逆序处理前N-1帧
                box_t, search_t, index, searchRegionImg_t, info_t, idx_x_t, idx_y_t, state_t, image_t, re_factor_t ,out_forward_t,state_t_1= \
                self.forward_tracklet[t]

                # Text Encoding
                text_outputs = None
                if enable_text and info_t is not None:
                    captions = info_t.get('caption', None)
                    if captions is not None:
                        if not isinstance(captions, (list, tuple)): captions = [captions]
                        captions = [c if c is not None else "" for c in captions]
                        try:
                            text_outputs = self.text_encoder(captions, device=template.device, fp16=bool(self.text_cfg.get('FP16', False)))
                            # print(f"[Text] Backtrack: Active with caption '{captions[0][:30]}...'")
                        except: pass

                original_size = [image_t.shape[:2]]
                search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                    self.process_search_crops_by_index([image_t], box_t.squeeze(0), last_pred_anno_gt=torch.tensor(self.state).unsqueeze(0), crop_using_gt=False, jitter=False)
                )

                template_candi = template.cuda()

                x, aux_dict = self.backbone(z=template_candi, x=search_in_crop.cuda(),
                                            ce_template_mask=None,
                                            ce_keep_rate=None)
                B, _, Ht, Wt = template_candi.shape
                _, _, C = x.shape
                _, _, Hs, Ws = search_t.shape

                searchRegionFeature = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)

                upsampled_template_candi = self.upsample(template_candi)
                # upsampled_template = template_candi
                template_mask = self.generateMask([None, None, None],
                                                  template_boxes,  # xywh
                                                  upsampled_template_candi, x,
                                                  visualizeMask=False, cxcywh=False)

                template_feature_candi = x[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)

                template_feature_candi = self.upsample(template_feature_candi)

                # Generate Template Text Feature
                prompt_text_template = None
                if enable_text and text_outputs is not None:
                    text_stages = self._stage_text_target_context(text_outputs)
                    f_L_T = text_stages['f_L_T']
                    # Flatten [B, C, H, W] -> [B, HW, C]
                    template_feat_flat = template_feature_candi.flatten(2).transpose(1, 2)
                    # Project [B, HW, C] -> [B, HW, 768]
                    query = self.visual_query_proj(template_feat_flat)
                    # Cross Attn
                    prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                    # Pool [B, HW, 768] -> [B, 768]
                    prompt_text_template = prompt_text_seq.mean(dim=1)

                ref_v_template_candi = self.HIP('encode',
                                                upsampled_template_candi,
                                                template_feature_candi,
                                                template_mask.unsqueeze(1))

                k16_template = self.key_proj(template_feature_candi)

                self.HIP.addbwdMemory(k16_template, ref_v_template_candi, searchRegionImg_t, prompt_text_template)

                searchRegionFeature_1 = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
                k16 = self.key_proj(searchRegionFeature_1)

                searchRegionFeature_1_thin = self.key_comp(searchRegionFeature_1)

                historical_prompt, text_out = self.HIP('train_bwd_decode',
                                             k16,  # queryFrame_key
                                             searchRegionFeature_1_thin,  # queryFrame value
                                             k16_template.unsqueeze(2),  # memoryKey
                                             ref_v_template_candi, # memoryValue
                                             prompt_text_template.unsqueeze(2) if prompt_text_template is not None else None) # memoryText

                B, C, H, W = historical_prompt.shape

                historical_prompt = historical_prompt.view(B, C, H * W).permute(0, 2, 1)

                # Text Fusion for Prediction
                search_tokens = x[:, -self.feat_len_s:]
                fused_tokens = search_tokens
                if enable_text and text_outputs is not None and text_out is not None:
                    text_stages = self._stage_text_target_context(text_outputs)
                    f_L_T = text_stages['f_L_T']
                    # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                    text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                    
                    text_fused = self.text_memory_fusion_proj(
                        torch.cat([f_L_T, text_out_expanded], dim=2)
                    )
                    text_fused = self.text_memory_fusion_norm(text_fused)
                    
                    attn_out, _ = self.search_text_fusion(search_tokens, text_fused, text_fused)
                    fused_tokens = search_tokens + self.text_attn_dropout(attn_out)

                out = self.bwd_forward_head(
                    torch.stack([fused_tokens, historical_prompt], dim=0), None,
                    None, None, None, return_topk_boxes=False)
                self.bwd_out_list.append(out)
                last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
                last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes.cuda(), batch_resize_factor.cuda(),
                                                                  original_size)
                self.pred_bwd_anno_gt_list.append((index, last_pred_anno_gt[0]))
                self.last_pred_anno_gt = last_pred_anno_gt
                self.pred_bwd_anno_gt.append(last_pred_anno_gt)
                box_bwd = out['pred_boxes']
                # test1,test2,test4 = self.detect_peaks(out['score_map'])
                # test5,test6,test7 = self.detect_peaks(out_forward_t['score_map'])

                iou = bbox_iou_xywh(torch.tensor(state_t).unsqueeze(0).cuda(), last_pred_anno_gt)

                # if iou > 0.8:
                #     self.target_to_distractors_vectors.append((index, test4))

                self.iou_list.append((index, iou))
                if iou > 0.7:

                    # print('addbwdbank')
                    mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search_in_crop.cuda(), x,
                                             visualizeMask=False, frame=index, seqName=info_t)

                    # Weighted Memory Write
                    if fuse_memory and enable_text and text_outputs is not None:
                        subj_prob = text_outputs.get('subject_mask_pred', None)
                        w_t = self._compute_write_weight(subj_prob)
                        w_t = w_t.view(-1, 1, 1)
                        Btok, Ntok, Ctok = fused_tokens.shape
                        side = int(Ntok ** 0.5)
                        
                        # Generate prompt_text
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_T = text_stages['f_L_T']
                        # Flatten [B, C, H, W] -> [B, HW, C]
                        search_feat_flat = searchRegionFeature.flatten(2).transpose(1, 2)
                        # Project [B, HW, C] -> [B, HW, 768]
                        query = self.visual_query_proj(search_feat_flat)
                        # Cross Attn
                        prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                        # Pool [B, HW, 768] -> [B, 768]
                        prompt_text = prompt_text_seq.mean(dim=1)

                        if side * side == Ntok:
                            fused_map = fused_tokens.permute(0,2,1).view(Btok, Ctok, side, side)
                            fused_for_memory = (1 - w_t) * searchRegionFeature + w_t * fused_map
                            ref_v = self.HIP('encode', search_in_crop.cuda(), fused_for_memory, mask.unsqueeze(1))
                            self.HIP.addbwdMemory(k16, ref_v, searchRegionImg_t, prompt_text)
                        else:
                            ref_v = self.HIP('encode', search_in_crop.cuda(), searchRegionFeature, mask.unsqueeze(1))
                            self.HIP.addbwdMemory(k16, ref_v, searchRegionImg_t, prompt_text)
                    else:
                        ref_v = self.HIP('encode',
                                         search_in_crop.cuda(),
                                         searchRegionFeature,
                                         mask.unsqueeze(1))

                        self.HIP.addbwdMemory(k16, ref_v, searchRegionImg_t)
                # 初始模板作为反向跟踪器的第一帧memerybank内容
                if iou.item() > 0.1:
                    hit_count += 1

            else:
                box_t, search_t, index, searchRegionImg_t, info_t, idx_x_t, idx_y_t, state_t, image_t, re_factor_t ,out_forward_t,state_t_1= \
                    self.forward_tracklet[t]
                
                # Text Encoding
                text_outputs = None
                if enable_text and info_t is not None:
                    captions = info_t.get('caption', None)
                    if captions is not None:
                        if not isinstance(captions, (list, tuple)): captions = [captions]
                        captions = [c if c is not None else "" for c in captions]
                        try:
                            text_outputs = self.text_encoder(captions, device=template.device, fp16=bool(self.text_cfg.get('FP16', False)))
                            # print(f"[Text] Backtrack: Active with caption '{captions[0][:30]}...'")
                        except: pass

                self.Firstidx = index
                original_size = [image_t.shape[:2]]
                search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                    self.process_search_crops_by_index([image_t], box_t.squeeze(0), last_pred_anno_gt=self.last_pred_anno_gt, crop_using_gt=False, jitter=False)
                )
                x, aux_dict = self.backbone(z=template.cuda(), x=search_in_crop.cuda(),
                                            ce_template_mask=None,
                                            ce_keep_rate=None)
                B, _, Ht, Wt = template.shape
                _, _, C = x.shape
                _, _, Hs, Ws = search_t.shape

                k16 = self.key_proj(x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

                searchRegionFeature = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
                searchRegionFeature_thin = self.key_comp(searchRegionFeature)

                historicalPrompt, _ = self.HIP('eval_bwd_decode',
                                            k16,  # queryFrame_key
                                            searchRegionFeature_thin,  # queryFrame_value
                                            )

                B, C, H, W = historicalPrompt.shape

                historicalPrompt = historicalPrompt.view(B, C, H * W).permute(0, 2, 1)

                # Text Fusion for Prediction
                search_tokens = x[:, -self.feat_len_s:]
                fused_tokens = search_tokens
                if enable_text and text_outputs is not None:
                    # Get Backward Memory
                    mem_v_bwd = self.HIP.decoder.mem_v_bwd
                    if mem_v_bwd is not None:
                        mem_v_flat = mem_v_bwd.transpose(1, 2) # [B, N, C]
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_T = text_stages['f_L_T']
                        f_L = text_stages['f_L']
                        f_L_C = self._stage_text_memory_align(f_L, {'mem_values': mem_v_flat})
                        f_XL = self._stage_text_memory_fusion(f_L_T, f_L_C, {'mem_values': mem_v_flat})
                        if f_XL is not None:
                            attn_out, _ = self.search_text_fusion(search_tokens, f_XL, f_XL)
                            fused_tokens = search_tokens + self.text_attn_dropout(attn_out)

                out = self.bwd_forward_head(
                    torch.stack([fused_tokens, historicalPrompt], dim=0), None,
                    None, None, None, return_topk_boxes=False)
                last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
                last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes.cuda(), batch_resize_factor.cuda(),
                                                                  original_size)
                self.last_pred_anno_gt = last_pred_anno_gt
                self.pred_bwd_anno_gt_list.append((index, last_pred_anno_gt[0]))
                self.bwd_out_list.append(out)
                self.pred_bwd_anno_gt.append(last_pred_anno_gt)

                box_bwd = out['pred_boxes']
                # test1,test2,test4 = self.detect_peaks(out['score_map'])
                # test5,test6,test7 = self.detect_peaks(out_forward_t['score_map'])
                iou = bbox_iou_xywh(torch.tensor(state_t).unsqueeze(0).cuda(), last_pred_anno_gt)

                # if iou > 0.8:
                #     self.target_to_distractors_vectors.append((index, test4))

                self.iou_list.append((index, iou))
                # visualizeyuantu2(image_t, torch.tensor(state_t).unsqueeze(0).cuda()[0], last_pred_anno_gt[0],index)
                if iou.item() > 0.7:
                    # print("addbwdMemory", index)
                    mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search_in_crop.cuda(), x,
                                             visualizeMask=False, frame=index, seqName=info_t)

                    # Weighted Memory Write
                    if fuse_memory and enable_text and text_outputs is not None:
                        subj_prob = text_outputs.get('subject_mask_pred', None)
                        w_t = self._compute_write_weight(subj_prob)
                        w_t = w_t.view(-1, 1, 1)
                        Btok, Ntok, Ctok = fused_tokens.shape
                        side = int(Ntok ** 0.5)
                        if side * side == Ntok:
                            fused_map = fused_tokens.permute(0,2,1).view(Btok, Ctok, side, side)
                            fused_for_memory = (1 - w_t) * searchRegionFeature + w_t * fused_map
                            ref_v = self.HIP('encode', search_in_crop.cuda(), fused_for_memory, mask.unsqueeze(1))
                            self.HIP.addbwdMemory(k16, ref_v, searchRegionImg_t)
                        else:
                            ref_v = self.HIP('encode', search_in_crop.cuda(), searchRegionFeature, mask.unsqueeze(1))
                            self.HIP.addbwdMemory(k16, ref_v, searchRegionImg_t)
                    else:
                        ref_v = self.HIP('encode',
                                         search_in_crop.cuda(),
                                         searchRegionFeature,
                                         mask.unsqueeze(1))

                        self.HIP.addbwdMemory(k16, ref_v, searchRegionImg_t)
                # else:
                #     print("xiao")
                if iou.item() > 0.1:
                    hit_count += 1

        iou_list, low_iou_indices = self.compare_frames(self.pred_anno_gt, self.pred_bwd_anno_gt)
        bad_frames = self.validate_low_iou_frames(iou_list, low_iou_indices, self.out_list, self.bwd_out_list)

        if bad_frames:
            min_index = min(bad_frames) + self.Firstidx
            if min_index % 10 == 1 or min_index == 11:
                result = (min_index, state_t_1)
            else:
                result = next((item for item in self.pred_bwd_anno_gt_list if item[0] == min_index - 1), None)
            self.error_index.append(result)
        if hit_count >= update_interval * 0.1:
            self.HIP.eval_bwd_decode_clear()
            return True
        else:
            return False

    def find_maxidx(self, out):
        pred_score_map = out['score_map'].detach().clone()
        response = pred_score_map
        max_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
        idx_y = torch.div(idx, self.feat_sz, rounding_mode="trunc")
        idx_x = idx % self.feat_sz

        return idx_x, idx_y

    def map_box_back(self, pred_box: list, resize_factor: float):
        x20, y20, w20, h20 = self.state_back_prev
        cx_pred, cy_pred, w_pred, h_pred = pred_box
        crop_center_x = x20 + 0.5 * w20
        crop_center_y = y20 + 0.5 * h20
        half_search = self.search_size / 2

        half_side = 0.5 * self.search_size / resize_factor

        cx_real = crop_center_x + (cx_pred - half_side)
        cy_real = crop_center_y + (cy_pred - half_side)
        x19 = cx_real - 0.5 * w_pred
        y19 = cy_real - 0.5 * h_pred

        return [x19, y19, w_pred, h_pred]

    def map_box_forward(self, pred_box: list, resize_factor: float, Train: bool = False):
        if Train:
            cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * \
                               self.state[3]
        else:
            cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx_prev + (cx - half_side)
        cy_real = cy_prev + (cy - half_side)
        # cx_real = cx + (cx_prev - half_side)
        # cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def forward_track(self, index: int, template: torch.Tensor, template_boxes: torch.Tensor, search: torch.Tensor,
                      ce_template_mask=None,
                      ce_keep_rate=None, searchRegionImg=None, info=None, resize_factor=None, state=None, H_image=None,
                      W_image=None, yuantu=None,
                      search_sz=None, feat_sz=None, output_window=None, image_info=None, error_idx=None, vectors=None,
                      template_factor=None, output_sz=None):

        # Text Encoding
        captions = None
        if info is not None and isinstance(info, dict):
            captions = info.get('caption', None)
        
        # Ensure captions is a list for the encoder
        if captions is not None and not isinstance(captions, (list, tuple)):
            captions = [captions]
            
        enable_text = (self.text_encoder is not None and self.text_fuse_type == 'cross_attn' and captions is not None)
        text_outputs = None
        if enable_text:
            try:
                # Handle potential None in list
                captions = [c if c is not None else "" for c in captions]
                
                # Check cache to avoid redundant encoding
                current_caption_key = str(captions)
                if hasattr(self, 'cached_caption_key') and self.cached_caption_key == current_caption_key and hasattr(self, 'cached_text_outputs'):
                    text_outputs = self.cached_text_outputs
                else:
                    device = template.device if isinstance(template, torch.Tensor) else self.device
                    text_outputs = self.text_encoder(captions, device=device,
                                                     fp16=bool(self.text_cfg.get('FP16', False)))
                    
                    # Update cache
                    self.cached_caption_key = current_caption_key
                    self.cached_text_outputs = text_outputs
                    
                    if index == 1:
                        print(f"[Text] Frame {index}: Active with caption '{captions[0][:30]}...'")
            except Exception as e:
                print(f"[Fusion][Warn] Text encoding failed: {e}")
                enable_text = False

        # Memory Fusion Alpha (Use config, no warmup in inference)
        mem_alpha_cfg = float(self.text_cfg.get('MEMORY_ALPHA', 0.5))

        self.state = state
        self.feat_sz = feat_sz
        self.output_window = output_window
        if index <= 10:
            x, aux_dict = self.backbone(z=template, x=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate)
            self.x = x
            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape
            self.search_size = Hs

            upsampled_template = self.upsample(template)
            template_mask = self.generateMask([None, None, None],
                                              template_boxes,
                                              upsampled_template, x,
                                              visualizeMask=False, cxcywh=False)

            template_feature = x[:, :(Ht // 16) ** 2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)

            template_feature = self.upsample(template_feature)

            if index == 1:
                self.batch_ids = torch.arange(B, device=template_boxes.device).view(-1, 1)
                # cat形状为[B, 5]：[batch_id, x1, y1, x2, y2]
                self.template_roi_feature = self.PrRoIPooling(
                    template_feature,
                    torch.cat((self.batch_ids, box_xywh_to_xyxy(template_boxes.float()) * 24), dim=1)
                )  # 结果形状为[B, C=768, 7, 7]

            ref_v_template = self.HIP('encode',
                                      upsampled_template,
                                      template_feature,
                                      template_mask.unsqueeze(1))

            k16_template = self.key_proj(template_feature)

            searchRegionFeature_1 = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            k16 = self.key_proj(searchRegionFeature_1)

            searchRegionFeature_1_thin = self.key_comp(searchRegionFeature_1)

            # Generate Template Text Feature
            prompt_text_template = None
            if enable_text and text_outputs is not None:
                text_stages = self._stage_text_target_context(text_outputs)
                f_L_T = text_stages['f_L_T']
                # Flatten [B, C, H, W] -> [B, HW, C]
                template_feat_flat = template_feature.flatten(2).transpose(1, 2)
                # Project [B, HW, C] -> [B, HW, 768]
                query = self.visual_query_proj(template_feat_flat)
                # Cross Attn
                prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                # Pool [B, HW, 768] -> [B, 768]
                prompt_text_template = prompt_text_seq.mean(dim=1)

            historical_prompt, text_out = self.HIP('train_decode',
                                         k16,  # queryFrame_key
                                         searchRegionFeature_1_thin,  # queryFrame value
                                         k16_template.unsqueeze(2),  # memoryKey
                                         ref_v_template,  # memoryValue
                                         prompt_text_template.unsqueeze(2) if prompt_text_template is not None else None) # memoryText

            B, C, H, W = historical_prompt.shape

            historical_prompt = historical_prompt.view(B, C, H * W).permute(0, 2, 1)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            search_tokens_init = feat_last[:, -self.feat_len_s:]
            fused_tokens_init = search_tokens_init
            
            # Fuse Text Features (Init)
            if enable_text and text_outputs is not None and text_out is not None:
                text_stages = self._stage_text_target_context(text_outputs)
                f_L_T = text_stages['f_L_T']
                # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                
                text_fused = self.text_memory_fusion_proj(
                    torch.cat([f_L_T, text_out_expanded], dim=2)
                )
                text_fused = self.text_memory_fusion_norm(text_fused)
                
                attn_out, _ = self.search_text_fusion(search_tokens_init, text_fused, text_fused)
                fused_tokens_init = search_tokens_init + self.text_attn_dropout(attn_out)

            out = self.forward_head(
                torch.stack([fused_tokens_init, historical_prompt], dim=0),None,None, None,
                None, None, return_topk_boxes=False)
            
            out.update(aux_dict)
            out['backbone_feat'] = x
            self.template_roi_feature = self.updateTemtoSearch(out)
            pred_boxes = out['pred_boxes']
            if index == 5 or index == 10:
                B, _, Ht, Wt = template.shape
                _, _, C = x.shape
                _, _, Hs, Ws = search.shape

                mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search, x,
                                         visualizeMask=False, frame=index, seqName=info)

                # Memory Write
                fuse_memory = bool(self.text_cfg.get('FUSE_MEMORY', False))
                if fuse_memory and enable_text and text_outputs is not None:
                    # Compute Write Weight
                    subj_prob = text_outputs.get('subject_mask_pred', None)
                    w_t = self._compute_write_weight(subj_prob) # [B]
                    w_t = w_t.view(-1, 1, 1) # Broadcast
                    
                    # Generate prompt_text
                    text_stages = self._stage_text_target_context(text_outputs)
                    f_L_T = text_stages['f_L_T']
                    # Flatten [B, C, H, W] -> [B, HW, C]
                    search_feat_flat = searchRegionFeature_1.flatten(2).transpose(1, 2)
                    # Project [B, HW, C] -> [B, HW, 768]
                    query = self.visual_query_proj(search_feat_flat)
                    # Cross Attn
                    prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                    # Pool [B, HW, 768] -> [B, 768]
                    prompt_text = prompt_text_seq.mean(dim=1)

                    # Reshape fused tokens to map
                    Btok, Ntok, Ctok = fused_tokens_init.shape
                    side = int(Ntok ** 0.5)
                    if side * side == Ntok:
                        fused_map = fused_tokens_init.permute(0,2,1).view(Btok, Ctok, side, side)
                        fused_for_memory = (1 - w_t) * searchRegionFeature_1 + w_t * fused_map
                        
                        k16_mem = self.key_proj(fused_for_memory)
                        ref_v = self.HIP('encode', search, fused_for_memory, mask.unsqueeze(1))
                        self.HIP.addMemory(k16_mem, ref_v, searchRegionImg, prompt_text)
                    else:
                        # Fallback
                        ref_v = self.HIP('encode', search, searchRegionFeature_1, mask.unsqueeze(1))
                        self.HIP.addMemory(k16, ref_v, searchRegionImg, prompt_text)
                else:
                    ref_v = self.HIP('encode',
                                     search,
                                     searchRegionFeature_1,
                                     mask.unsqueeze(1))

                    self.HIP.addMemory(k16, ref_v, searchRegionImg)
            out['needy_rectify_index'] = None
            out['idx'] = None
            out['state'] = None
            out['vectors'] = None
            return out

        else:
            self.state = state
            self.feat_sz = feat_sz
            self.output_window = output_window
            self.template_size = output_sz
            self.template_factor = template_factor

            # flops1, params1 = thop.profile(self.backbone, inputs=(template, search, ce_template_mask, ce_keep_rate, None, False, None, None))
            x, aux_dict = self.backbone(z=template, x=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape
            self.search_size = Hs

            k16 = self.key_proj(x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
            # flops2, params2 = thop.profile(self.key_proj, inputs=(x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16),))

            searchRegionFeature = x[:, (Ht // 16) ** 2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            searchRegionFeature_thin = self.key_comp(searchRegionFeature)
            # flops3, params3 = thop.profile(self.key_comp, inputs=(searchRegionFeature,))

            historicalPrompt, text_out = self.HIP('eval_decode',
                                        k16,  # queryFrame_key
                                        searchRegionFeature_thin,  # queryFrame_value
                                        )

            B, C, H, W = historicalPrompt.shape

            historicalPrompt = historicalPrompt.view(B, C, H * W).permute(0, 2, 1)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            search_tokens_eval = feat_last[:, -self.feat_len_s:]
            fused_tokens_eval = search_tokens_eval
            
            # Fuse Text Features (Stable)
            if enable_text and text_outputs is not None and text_out is not None:
                text_stages = self._stage_text_target_context(text_outputs)
                f_L_T = text_stages['f_L_T']
                # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                
                text_fused = self.text_memory_fusion_proj(
                    torch.cat([f_L_T, text_out_expanded], dim=2)
                )
                text_fused = self.text_memory_fusion_norm(text_fused)
                
                attn_out, _ = self.search_text_fusion(search_tokens_eval, text_fused, text_fused)
                fused_tokens_eval = search_tokens_eval + self.text_attn_dropout(attn_out)

            out = self.forward_head(
                torch.stack([fused_tokens_eval, historicalPrompt], dim=0),self.target_to_distractors_vectors_forward,  None, None,
                None, index, return_topk_boxes=False)

            self.template_roi_feature = self.updateTemtoSearch(out)
            out.update(aux_dict)
            # if test1 == 1 :
            #     self.target_to_distractors_vectors_forward[-1] = ((index,[]))
            out['needy_rectify_index'] = None

            out['vectors'] = None
            # out['idx'] = None
            # out['state'] = None
            self.out_list.append(out)

            out['backbone_feat'] = x
            pred_boxes = out['pred_boxes']
            self.backwardflag = False
            idx_x, idx_y = self.find_maxidx(out)

            self.state = self.box_back(pred_boxes, resize_factor, H_image, W_image, False)
            state_tensor = torch.tensor(self.state, device='cuda:0').unsqueeze(0)
            self.pred_anno_gt.append(state_tensor)

            self.pred_anno_gt_list.append((index, self.state))

            self.forward_tracklet.append((
                                         pred_boxes.clone(), search.clone(), index, searchRegionImg, info, idx_x, idx_y,
                                         self.state, yuantu, resize_factor, out,state))
            Flag = True

            # 初始化变量
            if not hasattr(self, 'reverse_executed'):
                self.reverse_executed = False
            if not hasattr(self, 'next_reset_index'):
                self.next_reset_index = None

            # 检查是否需要重置reverse_executed
            if self.next_reset_index is not None and index >= self.next_reset_index:
                self.reverse_executed = False
                self.next_reset_index = None  # 重置后清除

            # 执行反向验证的条件
            if not self.reverse_executed:
                if (index == 20 and len(self.forward_tracklet) == 10) or \
                        (index % (self.update_interval) == 0 and len(self.forward_tracklet) == self.update_interval):
                    Flag = self._run_backtrack_validation(template, template_boxes, H_image, W_image,
                                                          update_interval=len(self.forward_tracklet))

                    self.forward_tracklet.clear()
                    self.iou_list.clear()
                    self.pred_anno_gt.clear()
                    self.pred_bwd_anno_gt.clear()
                    self.reverse_executed = True
                    # 设置下一个重置点为当前索引+update_interval
                    self.next_reset_index = index + (self.update_interval)

            # 处理需要纠正的索引
            if index == 20 or index % self.update_interval == 0:
                if len(self.error_index) >= 1:
                    self.backwardflag = True
                    print(index)
                    out['needy_rectify_index'] = self.error_index.copy()
                    if self.target_to_distractors_vectors is not None:
                        out['vectors'] = self.target_to_distractors_vectors.copy()
                    self.error_index.clear()

            if index % (self.update_interval) == 0:
                # Memory Write (Stable Phase)
                mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search, x,
                                         visualizeMask=False, frame=index, seqName=info)
                
                fuse_memory = bool(self.text_cfg.get('FUSE_MEMORY', False))
                if fuse_memory and enable_text and text_outputs is not None:
                    # Compute Write Weight
                    subj_prob = text_outputs.get('subject_mask_pred', None)
                    w_t = self._compute_write_weight(subj_prob) # [B]
                    w_t = w_t.view(-1, 1, 1) # Broadcast
                    
                    # Reshape fused tokens to map
                    Btok, Ntok, Ctok = fused_tokens_eval.shape
                    side = int(Ntok ** 0.5)
                    
                    # Generate prompt_text
                    prompt_text = None
                    if enable_text and text_outputs is not None:
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_T = text_stages['f_L_T']
                        # Flatten [B, C, H, W] -> [B, HW, C]
                        search_feat_flat = searchRegionFeature.flatten(2).transpose(1, 2)
                        # Project [B, HW, C] -> [B, HW, 768]
                        query = self.visual_query_proj(search_feat_flat)
                        # Cross Attn
                        prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                        # Pool [B, HW, 768] -> [B, 768]
                        prompt_text = prompt_text_seq.mean(dim=1)

                    if side * side == Ntok:
                        fused_map = fused_tokens_eval.permute(0,2,1).view(Btok, Ctok, side, side)
                        fused_for_memory = (1 - w_t) * searchRegionFeature + w_t * fused_map
                        
                        k16_mem = self.key_proj(fused_for_memory)
                        ref_v = self.HIP('encode', search, fused_for_memory, mask.unsqueeze(1))
                        self.HIP.addMemory(k16_mem, ref_v, searchRegionImg, prompt_text)
                    else:
                        # Fallback
                        ref_v = self.HIP('encode', search, searchRegionFeature, mask.unsqueeze(1))
                        self.HIP.addMemory(k16, ref_v, searchRegionImg, prompt_text)
                else:
                    ref_v = self.HIP('encode', search, searchRegionFeature, mask.unsqueeze(1))
                    self.HIP.addMemory(k16, ref_v, searchRegionImg)

                self.forward_tracklet.clear()
            # and out['pred_iou'][0].item() > 0.6
            if index % 20 == 0 and not self.backwardflag:
                self.forward_tracklet.clear()
                print("更新一次模板库, index ", index)
                # HIPTrack.reset_retrack_count
                mask = self.generateMask(aux_dict['removed_indexes_s'],
                                         out['pred_boxes'].squeeze(1),
                                         search, x, visualizeMask=False, frame=index, seqName=info)

                fuse_memory = bool(self.text_cfg.get('FUSE_MEMORY', False))
                if fuse_memory and enable_text:
                    Btok, Ntok, Ctok = fused_tokens_eval.shape
                    side = int(Ntok ** 0.5)
                    
                    # Generate prompt_text
                    prompt_text = None
                    if text_outputs is not None:
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_T = text_stages['f_L_T']
                        # Flatten [B, C, H, W] -> [B, HW, C]
                        search_feat_flat = searchRegionFeature.flatten(2).transpose(1, 2)
                        # Project [B, HW, C] -> [B, HW, 768]
                        query = self.visual_query_proj(search_feat_flat)
                        # Cross Attn
                        prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                        # Pool [B, HW, 768] -> [B, 768]
                        prompt_text = prompt_text_seq.mean(dim=1)

                    if side * side == Ntok:
                        fused_map = fused_tokens_eval.permute(0,2,1).view(Btok, Ctok, side, side)
                        fused_for_memory = (1 - mem_alpha_cfg) * searchRegionFeature + mem_alpha_cfg * fused_map
                        k16_mem = self.key_proj(fused_for_memory)
                        ref_v = self.HIP('encode', search, fused_for_memory, mask.unsqueeze(1))
                        self.HIP.addMemory(k16_mem, ref_v, searchRegionImg, prompt_text)
                        if torch.rand(1).item() < 0.02:
                            mem_delta = (fused_for_memory - searchRegionFeature).norm().item()
                            print(f"[TrackMemoryFusion] phase=stable frame={index} mem_alpha={mem_alpha_cfg:.2f} mem_delta_norm={mem_delta:.3f}")
                    else:
                        print(f"[Fusion][Warn] 稳定阶段 token 数 {Ntok} 不能整形成方形, 跳过记忆融合写入")
                        ref_v = self.HIP('encode',
                                         search,
                                         searchRegionFeature,
                                         mask.unsqueeze(1))
                        self.HIP.addMemory(k16_mem, ref_v, searchRegionImg, prompt_text)
                else:
                    ref_v = self.HIP('encode',
                                     search,
                                     searchRegionFeature,
                                     mask.unsqueeze(1))

                    self.HIP.addMemory(k16, ref_v, searchRegionImg)
            return out

    def process_template_crops(self, template, template_anno):
        """
            Params：
                template (list): 长度为batch_size的列表，每个元素是[1, H, W, C]的numpy数组
                template_anno (torch.Tensor): shape为[B, 4]的CPU张量

            Returns：
                tuple: (template_tensor, template_bbox_in_crop)
                    - template_tensor: 形状为[B, 3, H, W]的GPU张量
                    - template_bbox_in_crop: 形状为[B, 4]的GPU张量，包含裁剪后的边界框坐标
        """
        template_size = self.template_cfg.SIZE
        batch_size = self.batch_size

        # 预分配numpy数组存储中间结果
        cropped_templates = np.zeros(
            (batch_size, template_size, template_size, 3),
            dtype=np.uint8
        )
        boxes_in_crop = np.zeros((batch_size, 4), dtype=np.float32)

        # 处理每个模板图像
        for i in range(batch_size):
            cropped_templates[i], boxes_in_crop[i], _, _ = crop_image_and_get_box(
                template[i],
                template_anno[i],
                self.template_cfg.FACTOR,
                template_size
            )

        # 批量转换为CPU张量并进行预处理
        normalized_template_tensor = torch.from_numpy(cropped_templates).permute(0, 3, 1, 2).float().contiguous().cuda()
        normalized_template_tensor = (normalized_template_tensor / 255.0).clamp(0.0, 1.0)

        normalized_template_tensor = tvisf.normalize(normalized_template_tensor, self.cfg.DATA.MEAN, self.cfg.DATA.STD, inplace=True)

        # 转换裁剪后的边界框为张量
        template_bbox_in_crop = torch.from_numpy(boxes_in_crop).cuda()

        return normalized_template_tensor, template_bbox_in_crop

    def map_pred_center_box_back(self, pred_box: torch.Tensor, jittered_box: torch.Tensor, resize_factor: torch.Tensor,
                                 origin_sz=None):
        """
        将裁剪图像中的预测框映射回原图坐标系。

        Args:
            pred_box: 网络输出的预测框，归一化的cx,cy,w,h格式，[B, 4]
            jittered_box: 用于裁剪的抖动框，原图坐标系下的x,y,w,h格式，[B, 4]
            resize_factor: 裁剪时使用的缩放系数，[B]
            origin_sz: list[B], 每个元素为(H, W)

        Returns:
            torch.Tensor: 原图坐标系下的预测框，x,y,w,h格式，[B, 4]
        """
        search_sz = self.search_cfg.SIZE

        # 1. 将归一化的预测框转换到裁剪图像坐标系
        pred_box_in_crop = pred_box * search_sz  # [B, 4], cx,cy,w,h

        # 2. 准备缩放因子
        resize_factor = resize_factor.unsqueeze(1)  # [B, 1]

        # 3. 预测框在原图中的尺寸
        gt_size = pred_box_in_crop[:, 2:] / resize_factor  # [B, 2], w,h

        # 4. 抖动框中心
        jittered_center = jittered_box[:, :2] + jittered_box[:, 2:] * 0.5  # [B, 2]

        # 5. 裁剪图像中心
        center_in_crop = torch.full((pred_box.shape[0], 2), (search_sz - 1) / 2, device=pred_box.device)  # [B, 2]

        # 6. 预测框中心在原图中的位置
        pred_center = jittered_center + (pred_box_in_crop[:, :2] - center_in_crop) / resize_factor  # [B, 2]

        # 7. 合并为 x, y, w, h 格式
        xywh_gt_bbox = torch.cat([pred_center - 0.5 * gt_size, gt_size], dim=1)  # [B, 4]

        # 8. 如果没有原图大小信息，直接返回
        if origin_sz is None:
            return xywh_gt_bbox

        # 9. 否则对每一个 bbox 进行边界裁剪，避免 inplace 操作
        new_bbox_list = []
        min_size = 1.0  # 最小宽高限制，防止为 0 或负数

        for i, (h, w) in enumerate(origin_sz):
            bbox = xywh_gt_bbox[i]  # [4]
            x = torch.clamp(bbox[0], min=0.0)
            y = torch.clamp(bbox[1], min=0.0)
            width = torch.clamp(bbox[2], min=0.0)
            height = torch.clamp(bbox[3], min=0.0)

            w_tensor = float(w)
            h_tensor = float(h)

            # 调整宽度和高度不超出图像边界
            if x + width > w_tensor:
                width = w_tensor - x
            if y + height > h_tensor:
                height = h_tensor - y

            # 最小宽高限制
            width = torch.clamp(width, min=min_size)
            height = torch.clamp(height, min=min_size)

            new_bbox = torch.stack([x, y, width, height])
            new_bbox_list.append(new_bbox)

        result_bbox = torch.stack(new_bbox_list, dim=0)  # [B, 4]

        return result_bbox
    def update_memery(self,out, memory_keys,memory_values,prompt_key,prompt_value, memory_text=None, prompt_text=None):
        new_keys = []
        new_values = []
        new_texts = []
        
        has_text = (memory_text is not None and prompt_text is not None)
        
        for batch in range(0, out['pred_iou'].shape[0]):
            pred_iou = out['pred_iou'][batch]
            prev_keys = memory_keys[-1]  # [2, 64, T, 24, 24]
            prev_values = memory_values[-1]  # [2, 384, T, 24, 24]
            
            if has_text:
                prev_text = memory_text[-1] # [B, C, T] -> [1, C, T] for this batch

            if pred_iou > 0.8:
                key_b = prompt_key[batch].unsqueeze(0).unsqueeze(2)  # [1, 64, 1, 24, 24]
                value_b = prompt_value[batch].unsqueeze(0)
                if has_text:
                    text_b = prompt_text[batch].unsqueeze(0).unsqueeze(2) # [1, C, 1]
            else:
                # 复用前一个 memory 的最后一个时间帧
                key_b = prev_keys[batch:batch + 1, :, -1:, :, :]  # [1, 64, 1, 24, 24]
                value_b = prev_values[batch:batch + 1, :, -1:, :, :]
                if has_text:
                    text_b = prev_text[batch:batch + 1, :, -1:] # [1, C, 1]

            new_keys.append(key_b)
            new_values.append(value_b)
            if has_text:
                new_texts.append(text_b)
                
        new_keys = torch.cat(new_keys, dim=0)  # [2, 64, 1, 24, 24]
        new_values = torch.cat(new_values, dim=0)  # [2, 384, 1, 24, 24]
        memory_keys.append(new_keys)
        memory_values.append(new_values)
        
        if has_text:
            new_texts = torch.cat(new_texts, dim=0) # [B, C, 1]
            memory_text.append(new_texts)
            return memory_keys, memory_values, memory_text

        return memory_keys,memory_values

    def updateTemtoSearch (self, out):
        pred_ious = out['pred_iou'].view(-1)  # [B]
        roi_feature = out['roi_feature']  # [B, C, H, W]
        mask = pred_ious > 0.5  # [B]

        updated_template = self.template_roi_feature.clone()
        updated_template[mask] = roi_feature[mask]
        return updated_template

    def forward(self, data, ce_keep_rate, return_last_attn=False, previous=None, previous_boxes=None, captions=None, text_outputs=None):
        '''
            template : [B 3 H_z W_z]
            search : [3 * [B 3 H_x W_x]]
            previous : [B L 3 H_x W_x]
        '''

        template = data['template_images'][0]  # tuple[B], elem [H, W, C]
        template_anno = data['template_anno'].squeeze(1).cuda()  # [B, 4]
        search = data['search_images']  # list[search_num], elem [B, H, W, C]
        search_anno = data['search_anno'].cuda()   # [B, search_num, 4]
        original_size = [img.shape[:2] for img in template]  # list[B], [h, w]
        self.batch_size = template_anno.shape[0]
        self.device = template_anno.device

        # 文本编码 (可选，供后续融合)
        if captions is not None and self.text_encoder is not None:
            if text_outputs is None:
                text_outputs = self.text_encoder(captions, device=self.device,
                                                 fp16=bool(self.text_cfg.get('FP16', False)))

        # crop template image
        template_in_crop, template_bbox_in_crop = self.process_template_crops(template, template_anno)

        ce_template_mask = generate_mask_cond(self.cfg, self.batch_size, self.device, template_bbox_in_crop)  # [B, 144]

        # crop search image using the bbox from last frame's result (align training and testing)
        search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
            self.process_search_crops_by_index(search[0], search_anno[:, 0], template_anno,False, False)
        )

        # recode search bbox gt for calculating loss
        all_search_bbox_in_crop = torch.zeros([self.search_cfg.NUMBER, self.batch_size, 4], device=self.device)  # [search_num, B, 4]
        all_search_bbox_in_crop[0] = search_bbox_in_crop  # x, y, w, h, [B, 4]

        x, aux_dict = self.backbone(z=template_in_crop,  # [B, 3, 192, 192]
                                    x=search_in_crop,  # [B, 3, 384, 384]
                                    ce_template_mask=ce_template_mask,  # [B, 144]
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    previous_frames=previous,
                                    previous_anno=previous_boxes)

        B, _, Ht, Wt = template_in_crop.shape
        _, L, C = x.shape
        _, _, Hs, Ws = search_in_crop.shape
        feat_sz_s = self.feat_sz_s
        feat_len_z = L - self.feat_len_s

        upsampled_template = self.upsample(template_in_crop)  # [B, 3, 384, 384]

        # Generate ce_mask and previous bbox masks for the template
        template_mask = self.generateMask([None, None, None],
                                          template_bbox_in_crop,  # [B, 4]
                                          upsampled_template, x,
                                          visualizeMask=False, cxcywh=False)  # [B, 384, 384], mask for template

        template_feature = x[:, :feat_len_z, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        template_feature = self.upsample(template_feature)  # [B, 1024, 24, 24]

        self.batch_ids = torch.arange(B, device=self.device).view(-1, 1)

        self.template_roi_feature = self.PrRoIPooling(
            template_feature,
            torch.cat((self.batch_ids, box_xywh_to_xyxy(template_bbox_in_crop) * 24), dim=1)
        )  # 结果形状为[B, C, 7, 7]


        prompt_value_template = self.HIP('encode',
                                         upsampled_template,
                                         template_feature,
                                         template_mask.unsqueeze(1))  # [B, 512, 1, 24, 24]

        prompt_key_template = self.key_proj(template_feature)  # [B, 64, 24, 24]
        search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)  # [B, 1024, 24, 24]
        query_key = self.key_proj(search_region_feature)  # [B, 64, 24, 24]
        query_value = self.key_comp(search_region_feature)  # [B, 512, 24, 24]

        memory_keys = []
        memory_values = []
        memory_keys.append(prompt_key_template.unsqueeze(2))
        memory_values.append(prompt_value_template)
        
        # [Dual Memory] Initialize Text Memory
        memory_text = []
        f_L = None
        text_stages = None
        if self.text_encoder is not None and text_outputs is not None:
             # Pre-compute Global Text Features
             text_stages = self._stage_text_target_context(text_outputs)
             f_L = text_stages['f_L'] # [B, L, C]
             
             # Generate Template Text Feature
             # template_feature: [B, 1024, 24, 24]
             # Flatten [B, C, H, W] -> [B, HW, C]
             template_feat_flat = template_feature.flatten(2).transpose(1, 2)
             # Project [B, HW, C] -> [B, HW, 768]
             query = self.visual_query_proj(template_feat_flat)
             
             # Cross-Attn: Query=Visual, Key=Text, Value=Text
             # Use f_L_T (Target Enhanced) for consistency
             prompt_text_seq, _ = self.text_visual_cross_attn(query, text_stages['f_L_T'], text_stages['f_L_T'])
             
             # Pool to [B, C]
             prompt_text_template = prompt_text_seq.mean(dim=1) # [B, C]
             
             memory_text.append(prompt_text_template.unsqueeze(2)) # [B, C, 1]

        historical_prompt, text_out = self.HIP('train_decode',
                                     query_key,  # queryFrame_key
                                     query_value,  # queryFrame value
                                     prompt_key_template.unsqueeze(2),  # memoryKey
                                     prompt_value_template, # memoryValue
                                     memory_text=memory_text[0] if memory_text else None)

        historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)
        
        # [Fix] Apply LayerNorm to historical_prompt before fusion
        historical_prompt = self.prompt_norm(historical_prompt)

        # Get the last feature map from the backbone output
        feat_last = x if not isinstance(x, list) else x[-1]  # [B, 720, 768]

        search_tokens = feat_last[:, feat_len_z:]
        
        # New Text-Memory Fusion Pipeline (Frame 1)
        w_t = torch.ones(self.batch_size, device=self.device)
        if self.text_encoder is not None and text_outputs is not None:
            # 1. Prepare Memory (Template only)
            # memory_values is list of [B, 512, 1, 24, 24]
            mem_v_flat = torch.cat([mv.flatten(2) for mv in memory_values], dim=2).transpose(1, 2) # [B, N, 512]
            
            # 2. Text Stages
            # text_stages already computed
            if text_stages is None:
                 text_stages = self._stage_text_target_context(text_outputs)
                 f_L = text_stages['f_L']

            f_L_T = text_stages['f_L_T']
            # f_L already computed
            
            # [Dual Memory] Fuse Retrieved Text Memory (text_out) with Global Text (f_L)
            if text_out is not None:
                # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                f_L_T_fused = torch.cat([f_L_T, text_out_expanded], dim=2) # [B, L, 2C]
                f_L_T = self.text_memory_fusion_proj(f_L_T_fused) # [B, L, C]
                f_L_T = self.text_memory_fusion_norm(f_L_T)
            
            f_L_C = self._stage_text_memory_align(f_L, {'mem_values': mem_v_flat})
            f_XL = self._stage_text_memory_fusion(f_L_T, f_L_C, {'mem_values': mem_v_flat})
            
            # 3. Search Fusion
            attn_weights = None
            if f_XL is not None:
                attn_out, attn_weights = self.search_text_fusion(search_tokens, f_XL, f_XL)
                search_tokens = search_tokens + self.text_attn_dropout(attn_out)
                
            # 4. Write Weight
            w_t = self._compute_write_weight(text_stages['subj_prob']).detach()

        out = self.forward_head(
            torch.stack([search_tokens, historical_prompt], dim=0),
            None, return_topk_boxes=False)

        self.template_roi_feature = self.updateTemtoSearch(out)
        out.update(aux_dict)
        if text_outputs is not None:
            out['subject_mask_pred'] = text_outputs.get('subject_mask_pred')
            if 'attn_weights' in locals() and attn_weights is not None:
                out['text_attn_weights'] = attn_weights
            if 'w_t' in locals() and w_t is not None:
                out['write_weight'] = w_t
        out['backbone_feat'] = x

        # lost_indices = torch.where(out['pred_iou'] < 0.6)[0]

        last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
        last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes, batch_resize_factor, origin_sz=original_size)
        out['last_pred_anno_gt'] = last_pred_anno_gt
        out_list = [out]
        pred_anno_gt = []
        pred_anno_gt.append(last_pred_anno_gt)

        fwd_search = []
        
        # process search image 2 to n
        for i in range(1, self.search_cfg.NUMBER):
            mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search_in_crop, x, visualizeMask=False)

            current_search_region_feature = search_region_feature
            
            # CLEAN MEMORY WRITE: Only visual features, weighted by w_t
            prompt_value = self.HIP('encode',
                                    search_in_crop,
                                    current_search_region_feature,
                                    mask.unsqueeze(1))
            
            # Apply write weight (w_t calculated from previous step)
            # w_t is [B], prompt_value is [B, 512, 1, 24, 24]
            prompt_value = prompt_value * w_t.view(B, 1, 1, 1, 1)

            # [Dual Memory] Generate Text Feature for Memory Write (Previous Frame)
            prompt_text = None
            if self.text_encoder is not None and text_outputs is not None:
                 # current_search_region_feature: [B, 1024, 24, 24]
                 # Flatten [B, C, H, W] -> [B, HW, C]
                 search_feat_flat = current_search_region_feature.flatten(2).transpose(1, 2)
                 # Project [B, HW, C] -> [B, HW, 768]
                 query = self.visual_query_proj(search_feat_flat)
                 
                 prompt_text_seq, _ = self.text_visual_cross_attn(query, text_stages['f_L_T'], text_stages['f_L_T'])
                 prompt_text = prompt_text_seq.mean(dim=1) # [B, C]

            prompt_key = self.key_proj(current_search_region_feature)

            search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                self.process_search_crops_by_index(search[i], search_anno[:, i], last_pred_anno_gt.cuda(),False, False)
            )

            all_search_bbox_in_crop[i] = search_bbox_in_crop
            fwd_search.append(search_in_crop)
            x, aux_dict = self.backbone(z=template_in_crop, x=search_in_crop,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, previous_frames=previous,
                                        previous_anno=previous_boxes)

            search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)
            query_key = self.key_proj(search_region_feature)
            query_value = self.key_comp(search_region_feature)

            historical_prompt, text_out = self.HIP('train_decode',
                                         query_key,  # queryFrame_key
                                         query_value,  # queryFrame value
                                         torch.cat(memory_keys, dim=2),  # memoryKey
                                         torch.cat(memory_values, dim=2), # memoryValue
                                         memory_text=torch.cat(memory_text, dim=2) if memory_text else None)

            historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)
            
            # [Fix] Apply LayerNorm to historical_prompt before fusion
            historical_prompt = self.prompt_norm(historical_prompt)

            feat_last = x if not isinstance(x, list) else x[-1]
            
            search_tokens = feat_last[:, feat_len_z:]
            
            # Text Fusion for current stage (i+1)
            if self.text_encoder is not None and text_outputs is not None:
                # Update Memory Context (now includes previous frame)
                mem_v_flat = torch.cat([mv.flatten(2) for mv in memory_values], dim=2).transpose(1, 2)
                
                # Re-run Text Stages with updated memory
                text_stages = self._stage_text_target_context(text_outputs) # f_L_T is same
                
                # [Dual Memory] Fuse Retrieved Text Memory (text_out) with Global Text (f_L)
                if text_out is not None:
                    # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                    text_out_expanded = text_out.unsqueeze(1).expand(-1, text_stages['f_L_T'].shape[1], -1)
                    f_L_T_fused = torch.cat([text_stages['f_L_T'], text_out_expanded], dim=2) # [B, L, 2C]
                    f_L_T = self.text_memory_fusion_proj(f_L_T_fused) # [B, L, C]
                    f_L_T = self.text_memory_fusion_norm(f_L_T)
                else:
                    f_L_T = text_stages['f_L_T']

                f_L_C = self._stage_text_memory_align(f_L, {'mem_values': mem_v_flat})
                f_XL = self._stage_text_memory_fusion(f_L_T, f_L_C, {'mem_values': mem_v_flat})
                
                # Search Fusion
                attn_weights = None
                if f_XL is not None:
                    attn_out, attn_weights = self.search_text_fusion(search_tokens, f_XL, f_XL)
                    search_tokens = search_tokens + self.text_attn_dropout(attn_out)
                
                # Update Write Weight for next iteration
                w_t = self._compute_write_weight(text_stages['subj_prob']).detach()
            
            tokens_for_head = search_tokens

            out = self.forward_head(
                torch.stack([tokens_for_head, historical_prompt], dim=0),
                None, return_topk_boxes=False)

            out.update(aux_dict)
            if text_outputs is not None:
                out['subject_mask_pred'] = text_outputs.get('subject_mask_pred')
                if 'attn_weights' in locals() and attn_weights is not None:
                    out['text_attn_weights'] = attn_weights
                if 'w_t' in locals() and w_t is not None:
                    out['write_weight'] = w_t

            self.template_roi_feature = self.updateTemtoSearch(out)

            out['backbone_feat'] = x
            last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
            last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes, batch_resize_factor, origin_sz=original_size)
            out['last_pred_anno_gt'] = last_pred_anno_gt
            out_list.append(out)

            if memory_text:
                 memory_keys, memory_values, memory_text = self.update_memery(out, memory_keys, memory_values, prompt_key, prompt_value, memory_text, prompt_text)
            else:
                 memory_keys, memory_values = self.update_memery(out, memory_keys, memory_values, prompt_key, prompt_value)
            pred_anno_gt.append(last_pred_anno_gt)

        # ==================== 反向处理 (search[-1] -> search[0]) ====================
        template = data['template_images'][0]  # tuple[B], elem [H, W, C]
        template_anno = data['template_anno'].squeeze(1)  # [B, 4]
        search = data['search_images']  # list[search_num], elem [B, H, W, C]
        search_anno = data['search_anno']  # [B, search_num, 4]
        original_size = [img.shape[:2] for img in template]  # list[B], [h, w]
        self.batch_size = template_anno.shape[0]
        self.device = 'cuda'
        # 反向操作1：先处理搜索区域
        search = list(reversed(search))  # 反转搜索序列顺序
        search_anno = torch.flip(search_anno, [1])  # 翻转注释顺序


        # crop template image
        template_in_crop, template_bbox_in_crop = self.process_template_crops(template, template_anno)

        ce_template_mask = generate_mask_cond(self.cfg, self.batch_size, self.device,
                                              template_bbox_in_crop)  # [B, 144]

        # crop search image using the bbox from last frame's result (align training and testing)
        search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
            self.process_search_crops_by_index(search[0], search_anno[:, 0], last_pred_anno_gt.cuda(),False, False)
        )

        # recode search bbox gt for calculating loss
        bwd_search_bbox_in_crop = torch.zeros([self.search_cfg.NUMBER, self.batch_size, 4],
                                              device=self.device)  # [search_num, B, 4]
        bwd_search_bbox_in_crop[0] = search_bbox_in_crop  # x, y, w, h, [B, 4]

        x, aux_dict = self.backbone(z=template_in_crop,  # [B, 3, 192, 192]
                                    x=search_in_crop,  # [B, 3, 384, 384]
                                    ce_template_mask=ce_template_mask,  # [B, 144]
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    previous_frames=previous,
                                    previous_anno=previous_boxes)

        B, _, Ht, Wt = template_in_crop.shape
        _, L, C = x.shape
        _, _, Hs, Ws = search_in_crop.shape
        feat_sz_s = self.feat_sz_s
        feat_len_z = L - self.feat_len_s

        upsampled_template = self.upsample(template_in_crop)  # [B, 3, 384, 384]

        # Generate ce_mask and previous bbox masks for the template
        template_mask = self.generateMask([None, None, None],
                                          template_bbox_in_crop,  # [B, 4]
                                          upsampled_template, x,
                                          visualizeMask=False, cxcywh=False)  # [B, 384, 384], mask for template

        template_feature = x[:, :feat_len_z, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        template_feature = self.upsample(template_feature)  # [B, 1024, 24, 24]

        self.batch_ids = torch.arange(B, device=self.device).view(-1, 1)

        prompt_value_template = self.HIP('encode',
                                         upsampled_template,
                                         template_feature,
                                         template_mask.unsqueeze(1))  # [B, 512, 1, 24, 24]

        prompt_key_template = self.key_proj(template_feature)  # [B, 64, 24, 24]
        search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s,
                                                                           feat_sz_s)  # [B, 1024, 24, 24]
        query_key = self.key_proj(search_region_feature)  # [B, 64, 24, 24]
        query_value = self.key_comp(search_region_feature)  # [B, 512, 24, 24]
        memory_bwd_keys = []
        memory_bwd_values = []
        memory_bwd_text = []

        memory_bwd_keys.append(prompt_key_template.unsqueeze(2))
        memory_bwd_values.append(prompt_value_template)
        
        # [Dual Memory] Initialize Backward Text Memory
        if self.text_encoder is not None and text_outputs is not None:
             # Generate template text feature
             text_stages = self._stage_text_target_context(text_outputs)
             f_L = text_stages['f_L'] # [B, C, L]
             
             # Flatten [B, C, H, W] -> [B, HW, C]
             template_feat_flat = template_feature.flatten(2).transpose(1, 2)
             # Project [B, HW, C] -> [B, HW, 768]
             query = self.visual_query_proj(template_feat_flat)
             
             prompt_text_seq, _ = self.text_visual_cross_attn(query, text_stages['f_L_T'], text_stages['f_L_T'])
             prompt_text_template = prompt_text_seq.mean(dim=1) # [B, C]
             
             memory_bwd_text.append(prompt_text_template.unsqueeze(2)) # [B, C, 1]

        historical_prompt, text_out = self.HIP('train_bwd_decode',
                                     query_key,  # queryFrame_key
                                     query_value,  # queryFrame value
                                     prompt_key_template.unsqueeze(2),  # memoryKey
                                     prompt_value_template, # memoryValue
                                     memory_text=memory_bwd_text[0] if memory_bwd_text else None)

        historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)

        # Get the last feature map from the backbone output
        feat_last = x if not isinstance(x, list) else x[-1]  # [B, 720, 768]
        
        search_tokens = feat_last[:, feat_len_z:]

        # --- TEXT FUSION (Backward Init) ---
        w_t_bwd = torch.ones(self.batch_size, device=self.device)
        if self.text_encoder is not None and text_outputs is not None:
             # 1. Prepare Memory (Template only)
             mem_v_flat = torch.cat([mv.flatten(2) for mv in memory_bwd_values], dim=2).transpose(1, 2)
             
             # 2. Text Stages
             # text_stages already computed
             f_L_T = text_stages['f_L_T']
             
             # [Dual Memory] Fuse Retrieved Text Memory (text_out) with Global Text (f_L)
             if text_out is not None:
                # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                f_L_T_fused = torch.cat([f_L_T, text_out_expanded], dim=2) # [B, L, 2C]
                f_L_T = self.text_memory_fusion_proj(f_L_T_fused) # [B, L, C]
                f_L_T = self.text_memory_fusion_norm(f_L_T)
             
             f_L_C = self._stage_text_memory_align(text_stages['f_L'], {'mem_values': mem_v_flat})
             f_XL = self._stage_text_memory_fusion(f_L_T, f_L_C, {'mem_values': mem_v_flat})
             
             # 3. Search Fusion
             if f_XL is not None:
                 attn_out, _ = self.search_text_fusion(search_tokens, f_XL, f_XL)
                 search_tokens = search_tokens + self.text_attn_dropout(attn_out)
             
             # 4. Write Weight
             w_t_bwd = self._compute_write_weight(text_stages['subj_prob']).detach()
        # -----------------------------------

        out = self.bwd_forward_head(
            torch.stack([search_tokens, historical_prompt], dim=0),
            None, return_topk_boxes=False)

        out.update(aux_dict)
        out['backbone_feat'] = x

        self.template_roi_feature = self.updateTemtoSearch(out)

        last_pred_anno = out['pred_boxes'].squeeze(1)   # [B, 1, 4] -> [B, 4], cx, cy, w, h
        last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes, batch_resize_factor,
                                                          origin_sz=original_size)
        out['last_pred_anno_gt'] = last_pred_anno_gt
        out_bwd_list = [out]
        bwd_pred_anno_gt = []
        bwd_pred_anno_gt.append(last_pred_anno_gt)


        # process search image 2 to n
        for i in range(1, self.search_cfg.NUMBER):
            mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search_in_crop, x,
                                     visualizeMask=False)
            
            # [Dual Memory] Generate Text Feature for Memory Write (Previous Frame)
            prompt_text = None
            if self.text_encoder is not None and text_outputs is not None:
                 # search_region_feature: [B, 1024, 24, 24]
                 # Flatten [B, C, H, W] -> [B, HW, C]
                 search_feat_flat = search_region_feature.flatten(2).transpose(1, 2)
                 # Project [B, HW, C] -> [B, HW, 768]
                 query = self.visual_query_proj(search_feat_flat)
                 
                 prompt_text_seq, _ = self.text_visual_cross_attn(query, text_stages['f_L_T'], text_stages['f_L_T'])
                 prompt_text = prompt_text_seq.mean(dim=1) # [B, C]

            prompt_value = self.HIP('encode',
                                    search_in_crop,
                                    search_region_feature,
                                    mask.unsqueeze(1))
            
            # Apply write weight (Backward)
            if 'w_t_bwd' in locals() and w_t_bwd is not None:
                prompt_value = prompt_value * w_t_bwd.view(B, 1, 1, 1, 1)

            prompt_key = self.key_proj(search_region_feature)

            # crop search image using the bbox from last frame's result
            search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                self.process_search_crops_by_index(search[i], search_anno[:, i], last_pred_anno_gt.cuda(), False, False)
            )

            bwd_search_bbox_in_crop[i] = search_bbox_in_crop

            x, aux_dict = self.backbone(z=template_in_crop, x=search_in_crop,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, previous_frames=previous,
                                        previous_anno=previous_boxes)

            search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)
            query_key = self.key_proj(search_region_feature)
            query_value = self.key_comp(search_region_feature)

            historical_prompt, text_out = self.HIP('train_bwd_decode',
                                         query_key,  # queryFrame_key
                                         query_value,  # queryFrame value
                                         torch.cat(memory_bwd_keys, dim=2),  # memoryKey
                                         torch.cat(memory_bwd_values, dim=2), # memoryValue
                                         memory_text=torch.cat(memory_bwd_text, dim=2) if memory_bwd_text else None)

            historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)

            feat_last = x if not isinstance(x, list) else x[-1]
            
            search_tokens = feat_last[:, feat_len_z:]

            # --- TEXT FUSION (Backward Loop) ---
            if self.text_encoder is not None and text_outputs is not None:
                # Update Memory Context
                mem_v_flat = torch.cat([mv.flatten(2) for mv in memory_bwd_values], dim=2).transpose(1, 2)
                
                # Re-run Text Stages
                # text_stages already computed
                f_L_T = text_stages['f_L_T']
                
                # [Dual Memory] Fuse Retrieved Text Memory (text_out) with Global Text (f_L)
                if text_out is not None:
                    # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                    text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                    f_L_T_fused = torch.cat([f_L_T, text_out_expanded], dim=2) # [B, L, 2C]
                    f_L_T = self.text_memory_fusion_proj(f_L_T_fused) # [B, L, C]
                    f_L_T = self.text_memory_fusion_norm(f_L_T)

                f_L_C = self._stage_text_memory_align(text_stages['f_L'], {'mem_values': mem_v_flat})
                f_XL = self._stage_text_memory_fusion(f_L_T, f_L_C, {'mem_values': mem_v_flat})
                
                # Search Fusion
                if f_XL is not None:
                    attn_out, _ = self.search_text_fusion(search_tokens, f_XL, f_XL)
                    search_tokens = search_tokens + self.text_attn_dropout(attn_out)
                
                # Update Write Weight
                w_t_bwd = self._compute_write_weight(text_stages['subj_prob']).detach()
            # -----------------------------------

            out = self.bwd_forward_head(
                torch.stack([search_tokens, historical_prompt], dim=0),
                None, return_topk_boxes=False)

            out.update(aux_dict)
            out['backbone_feat'] = x

            self.template_roi_feature = self.updateTemtoSearch(out)

            last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
            last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes, batch_resize_factor,
                                                              origin_sz=original_size)
            out['last_pred_anno_gt'] = last_pred_anno_gt
            out_bwd_list.append(out)
            if memory_bwd_text:
                 memory_bwd_keys, memory_bwd_values, memory_bwd_text = self.update_memery(out, memory_bwd_keys, memory_bwd_values, prompt_key, prompt_value, memory_bwd_text, prompt_text)
            else:
                 memory_bwd_keys, memory_bwd_values = self.update_memery(out, memory_bwd_keys, memory_bwd_values, prompt_key, prompt_value)

            bwd_pred_anno_gt.append(last_pred_anno_gt)

        iou_list, low_iou_indices = self.compare_frames(pred_anno_gt, bwd_pred_anno_gt)
        bad_frames = self.validate_low_iou_frames(iou_list, low_iou_indices,out_list,out_bwd_list)

        if not bad_frames:
            idx = None
        else:
            print("badframes",bad_frames)
            idx = bad_frames[0]
        if idx is not None:
            # print(f"警告：以下帧的 IoU < 0.3: {low_iou_indices}")
            # print(f"处理：以下帧的 IoU < 0.3: {idx}")
            if idx == 0:
                template = data['template_images'][0]  # tuple[B], elem [H, W, C]
                template_anno = data['template_anno'].squeeze(1).cuda()  # [B, 4]
                search = data['search_images']  # list[search_num], elem [B, H, W, C]
                search_anno = data['search_anno'].cuda()  # [B, search_num, 4]
                original_size = [img.shape[:2] for img in template]  # list[B], [h, w]
                self.batch_size = template_anno.shape[0]
                self.device = template_anno.device

                # crop template image
                template_in_crop, template_bbox_in_crop = self.process_template_crops(template, template_anno)

                ce_template_mask = generate_mask_cond(self.cfg, self.batch_size, self.device,
                                                      template_bbox_in_crop)  # [B, 144]

                # crop search image using the bbox from last frame's result (align training and testing)
                search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                    self.process_search_crops_by_index(search[0], search_anno[:, 0], template_anno,False, False)
                )
                fwd_search = []
                fwd_search.append(search_in_crop)
                # recode search bbox gt for calculating loss
                all_search_bbox_in_crop = torch.zeros([self.search_cfg.NUMBER, self.batch_size, 4],
                                                      device=self.device)  # [search_num, B, 4]
                all_search_bbox_in_crop[0] = search_bbox_in_crop  # x, y, w, h, [B, 4]

                x, aux_dict = self.backbone(z=template_in_crop,  # [B, 3, 192, 192]
                                            x=search_in_crop,  # [B, 3, 384, 384]
                                            ce_template_mask=ce_template_mask,  # [B, 144]
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn,
                                            previous_frames=previous,
                                            previous_anno=previous_boxes)

                B, _, Ht, Wt = template_in_crop.shape
                _, L, C = x.shape
                _, _, Hs, Ws = search_in_crop.shape
                feat_sz_s = self.feat_sz_s
                feat_len_z = L - self.feat_len_s

                upsampled_template = self.upsample(template_in_crop)  # [B, 3, 384, 384]

                # Generate ce_mask and previous bbox masks for the template
                template_mask = self.generateMask([None, None, None],
                                                  template_bbox_in_crop,  # [B, 4]
                                                  upsampled_template, x,
                                                  visualizeMask=False, cxcywh=False)  # [B, 384, 384], mask for template

                template_feature = x[:, :feat_len_z, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
                template_feature = self.upsample(template_feature)  # [B, 1024, 24, 24]

                self.batch_ids = torch.arange(B, device=self.device).view(-1, 1)

                prompt_value_template = self.HIP('encode',
                                                 upsampled_template,
                                                 template_feature,
                                                 template_mask.unsqueeze(1))  # [B, 512, 1, 24, 24]

                prompt_key_template = self.key_proj(template_feature)  # [B, 64, 24, 24]
                search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s,
                                                                                   feat_sz_s)  # [B, 1024, 24, 24]
                query_key = self.key_proj(search_region_feature)  # [B, 64, 24, 24]
                query_value = self.key_comp(search_region_feature)  # [B, 512, 24, 24]
                memory_keys = []
                memory_values = []
                memory_text = []

                memory_keys.append(prompt_key_template.unsqueeze(2))
                memory_values.append(prompt_value_template)
                
                # [Dual Memory] Initialize Text Memory (Re-tracking)
                if self.text_encoder is not None and text_outputs is not None:
                     text_stages = self._stage_text_target_context(text_outputs)
                     f_L_T = text_stages['f_L_T']
                     # Flatten [B, C, H, W] -> [B, HW, C]
                     template_feat_flat = template_feature.flatten(2).transpose(1, 2)
                     # Project [B, HW, C] -> [B, HW, 768]
                     query = self.visual_query_proj(template_feat_flat)
                     # Cross Attn
                     prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                     # Pool [B, HW, 768] -> [B, 768]
                     prompt_text_template = prompt_text_seq.mean(dim=1)
                     
                     memory_text.append(prompt_text_template.unsqueeze(2)) # [B, C, 1]

                historical_prompt, text_out = self.HIP('train_decode',
                                             query_key,  # queryFrame_key
                                             query_value,  # queryFrame value
                                             prompt_key_template.unsqueeze(2),  # memoryKey
                                             prompt_value_template, # memoryValue
                                             memory_text=memory_text[0] if memory_text else None)

                historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)

                # Get the last feature map from the backbone output
                feat_last = x if not isinstance(x, list) else x[-1]  # [B, 720, 768]
                
                search_tokens = feat_last[:, feat_len_z:]

                # --- TEXT FUSION (Re-tracking Init) ---
                w_t = torch.ones(self.batch_size, device=self.device)
                if self.text_encoder is not None and text_outputs is not None and text_out is not None:
                     text_stages = self._stage_text_target_context(text_outputs)
                     f_L_T = text_stages['f_L_T']
                     
                     # [Dual Memory] Fuse Retrieved Text Memory (text_out) with Global Text (f_L)
                     # f_L_T: [B, L, C], text_out: [B, C]
                     # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                     text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                     
                     text_fused = self.text_memory_fusion_proj(
                        torch.cat([f_L_T, text_out_expanded], dim=2) # Cat on feature dim
                     )
                     text_fused = self.text_memory_fusion_norm(text_fused)
                     
                     attn_out, _ = self.search_text_fusion(search_tokens, text_fused, text_fused)
                     search_tokens = search_tokens + self.text_attn_dropout(attn_out)
                     
                     # Re-compute text_stages if needed or use cached
                     if 'text_stages' not in locals():
                         text_stages = self._stage_text_target_context(text_outputs)
                     w_t = self._compute_write_weight(text_stages['subj_prob']).detach()
                # --------------------------------------

                out = self.forward_head(
                    torch.stack([search_tokens, historical_prompt], dim=0),
                    None, return_topk_boxes=False)

                out.update(aux_dict)
                out['backbone_feat'] = x
                self.template_roi_feature = self.updateTemtoSearch(out)
                # lost_indices = torch.where(out['pred_iou'] < 0.6)[0]
                last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
                last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes, batch_resize_factor,
                                                                  origin_sz=original_size)
                out['last_pred_anno_gt'] = last_pred_anno_gt
                out_list = [out]
                # process search image 2 to n
                for i in range(1, self.search_cfg.NUMBER):
                    mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1),
                                             search_in_crop, x, visualizeMask=False)

                    prompt_value = self.HIP('encode',
                                            search_in_crop,
                                            search_region_feature,
                                            mask.unsqueeze(1))
                    
                    # Apply write weight (Re-tracking)
                    if 'w_t' in locals() and w_t is not None:
                        prompt_value = prompt_value * w_t.view(B, 1, 1, 1, 1)

                    prompt_key = self.key_proj(search_region_feature)

                    # crop search image using the bbox from last frame's result
                    search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                        self.process_search_crops_by_index(search[i], search_anno[:, i], last_pred_anno_gt.cuda())
                    )

                    all_search_bbox_in_crop[i] = search_bbox_in_crop
                    fwd_search.append(search_in_crop)
                    x, aux_dict = self.backbone(z=template_in_crop, x=search_in_crop,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn, previous_frames=previous,
                                                previous_anno=previous_boxes)

                    search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)
                    query_key = self.key_proj(search_region_feature)
                    query_value = self.key_comp(search_region_feature)

                    # Generate text prompt for current frame (Write)
                    prompt_text = None
                    if self.text_encoder is not None and text_outputs is not None:
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_T = text_stages['f_L_T']  # [B, C]
                        # Flatten [B, C, H, W] -> [B, HW, C]
                        search_feat_flat = search_region_feature.flatten(2).transpose(1, 2)
                        # Project [B, HW, C] -> [B, HW, 768]
                        query = self.visual_query_proj(search_feat_flat)
                        # Cross Attn
                        prompt_text_seq, _ = self.text_visual_cross_attn(query, f_L_T, f_L_T)
                        # Pool [B, HW, 768] -> [B, 768]
                        prompt_text = prompt_text_seq.mean(dim=1)

                    historical_prompt, text_out = self.HIP('train_decode',
                                                 query_key,  # queryFrame_key
                                                 query_value,  # queryFrame value
                                                 torch.cat(memory_keys, dim=2),  # memoryKey
                                                 torch.cat(memory_values, dim=2), # memoryValue
                                                 torch.cat(memory_text, dim=2) if memory_text else None) # memoryText

                    historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)

                    feat_last = x if not isinstance(x, list) else x[-1]
                    
                    search_tokens = feat_last[:, feat_len_z:]

                    # Fuse Text Features (Read)
                    if self.text_encoder is not None and text_outputs is not None and text_out is not None:
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_T = text_stages['f_L_T']
                        # Fuse global text with retrieved text memory
                        # Expand text_out to [B, 1, C] and repeat to [B, L, C]
                        text_out_expanded = text_out.unsqueeze(1).expand(-1, f_L_T.shape[1], -1)
                        f_L_T_fused = torch.cat([f_L_T, text_out_expanded], dim=2) # [B, L, 2C]
                        text_fused = self.text_memory_fusion_proj(f_L_T_fused) # [B, L, C]
                        text_fused = self.text_memory_fusion_norm(text_fused)
                        
                        attn_out, _ = self.search_text_fusion(search_tokens, text_fused, text_fused)
                        search_tokens = search_tokens + self.text_attn_dropout(attn_out)

                    out = self.forward_head(
                        torch.stack([search_tokens, historical_prompt], dim=0),
                        None, return_topk_boxes=False)

                    out.update(aux_dict)
                    if text_outputs is not None:
                        out['subject_mask_pred'] = text_outputs.get('subject_mask_pred')
                        if 'w_t' in locals() and w_t is not None:
                            out['write_weight'] = w_t
                    self.template_roi_feature = self.updateTemtoSearch(out)
                    out['backbone_feat'] = x
                    last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
                    last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes,
                                                                      batch_resize_factor,
                                                                      origin_sz=original_size)
                    out['last_pred_anno_gt'] = last_pred_anno_gt
                    out_list.append(out)
                    memory_keys, memory_values, memory_text = self.update_memery(out, memory_keys, memory_values, prompt_key,
                                                                    prompt_value, memory_text, prompt_text)

            else:

                template = data['template_images'][0]  # tuple[B], elem [H, W, C]
                template_anno = data['template_anno'].squeeze(1).cuda()  # [B, 4]
                search = data['search_images']  # list[search_num], elem [B, H, W, C]
                search_anno = data['search_anno'].cuda()  # [B, search_num, 4]
                original_size = [img.shape[:2] for img in template]  # list[B], [h, w]
                self.batch_size = template_anno.shape[0]
                self.device = template_anno.device

                out_list = out_list[:idx]
                memory_keys = memory_keys[: idx]
                memory_values = memory_values[:idx]
                anno_gt = bwd_pred_anno_gt[self.search_cfg.NUMBER - idx]

                searchidx = idx
                # crop search image using the bbox from last frame's result (align training and testing)
                search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                    self.process_search_crops_by_index(search[searchidx], search_anno[:, searchidx], anno_gt,False,False)
                )
                for i in range(idx, self.search_cfg.NUMBER):
                    all_search_bbox_in_crop[i] = 0 # x, y, w, h, [B, 4]

                all_search_bbox_in_crop[idx] = search_bbox_in_crop

                x, aux_dict = self.backbone(z=template_in_crop, x=search_in_crop,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, previous_frames=previous,
                                            previous_anno=previous_boxes)

                search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)
                query_key = self.key_proj(search_region_feature)
                query_value = self.key_comp(search_region_feature)

                historical_prompt, _ = self.HIP('train_decode',
                                             query_key,  # queryFrame_key
                                             query_value,  # queryFrame value
                                             torch.cat(memory_keys, dim=2),  # memoryKey
                                             torch.cat(memory_values, dim=2))  # memoryValue

                historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)

                # Fix missing feat_last extraction in original code? Or just ensure it's correct for fusion
                feat_last = x if not isinstance(x, list) else x[-1]
                search_tokens = feat_last[:, feat_len_z:]

                # --- TEXT FUSION (Partial Restart) ---
                w_t = torch.ones(self.batch_size, device=self.device)
                if self.text_encoder is not None and text_outputs is not None:
                     mem_v_flat = torch.cat([mv.flatten(2) for mv in memory_values], dim=2).transpose(1, 2)
                     text_stages = self._stage_text_target_context(text_outputs)
                     f_L_T = text_stages['f_L_T']
                     f_L_C = self._stage_text_memory_align(text_stages['f_L'], {'mem_values': mem_v_flat})
                     f_XL = self._stage_text_memory_fusion(f_L_T, f_L_C, {'mem_values': mem_v_flat})
                     
                     if f_XL is not None:
                         attn_out, _ = self.search_text_fusion(search_tokens, f_XL, f_XL)
                         search_tokens = search_tokens + self.text_attn_dropout(attn_out)
                     
                     w_t = self._compute_write_weight(text_stages['subj_prob']).detach()
                # -------------------------------------

                out = self.forward_head(
                    torch.stack([search_tokens, historical_prompt], dim=0),
                    None, return_topk_boxes=False)

                out.update(aux_dict)
                self.template_roi_feature = self.updateTemtoSearch(out)
                out['backbone_feat'] = x
                last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
                last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes, batch_resize_factor,
                                                                  origin_sz=original_size)
                out['last_pred_anno_gt'] = last_pred_anno_gt
                out_list.append(out)

                for i in range(idx + 1, self.search_cfg.NUMBER):
                    mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1),
                                             search_in_crop, x, visualizeMask=False)

                    prompt_value = self.HIP('encode',
                                            search_in_crop,
                                            search_region_feature,
                                            mask.unsqueeze(1))
                    
                    # Apply write weight (Partial Loop)
                    if 'w_t' in locals() and w_t is not None:
                        prompt_value = prompt_value * w_t.view(B, 1, 1, 1, 1)

                    prompt_key = self.key_proj(search_region_feature)

                    # crop search image using the bbox from last frame's result
                    search_in_crop, search_bbox_in_crop, batch_resize_factor, batch_crop_boxes = (
                        self.process_search_crops_by_index(search[i], search_anno[:, i], last_pred_anno_gt.cuda(),False, False)
                    )

                    all_search_bbox_in_crop[i] = search_bbox_in_crop
                    fwd_search.append(search_in_crop)
                    x, aux_dict = self.backbone(z=template_in_crop, x=search_in_crop,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn, previous_frames=previous,
                                                previous_anno=previous_boxes)

                    search_region_feature = x[:, feat_len_z:, :].permute(0, 2, 1).view(B, C, feat_sz_s, feat_sz_s)
                    query_key = self.key_proj(search_region_feature)
                    query_value = self.key_comp(search_region_feature)

                    historical_prompt, _ = self.HIP('train_decode',
                                                 query_key,  # queryFrame_key
                                                 query_value,  # queryFrame value
                                                 torch.cat(memory_keys, dim=2),  # memoryKey
                                                 torch.cat(memory_values, dim=2))  # memoryValue

                    historical_prompt = historical_prompt.view(B, C, self.feat_len_s).permute(0, 2, 1)

                    feat_last = x if not isinstance(x, list) else x[-1]
                    
                    search_tokens = feat_last[:, feat_len_z:]

                    # --- TEXT FUSION (Partial Loop) ---
                    if self.text_encoder is not None and text_outputs is not None:
                        mem_v_flat = torch.cat([mv.flatten(2) for mv in memory_values], dim=2).transpose(1, 2)
                        text_stages = self._stage_text_target_context(text_outputs)
                        f_L_C = self._stage_text_memory_align(text_stages['f_L'], {'mem_values': mem_v_flat})
                        f_XL = self._stage_text_memory_fusion(text_stages['f_L_T'], f_L_C, {'mem_values': mem_v_flat})
                        
                        if f_XL is not None:
                            attn_out, _ = self.search_text_fusion(search_tokens, f_XL, f_XL)
                            search_tokens = search_tokens + self.text_attn_dropout(attn_out)
                        
                        w_t = self._compute_write_weight(text_stages['subj_prob']).detach()
                    # ----------------------------------

                    out = self.forward_head(
                        torch.stack([search_tokens, historical_prompt], dim=0),
                        None, return_topk_boxes=False)
                    self.template_roi_feature = self.updateTemtoSearch(out)
                    out.update(aux_dict)
                    if text_outputs is not None:
                        out['subject_mask_pred'] = text_outputs.get('subject_mask_pred')
                        if 'w_t' in locals() and w_t is not None:
                            out['write_weight'] = w_t
                    out['backbone_feat'] = x

                    last_pred_anno = out['pred_boxes'].squeeze(1)  # [B, 1, 4] -> [B, 4], cx, cy, w, h
                    last_pred_anno_gt = self.map_pred_center_box_back(last_pred_anno, batch_crop_boxes,
                                                                      batch_resize_factor,
                                                                      origin_sz=original_size)
                    out['last_pred_anno_gt'] = last_pred_anno_gt

                    out_list.append(out)
                    memory_keys, memory_values = self.update_memery(out, memory_keys, memory_values, prompt_key,
                                                                    prompt_value)


        return all_search_bbox_in_crop, bwd_search_bbox_in_crop, out_list, out_bwd_list

    def validate_low_iou_frames(self, iou_list, low_iou_indices,
                                out_list, out_bwd_list):
        """
        多级验证低IoU帧的特征一致性（使用IoU预测器）

        Args:
            iou_list (Tensor): [B, T] 的IoU矩阵
            low_iou_indices (List[List[int]]): 各batch的低IoU帧索引
            out_list (List[Dict]): 正序帧特征列表（1-10帧），每个元素包含'roi_feature'
            out_bwd_list (List[Dict]): 反序帧特征列表（10-1帧），每个元素包含'roi_feature'

        Returns:
            List[Tuple[int, int]]: 通过验证的(batch_idx, frame_idx)元组列表
        """
        final_indices = []
        T = len(out_list)  # 总帧数（应为10）
        # 收集所有batch中iou<0.3的帧索引（去重）
        all_low_iou_frames = set()
        for batch_frames in low_iou_indices:
            all_low_iou_frames.update(batch_frames)
        # 对每个低IoU帧进行批处理验证
        for frame_idx in sorted(all_low_iou_frames):
            # 判断1：反序帧特征 vs 模板（注意out_bwd_list是倒序存储）
            bwd_feature = out_bwd_list[T - 1 - frame_idx]['roi_feature']
            if self.active_iou_mode == "Cosine":
                pred_iou_bwd = self.iou_predictor_Cosine(self.template_roi_feature, bwd_feature)
            elif self.active_iou_mode == "Dot":
                pred_iou_bwd = self.iou_predictor_Dot(self.template_roi_feature, bwd_feature)
            elif self.active_iou_mode == "Attn":
                pred_iou_bwd = self.iou_predictor_Attn(self.template_roi_feature, bwd_feature)
            elif self.active_iou_mode == "Corr":
                pred_iou_bwd = self.iou_predictor_Corr(self.template_roi_feature, bwd_feature)
            elif self.active_iou_mode == "CEN":
                pred_iou_bwd = self.iou_predictor(self.template_roi_feature, bwd_feature)
            # pred_iou_bwd = self.iou_predictor(self.template_roi_feature, bwd_feature)
            if (pred_iou_bwd < 0.6).any():
                # 判断2：正序帧特征 vs 模板
                pred_feature = out_list[frame_idx]['roi_feature']

                if self.active_iou_mode == "Cosine":
                    pred_iou_fwd = self.iou_predictor_Cosine(self.template_roi_feature, pred_feature)
                elif self.active_iou_mode == "Dot":
                    pred_iou_fwd = self.iou_predictor_Dot(self.template_roi_feature, pred_feature)
                elif self.active_iou_mode == "Attn":
                    pred_iou_fwd = self.iou_predictor_Attn(self.template_roi_feature, pred_feature)
                elif self.active_iou_mode == "Corr":
                    pred_iou_fwd = self.iou_predictor_Corr(self.template_roi_feature, pred_feature)
                elif self.active_iou_mode == "CEN":
                    pred_iou_fwd = self.iou_predictor(self.template_roi_feature, pred_feature)
                # pred_iou_fwd = self.iou_predictor(self.template_roi_feature, pred_feature)

                if (pred_iou_fwd < 0.6).any():
                    final_indices.append(frame_idx)

        return final_indices

    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU（交并比）

        参数:
            box1: [x1, y1, w1, h1] 第一个边界框的左上角坐标和宽高
            box2: [x2, y2, w2, h2] 第二个边界框的左上角坐标和宽高

        返回:
            iou: 两个边界框的IoU值
        """
        # 计算两个边界框的左上角和右下角坐标
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

        # 计算交集的左上角和右下角坐标
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)

        # 计算交集的面积
        intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

        # 计算两个边界框的面积
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        # 计算并集的面积
        union_area = box1_area + box2_area - intersection_area

        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def compare_frames(self, pred_anno_gt, bwd_pred_anno_gt):
        """
        比较正序和反序的边界框，逐帧逐batch计算IoU

        参数:
            pred_anno_gt: List[Tensor[B, 4]]，正序帧预测框列表（长度为 T）
            bwd_pred_anno_gt: List[Tensor[B, 4]]，反序帧预测框列表（长度为 T）

        返回:
            iou_list: Tensor，形状 [B, T]，每个batch每一帧的IoU
            low_iou_indices: List[List[int]]，每个batch中IoU < 0.5 的帧索引
        """
        assert len(pred_anno_gt) == len(bwd_pred_anno_gt), "帧数必须一致"
        T = len(pred_anno_gt)
        B = pred_anno_gt[0].shape[0]

        iou_list = torch.zeros(B, T, device=pred_anno_gt[0].device)
        low_iou_indices = [[] for _ in range(B)]

        for t in range(T):
            box1 = pred_anno_gt[t]  # [B, 4]
            box2 = bwd_pred_anno_gt[T - 1 - t]  # [B, 4]

            for b in range(B):
                iou = self.calculate_iou(box1[b], box2[b])
                iou_list[b, t] = iou
                if iou < 0.25:
                    low_iou_indices[b].append(t)

        return iou_list, low_iou_indices

    def process_search_crops_by_index(self, search_img, search_anno, last_pred_anno_gt, crop_using_gt=False, jitter=False):
        """
        Params：
            search_img (list): len=B, [H, W, C]
            search_anno (torch.Tensor): [B, 4]
            last_pred_anno_gt (torch.Tensor): [B, 4]
            training (bool)
            crop_using_gt (bool):
                True - crop search image using gt
                False - crop search image using previous frame's result

        Returns：
            tuple:
                - search_in_crop (torch.Tensor): [B, 3, H, W]
                - search_bbox_in_crop (torch.Tensor): [B, 4]
                - batch_resize_factor (torch.Tensor): [B]
                - batch_crop_boxes (torch.Tensor): [B, 4]
        """
        search_size = self.search_cfg.SIZE
        if self.batch_size:
            batch_size = self.batch_size
        else:
            batch_size = 1
        # 预分配numpy数组存储中间结果
        cropped_searches = np.zeros(
            (batch_size, search_size, search_size, 3),
            dtype=np.uint8
        )
        boxes_in_crop = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)
        resize_factors = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        crop_boxes = torch.zeros((batch_size, 4), dtype=torch.float32, device=self.device)

        # 并行处理多个图像 (可以用多线程进一步优化，但这里保持简单)
        for i in range(batch_size):
            current_gt_box = search_anno[i].cuda()
            last_pred_box = last_pred_anno_gt[i]
            if self.training and jitter:
                # jitter bbox in training
                jittered_box = self.get_jittered_box(current_gt_box) if crop_using_gt else self.get_jittered_box(
                    last_pred_box)
                # print(jittered_box)
                cropped_searches[i], boxes_in_crop[i], resize_factors[i], _ = crop_image_and_get_box_using_jittered_box(
                    search_img[i],
                    current_gt_box,
                    jittered_box,
                    self.search_cfg.FACTOR,
                    search_size
                )
                crop_boxes[i] = jittered_box
            else:
                # 测试模式：使用上一帧预测框的中心
                cropped_searches[i], boxes_in_crop[i], resize_factors[i], _ = crop_image_and_get_box_using_last_center(
                    search_img[i],
                    current_gt_box,
                    last_pred_box,
                    self.search_cfg.FACTOR,
                    search_size
                )
                crop_boxes[i] = last_pred_box

        # 完全保持原代码的转换逻辑，确保设备匹配
        normalized_search_tensor = torch.from_numpy(cropped_searches).to(self.device).permute(0, 3, 1,
                                                                                              2).float().contiguous()  # [B, 3, 384, 384]

        if self.training:
            brightness_factors = torch.rand(batch_size, 1, 1, 1,
                                            device=self.device) * 0.4 + 0.8  # [0.8, 1.2], gaussian distribution
            normalized_search_tensor = (normalized_search_tensor * brightness_factors / 255.0).clamp(0.0, 1.0)
        else:
            normalized_search_tensor = (normalized_search_tensor / 255.0).clamp(0.0, 1.0)

        normalized_search_tensor = tvisf.normalize(normalized_search_tensor, self.cfg.DATA.MEAN, self.cfg.DATA.STD,
                                                   inplace=True)

        return normalized_search_tensor, boxes_in_crop, resize_factors, crop_boxes

    def get_jittered_box(self, box):
        """ 对框进行中心点和尺度抖动
        args:
            box: tensor[4], [x, y, w, h]
            center_jitter_factor: float
            scale_jitter_factor: float
        """
        center_jitter_factor = self.search_cfg.CENTER_JITTER
        scale_jitter_factor = self.search_cfg.SCALE_JITTER

        size_jitter = torch.randn(2, device=box.device)  # gaussian distribution
        center_jitter = torch.rand(2, device=box.device)  # uniform distribution

        jittered_size = box[2:4] * torch.exp(size_jitter * scale_jitter_factor)
        jittered_size = torch.clamp(jittered_size, min=1e-4)  # 防止零或负值
        max_offset = (jittered_size.prod().sqrt() * center_jitter_factor)

        box_center = box[0:2] + 0.5 * box[2:4]
        jittered_center = box_center + max_offset * (center_jitter - 0.5)

        # return tensor[4], [x, y, w, h]
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def deNorm(self, image):
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        return img

    def generateMask(self, ceMasks, predBoxes, img_normed, img_feat, visualizeMask=False, cxcywh=True, frame=None,
                     seqName=None):
        B, _, H_origin, W_origin = img_normed.shape
        masks = torch.zeros((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        pure_ce_masks = torch.ones((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        for i in range(B):
            if cxcywh:
                box = (box_cxcywh_to_xyxy((predBoxes[i])) * H_origin).int()
            else:
                box = (predBoxes[i] * H_origin).int()
                box[2] += box[0]
                box[3] += box[1]

            box[0] = 0 if box[0] < 0 else box[0]
            box[1] = H_origin if box[1] > H_origin else box[1]
            box[2] = W_origin if box[2] > W_origin else box[2]
            box[3] = 0 if box[3] < 0 else box[3]

            if visualizeMask:
                if not os.path.exists(f"./masks_vis/{seqName}/{frame}"):
                    os.makedirs(f"./masks_vis/{seqName}/{frame}")
                img = self.deNorm(img_normed[i])
            # masks[i] = torch.zeros((H_origin, W_origin), dtype=np.uint8)
            masks[i][box[1].item():box[3].item(), box[0].item():box[2].item()] = 1
            if ceMasks[0] is not None and ceMasks[1] is not None and ceMasks[2] is not None:
                ce1 = ceMasks[0][i]
                ce2 = ceMasks[1][i]
                ce3 = ceMasks[2][i]
                ce = torch.cat([ce1, ce2, ce3], axis=0)
                for num in ce:
                    x = int(num) // 24
                    y = int(num) % 24
                    masks[i][x * 16: (x + 1) * 16, y * 16: (y + 1) * 16] = 0
                    pure_ce_masks[i][x * 16: (x + 1) * 16, y * 16: (y + 1) * 16] = 0

            if visualizeMask:
                mask = masks[i].cpu().detach().numpy().astype(np.uint8)
                mask = np.stack([mask, mask, mask], axis=2) * 255
                pure_ce_mask = pure_ce_masks[i].cpu().detach().numpy().astype(np.uint8)
                pure_ce_mask = np.stack([pure_ce_mask, pure_ce_mask, pure_ce_mask], axis=2) * 255
                #cv2.rectangle(mask, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/mask.jpg", mask[:,:,::-1])
                #cv2.imwrite(f"./masks_vis/{seqName}/{frame}/ce_mask.jpg", pure_ce_mask[:,:,::-1])
                #import pdb; pdb.set_trace()
                img2 = img.copy()
                img3 = img.copy()
                img2[mask == 0] = 255
                img3[pure_ce_mask == 0] = 255
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img_with_mask.jpg", img2[:,:,::-1])
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img_CE_mask.jpg", img3[:,:,::-1])
                cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img.jpg", img[:,:,::-1])
        return masks
    def bwd_forward_head(self, cat_feature, target_to_distractors_vectors_forward=None,vectors=None, error_idx=None, gt_score_map=None, index=None, return_topk_boxes=False):
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # self.error_index = error_idx
        _, B, HW, C = cat_feature.shape
        H = int(HW ** 0.5)
        W = H
        originSearch = cat_feature[0].view(B, H, W, C).permute(0, 3, 1, 2)
        dynamicSearch = cat_feature[1].view(B, H, W, C).permute(0, 3, 1, 2)
        enc_opt = self.searchRegionFusion(originSearch + dynamicSearch).view(B, C, HW).permute(0, 2, 1)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map, topkBbox, last_flag = self.bwd_box_head(opt_feat, gt_score_map, return_topk_boxes,index, error_idx , vectors,target_to_distractors_vectors_forward)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)

            # cat形状为[B, 5]：[batch_id, x1, y1, x2, y2]
            search_roi_feature = self.PrRoIPooling(
                originSearch, torch.cat((self.batch_ids, box_cxcywh_to_xyxy(outputs_coord_new.squeeze(1)) * 24), dim=1)
            )  # 结果形状为[B, C=768, 7, 7]
            if self.active_iou_mode == "Cosine":
                pred_iou = self.iou_predictor_Cosine(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "Dot":
                pred_iou = self.iou_predictor_Dot(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "Attn":
                pred_iou = self.iou_predictor_Attn(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "Corr":
                pred_iou = self.iou_predictor_Corr(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "CEN":
                pred_iou = self.iou_predictor(self.template_roi_feature, search_roi_feature)
            # pred_iou = self.iou_predictor(self.template_roi_feature, search_roi_feature)
            if return_topk_boxes:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'topk_pred_boxes': topkBbox,
                    }
            else:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'last_flag': last_flag,
                       'roi_feature':search_roi_feature,
                       'pred_iou': pred_iou
                    }
            return out
        else:
            raise NotImplementedError

    def forward_head(self, cat_feature, target_to_distractors_vectors_forward=None,vectors=None, error_idx=None, gt_score_map=None, index=None, return_topk_boxes=False):
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # self.error_index = error_idx
        _, B, HW, C = cat_feature.shape
        H = int(HW ** 0.5)
        W = H
        originSearch = cat_feature[0].view(B, H, W, C).permute(0, 3, 1, 2)
        dynamicSearch = cat_feature[1].view(B, H, W, C).permute(0, 3, 1, 2)
        enc_opt = self.searchRegionFusion(originSearch + dynamicSearch).view(B, C, HW).permute(0, 2, 1)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out


        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map, topkBbox, last_flag = self.box_head(opt_feat, gt_score_map, return_topk_boxes,index, error_idx , vectors,target_to_distractors_vectors_forward)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)

            # cat形状为[B, 5]：[batch_id, x1, y1, x2, y2]
            search_roi_feature = self.PrRoIPooling(
                originSearch, torch.cat((self.batch_ids, box_cxcywh_to_xyxy(outputs_coord_new.squeeze(1)) * 24), dim=1)
            )  # 结果形状为[B, C=768, 7, 7]

            if self.active_iou_mode == "Cosine":
                pred_iou = self.iou_predictor_Cosine(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "Dot":
                pred_iou = self.iou_predictor_Dot(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "Attn":
                pred_iou = self.iou_predictor_Attn(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "Corr":
                pred_iou = self.iou_predictor_Corr(self.template_roi_feature, search_roi_feature)
            elif self.active_iou_mode == "CEN":
                pred_iou = self.iou_predictor(self.template_roi_feature, search_roi_feature)
            # pred_iou = self.iou_predictor(self.template_roi_feature, search_roi_feature)

            if return_topk_boxes:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'topk_pred_boxes': topkBbox,
                    }
            else:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'last_flag': last_flag,
                       'pred_iou': pred_iou,
                       'roi_feature':search_roi_feature
                    }
            return out
        else:
            raise NotImplementedError


def build_hiptrack(cfg, training=True, text_encoder=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('HIPTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    bwd_box_head = build_box_head(cfg, hidden_dim)

    # 文本编码器自动构建逻辑(仅在训练阶段且配置开启时执行)
    text_cfg = getattr(cfg, 'TEXT', None) or {}
    if text_encoder is None and isinstance(text_cfg, dict) and text_cfg.get('ENABLE', False):
        try:
            # 尝试相对导入，解决部分环境下的路径解析问题
            try:
                from ..text.text_encoder_wrapper import TextEncoderWrapper
            except (ImportError, ValueError):
                from lib.models.text.text_encoder_wrapper import TextEncoderWrapper
            
            local_model_name = text_cfg.get('MODEL_NAME', 'roberta-base')
            text_encoder = TextEncoderWrapper(
                model_name=local_model_name,
                visual_hidden_dim=1024,
                max_len=text_cfg.get('MAX_LEN', 256),
                enable_subject_classifier=bool(text_cfg.get('SUBJECT_CLASSIFIER', True)),
                need_projection=bool(text_cfg.get('PROJ', False)),
                dropout=float(text_cfg.get('DROPOUT', 0.1)),
                cache=bool(text_cfg.get('CACHE', True))
            )
            print(f"[Text] 已构建文本编码器: {local_model_name}")
        except Exception as e:
            print(f"[Text][Warning] 构建文本编码器失败: {e}. 将在后续 forward 中跳过文本处理。")
            text_encoder = None

    model = HIPTrack(
        backbone,
        box_head,
        bwd_box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        new_hip=cfg.MODEL.NEW_HIP,
        memory_max=cfg.MODEL.MAX_MEM,
        update_interval=cfg.TEST.UPDATE_INTERVAL,
        cfg=cfg,
        text_encoder=text_encoder,
        text_cfg=text_cfg if isinstance(text_cfg, dict) else {}
    )

    # Stage Switching Logic
    stage = getattr(cfg.TRAIN, 'STAGE', 1)
    if stage == 2:
        # Stage 2: Freeze backbone, train fusion
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("[Train] Stage 2: Backbone frozen.")
        
        # Ensure text parts are trainable
        if model.text_q is not None:
            for p in model.text_q.parameters(): p.requires_grad = True
            for p in model.text_k.parameters(): p.requires_grad = True
            for p in model.text_v.parameters(): p.requires_grad = True
            for p in model.text_out.parameters(): p.requires_grad = True
            if model.text_gate is not None:
                for p in model.text_gate.parameters(): p.requires_grad = True
    elif stage == 1:
        # Stage 1: Train backbone, freeze fusion (if exists)
        if model.text_q is not None:
            for p in model.text_q.parameters(): p.requires_grad = False
            for p in model.text_k.parameters(): p.requires_grad = False
            for p in model.text_v.parameters(): p.requires_grad = False
            for p in model.text_out.parameters(): p.requires_grad = False
            if model.text_gate is not None:
                for p in model.text_gate.parameters(): p.requires_grad = False
        print("[Train] Stage 1: Text fusion frozen.")

    if ('HIPTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models', cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        # ✅ 将 box_head 的权重复制给 bwd_box_head
        model.bwd_box_head.load_state_dict(model.box_head.state_dict())
    return model

def crop_image_and_get_box_using_last_center(image, current_box, last_box, factor, output_sz):
    """
        按上一帧的坐标中心和尺寸裁剪图像，获取这一帧box在裁剪坐标下的表示（以上一帧box为中心）。

        Params:
            image: 需要裁剪的图像
            current_box: 本帧搜索图像在原图上的坐标
            last_box: 上一帧搜索图像在原图上的坐标 (裁剪中心)
            factor: 放大系数
            output_sz: 最终大小
    """
    # 以上一帧搜索图像box为中心进行裁剪，裁剪大小也由上一帧box决定
    cropped_image, resize_factor, att_mask = sample_target(image, last_box, factor, output_sz)

    # 获取裁剪后本帧搜索图像的box，输入本帧gt_box与上一帧gt_box，计算本帧裁剪后的box坐标
    box_in_crop = transform_image_to_crop(current_box, last_box, resize_factor, torch.Tensor([output_sz, output_sz]),
                                          normalize=True)

    # 返回裁剪后的image、box、缩放系数、有效区域（非pad区域）标记
    return cropped_image, box_in_crop, resize_factor, att_mask

def crop_image_and_get_box_using_jittered_box(image, gt_box, jittered_box, search_area_factor, output_sz):
    """使用抖动框进行裁剪，并计算gt_box在裁剪图像中的位置"""

    # 使用抖动框的中心和尺寸进行裁剪
    cropped_image, resize_factor, att_mask = sample_target(
        image,  # 原图，[H, W, C]，ndarray
        jittered_box,  # 使用抖动后的bbox进行裁剪
        search_area_factor,
        output_sz
    )

    # 计算原gt_box在裁剪图像中的位置
    box_in_crop = transform_image_to_crop(
        gt_box.cuda(),  # 原始gt_box
        jittered_box,  # 裁剪用的抖动框
        resize_factor,
        torch.Tensor([output_sz, output_sz]),
        normalize=True
    )

    return cropped_image, box_in_crop, resize_factor, att_mask

def crop_image_and_get_box(image, box, factor, output_sz):
    """
        裁剪图像并获取裁剪坐标下的box。

        Params:
            image: 需要裁剪的图像，[H, W, C]
            box: 裁剪图像在原图上的坐标，[x, y, w, h]
            factor: 放大系数，float
            output_sz: 最终大小，int
    """
    # 图像裁剪
    cropped_image, resize_factor, att_mask = sample_target(image, box, factor, output_sz)  # [H, W, C], ndarray

    # 获取裁剪后的box
    box=box.cuda()
    box_in_crop = transform_image_to_crop(box, box, resize_factor, torch.Tensor([output_sz, output_sz]),
                                          normalize=True)
    box_in_crop = box_in_crop.cpu().numpy()
    # 返回裁剪后的image、box、缩放系数、有效区域（非pad区域）标记
    return cropped_image, box_in_crop, resize_factor, att_mask

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_template(template, template_boxes, denormalize=True):
    """
    可视化模板图像及边界框

    参数：
    - template: Tensor (1,3,192,192) 模板图像
    - template_boxes: Tensor (1,4) 边界框 [cx, cy, w, h] 或 [x1, y1, x2, y2]
    - denormalize: bool 是否将坐标从归一化转为绝对坐标
    """
    # 转换张量到CPU并解包
    img_tensor = template.squeeze(0).cpu().detach()  # (3,192,192)
    boxes = template_boxes.cpu().numpy()[0]  # (4,)

    # 转换为Matplotlib可显示的格式
    img = img_tensor.permute(1, 2, 0).numpy()  # HWC格式

    # 检查是否需要反归一化（假设输入为归一化坐标）

    H, W = img.shape[:2]
            # 转换 [cx, cy, w, h] 到像素坐标
    x = boxes[0] * W
    y = boxes[1] * H
    w = boxes[2] * W
    h = boxes[3] * H
    # 转为 [x1, y1, x2, y2]
    x1 = x + w / 2
    y1 = y + h / 2
    x2 = x + w
    y2 = y + h
    # 创建绘图对象
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    # 绘制边界框（红色矩形框）
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

    # 添加中心点标记
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    ax.plot(center[0], center[1], 'yo', markersize=8)

    # 隐藏坐标轴
    ax.axis('off')

    # 显示标题信息
    title_info = f"Box: ({x1:.1f}, {y1:.1f})→({x2:.1f}, {y2:.1f})"
    plt.title(f"Template Visualization\n{title_info}")
    plt.show()

def visual(search_t, box_bwd, box_t):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H, W = search_t.shape[2:]  # (H=384, W=384)
    image = search_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [384, 384, 3]
    image = (image - image.min()) / (image.max() - image.min() + 1e-5)  # 避免除0

    def draw_box(ax, box, color, label):
        cx, cy, w, h = box[0][0]
        cx, cy, w, h = cx.item(), cy.item(), w.item(), h.item()

        x_min = (cx - w / 2) * W
        y_min = (cy - h / 2) * H
        w_pixel = w * W
        h_pixel = h * H

        ax.add_patch(Rectangle(
            (x_min, y_min), w_pixel, h_pixel,
            edgecolor=color, facecolor='none', linewidth=2, label=label
        ))
        ax.scatter(cx * W, cy * H, color=color, s=50, marker='x')  # 中心点

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)

    # 画两个框：一个预测框，一个 ground truth 框
    draw_box(ax, box_bwd, 'red', 'back Box')

    draw_box(ax, box_t, 'green', 'forwad Box')

    ax.legend()
    ax.set_title("Prediction vs Ground Truth")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def visual2(search_t, box_bwd, box_t):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H, W = search_t.shape[2:]  # 例如 (384, 384)
    image = search_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    image = (image - image.min()) / (image.max() - image.min() + 1e-5)  # 归一化

    def draw_box_xywh(ax, box, color, label):
        # box格式：[x1, y1, w, h]，且坐标已归一化（0~1）
        x1, y1, w, h = box[0].detach().cpu().numpy()
        x_min = x1 * W
        y_min = y1 * H
        w_pixel = w * W
        h_pixel = h * H

        ax.add_patch(Rectangle(
            (x_min, y_min), w_pixel, h_pixel,
            edgecolor=color, facecolor='none', linewidth=2, label=label
        ))
        # 画中心点
        cx = x_min + w_pixel / 2
        cy = y_min + h_pixel / 2
        ax.scatter(cx, cy, color=color, s=50, marker='x')

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)

    draw_box_xywh(ax, box_bwd, 'red', 'Back Box')
    draw_box_xywh(ax, box_t, 'green', 'Forward Box')

    ax.legend()
    ax.set_title("Prediction vs Ground Truth")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# def visualizeyuantu2(image, box, bwd_box=None, root_path="/data/code_Lon/PycharmProjects/HIPB_up_large/output"):
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches
#     import cv2
#     import os
#
#     # 加载原图 (RGB 格式)
#     image_rgb = image
#
#     # 创建绘图窗口
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     ax.imshow(image_rgb)
#
#     # 提取第一个框的坐标
#     x, y, w, h = box
#     x_min = int(x)
#     y_min = int(y)
#     x_max = int(x_min + w)
#     y_max = int(y_min + h)
#
#     # 创建第一个矩形框并添加到图像中（红色）
#     rect = patches.Rectangle(
#         (x_min, y_min), x_max - x_min, y_max - y_min,
#         linewidth=2, edgecolor='r', facecolor='none'
#     )
#     ax.add_patch(rect)
#     ax.text(x_min, y_min, 'Forward', fontsize=12, color='red', weight='bold')
#
#     # 如果有第二个框，也绘制出来（用不同颜色）
#     if bwd_box is not None:
#         x_bwd, y_bwd, w_bwd, h_bwd = bwd_box
#         x_min_bwd = int(x_bwd)
#         y_min_bwd = int(y_bwd)
#         x_max_bwd = int(x_min_bwd + w_bwd)
#         y_max_bwd = int(y_min_bwd + h_bwd)
#
#         # 创建第二个矩形框并添加到图像中（蓝色）
#         rect_bwd = patches.Rectangle(
#             (x_min_bwd, y_min_bwd), x_max_bwd - x_min_bwd, y_max_bwd - y_min_bwd,
#             linewidth=2, edgecolor='b', facecolor='none'
#         )
#         ax.add_patch(rect_bwd)
#         ax.text(x_min_bwd, y_min_bwd, 'Backward', fontsize=12, color='blue', weight='bold')
#
#     plt.show()
#     # 保存绘制的图像
#     seq_path = os.path.join(root_path, "hiptrack")
#     os.makedirs(seq_path, exist_ok=True)
#     img_save_path = os.path.join(seq_path, f"_matplotlib.jpg")
#     plt.savefig(img_save_path)

def visualizeyuantu2(image, box, bwd_box=None, index=0, root_path="./output"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2
    import os

    # 加载原图 (RGB 格式)
    image_rgb = image

    # 创建绘图窗口
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)

    # 在左上角显示编号（红色文字，无背景框避免遮挡）
    ax.text(10, 30, f'#{index}', fontsize=20, color='red', weight='bold')

    # 提取第一个框的坐标
    x, y, w, h = box
    x_min = int(x)
    y_min = int(y)
    x_max = int(x_min + w)
    y_max = int(y_min + h)

    # 创建第一个矩形框并添加到图像中（红色）
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x_min, y_min, 'Forward', fontsize=12, color='red', weight='bold')

    # 如果有第二个框，也绘制出来（用不同颜色）
    if bwd_box is not None:
        x_bwd, y_bwd, w_bwd, h_bwd = bwd_box
        x_min_bwd = int(x_bwd)
        y_min_bwd = int(y_bwd)
        x_max_bwd = int(x_min_bwd + w_bwd)
        y_max_bwd = int(y_min_bwd + h_bwd)

        # 创建第二个矩形框并添加到图像中（蓝色）
        rect_bwd = patches.Rectangle(
            (x_min_bwd, y_min_bwd), x_max_bwd - x_min_bwd, y_max_bwd - y_min_bwd,
            linewidth=2, edgecolor='b', facecolor='none'
        )
        ax.add_patch(rect_bwd)
        ax.text(x_min_bwd, y_min_bwd, 'Backward', fontsize=12, color='blue', weight='bold')

    # plt.show()
    # 保存绘制的图像
    seq_path = os.path.join(root_path, "hiptrack_11")
    os.makedirs(seq_path, exist_ok=True)
    img_save_path = os.path.join(seq_path, f"frame_{index:06d}.jpg")
    plt.savefig(img_save_path)
    plt.close()  # 关闭图像避免内存泄漏

def visualizeyuantu(image, box, root_path="./output"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import cv2
        import os

        # 加载原图 (RGB 格式)
        image_rgb = image


            # 创建绘图窗口
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)

        # 提取框的坐标
        x, y, w, h = box
        x_min = int(x)
        y_min = int(y)
        x_max = int(x_min + w)
        y_max = int(y_min + h)

        # 创建一个矩形框并添加到图像中
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # 可选：显示预测框信息
        ax.text(x_min, y_min, 'Prediction', fontsize=12, color='red', weight='bold')
        plt.show()
        # 保存绘制的图像
        seq_path = os.path.join(root_path, "hiptrack")
        os.makedirs(seq_path, exist_ok=True)
        img_save_path = os.path.join(seq_path, f"_matplotlib.jpg")
        plt.savefig(img_save_path)
def visualize_4d_score_map(score_map, title="Score Map", save_path=None):
    """
    可视化 [1, 1, 24, 24] 四维得分图的函数

    参数：
        score_map: 四维得分图 (torch.Tensor或numpy数组)
        title: 图像标题
        save_path: 图片保存路径（可选）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # 输入检查和处理
    if isinstance(score_map, torch.Tensor):
        score_map = score_map.detach().cpu().numpy()
    assert score_map.shape == (1, 1, 24, 24), f"输入形状应为 [1,1,24,24]，实际得到 {score_map.shape}"

    # 提取24x24矩阵
    data = np.squeeze(score_map)

    # 创建画布
    plt.figure(figsize=(10, 8))

    # 绘制热力图（添加网格线）
    heatmap = plt.imshow(data,
                         cmap='plasma',  # 改用高对比度配色
                         vmin=0, vmax=1,  # 固定颜色范围
                         extent=[0.5, 24.5, 24.5, 0.5])  # 精确对齐像素中心

    # 添加颜色条
    cbar = plt.colorbar(heatmap, pad=0.02)
    cbar.set_label('Response Score', rotation=270, labelpad=20)

    # 标记最大值
    max_val = data.max()
    max_loc = np.unravel_index(data.argmax(), data.shape)
    plt.scatter(max_loc[1] + 1, max_loc[0] + 1,  # +1对齐像素中心
                color='lime', marker='x', s=200,
                linewidth=3, label=f'Max: {max_val:.3f}')

    # 增强可视化元素
    plt.xticks(np.arange(1, 25, 2), fontsize=8)
    plt.yticks(np.arange(1, 25, 2), fontsize=8)
    plt.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.3)
    plt.xlabel('Width (pixel)', fontsize=10)
    plt.ylabel('Height (pixel)', fontsize=10)
    plt.title(f"{title}\nShape: {score_map.shape}", fontsize=12)
    plt.legend(loc='upper right', framealpha=0.8)

    # 可选保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def show_tensor_image(tensor_img, title="Cropped Patch",save_root="./output/tensor"):
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.use('Agg')
    # matplotlib.use('TkAgg')
    os.makedirs(save_root, exist_ok=True)
    """
    展示单张 shape 为 [1, 1, H, W] 的灰度图像
    """
    # 将 Tensor 转换为 NumPy 数组
    image = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 去掉尺寸为1的维度
    image = (image + 1) / 2.0  # 将范围从[-1,1]映射到[0,1]
    image = np.clip(image, 0, 1)  # 若存在浮点数溢出则裁剪
    # 可视化图像
    plt.figure(figsize=(5, 5))
    plt.imshow(image)  # 灰度图
    plt.title(title)
    plt.axis('off')
    save_path = os.path.join(save_root, f"frame.jpg")
    plt.savefig(save_path)
    plt.close()
    print(f"[保存] {save_path}")



import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_and_save_tracklet_boxes(forward_tracklet, pred_anno_gt, pred_bwd_anno_gt,
                                      save_root="./output/tracklet_viz"):
    os.makedirs(save_root, exist_ok=True)
    n = len(forward_tracklet)
    for i, (fwd_box,bwd_box) in enumerate(zip(pred_anno_gt,reversed(pred_bwd_anno_gt))):
        # 取出图像

        try:
            image_t = forward_tracklet[i][8]
        except Exception as e:
            print(f"[错误] 第 {i} 帧取图像失败：{e}")
            continue

        # 转为 numpy uint8 图像（支持 RGB 或 BGR）
        if isinstance(image_t, torch.Tensor):
            if image_t.dim() == 3 and image_t.shape[0] in [1, 3]:  # CHW
                image_np = image_t.detach().cpu().permute(1, 2, 0).numpy()
            else:  # HWC
                image_np = image_t.detach().cpu().numpy()
        elif isinstance(image_t, np.ndarray):
            image_np = image_t.copy()
        else:
            print(f"[跳过] 图像不是有效类型：{type(image_t)}")
            continue

        # 修正格式
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        if image_np.shape[-1] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # 转成 RGB 供 matplotlib 使用
        image_rgb = image_np[..., ::-1]

        # 创建画布
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_rgb)

        def draw_box(ax, box, color, label):
            print(f"[DEBUG] box type: {type(box)}, content: {box}")
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()
            box = box.reshape(-1)
            if len(box) != 4:
                return
            x, y, w, h = box
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, label, fontsize=10, color=color, weight='bold')

        # 画前向和反向框
        draw_box(ax, fwd_box, 'red', 'Forward')
        draw_box(ax, bwd_box, 'green', 'Backward')

        ax.set_title(f"Frame {i}")
        ax.axis('off')

        # 保存
        save_path = os.path.join(save_root, f"frame_{i:03d}.jpg")
        plt.savefig(save_path)
        plt.close()
        print(f"[保存] {save_resultpath}")