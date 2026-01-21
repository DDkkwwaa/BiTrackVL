import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.models.hip.modules import *
import skimage
from thop import profile

def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    
    # [Fix] Numerical Stability: Subtract max before exp
    maxes = torch.max(values, dim=1, keepdim=True)[0]
    values = values - maxes
    
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    
    # Scatter 0 is fine, but we need to put exp values back
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x

class HistoricalPromptDecoder(nn.Module):
    def __init__(self, isEval=False):
        super().__init__()
        # [Fusion Fix] Residual + Norm layers to prevent gradient explosion
        # mem (512) + qv (512) -> Residual Add -> Norm -> Proj (1024)
        self.fusion_dropout = nn.Dropout(0.1)
        self.fusion_norm = nn.LayerNorm(512)
        self.output_proj = nn.Conv2d(512, 1024, kernel_size=1)

    def set_eval(self, mem_max=150):
        self.topk = 20
        self.CK = None
        self.CV = None
        self.mem_k = None
        self.mem_v = None
        self.mem_text = None # [New] Text Memory
        self.mem_max = mem_max
        self.mem_cnt = 0
        self.imgs = None
        self.path = None

        self.mem_k_bwd = None
        self.mem_v_bwd = None
        self.mem_text_bwd = None # [New] Text Memory Bwd
        self.imgs_bwd = None
        self.mem_cnt_bwd = 0
        self.CK_bwd = None
        self.CV_bwd = None

    def setPath(self, path, seqName, frameId):
        self.path = f"{path}/{seqName}/{frameId}"

    def add_bwd_memory(self, key, value, is_temp=False, searchRegionImg=None, text_feature=None):
        """
        Backward memory writer (SYNC-SAFE)

        Key idea:
          - Only update (mem_k_bwd, mem_v_bwd, mem_text_bwd, mem_cnt_bwd) when we truly commit a frame
            into the official memory (i.e., is_temp == False).
          - If is_temp == True: only update temp_k/temp_v (and optionally temp_text), and RETURN immediately.
          - mem_text_bwd is kept strictly synchronized with the number of committed frames (mem_cnt_bwd).
            If text_feature is missing for a committed frame, we pad a zero text token (only if text memory is enabled).
          - If text memory was not enabled (mem_text_bwd is None) but later a text_feature appears,
            we enable it and backfill zeros for past frames to keep alignment.
        """
        # -----------------------------
        # Optional: cache visualization frames
        # -----------------------------
        if searchRegionImg is not None:
            if self.imgs_bwd is None:
                self.imgs_bwd = torch.from_numpy(searchRegionImg[:, :, ::-1].copy())
            else:
                img = torch.from_numpy(searchRegionImg[:, :, ::-1].copy())
                # NOTE: You used dim=1 in your original code; keep it consistent with your existing design.
                self.imgs_bwd = torch.cat([self.imgs_bwd, img], dim=1)

        # -----------------------------
        # Reset temp each call (as in your original code)
        # -----------------------------
        self.temp_k = None
        self.temp_v = None
        # (Optional) keep a temp text token if you ever want it for temp matching
        self.temp_text_bwd = None

        # -----------------------------
        # Flatten key/value to [B, C, HW]
        # -----------------------------
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        # -----------------------------
        # Normalize text_feature shape to [B, C_text, 1]
        # -----------------------------
        if text_feature is not None and text_feature.dim() == 2:
            text_feature = text_feature.unsqueeze(2)

        # -----------------------------
        # TEMP PATH: do not touch official memory
        # -----------------------------
        if is_temp:
            self.temp_k = key
            self.temp_v = value
            self.temp_text_bwd = text_feature  # optional, not used by default
            return

        # -----------------------------
        # FIRST COMMIT: initialize official memory
        # -----------------------------
        if self.mem_k_bwd is None:
            self.mem_k_bwd = key
            self.mem_v_bwd = value
            self.CK_bwd = key.shape[1]
            self.CV_bwd = value.shape[1]
            self.mem_cnt_bwd = 1

            # Initialize text memory:
            # - If first committed frame has text_feature, store it.
            # - Otherwise keep None (text memory disabled until we see one later).
            self.mem_text_bwd = text_feature  # can be None
            return

        # -----------------------------
        # Capacity control: pop oldest committed frame (k/v AND text)
        # -----------------------------
        if self.mem_cnt_bwd == self.mem_max:
            # Each committed frame contributes HW tokens to mem_k_bwd/mem_v_bwd
            hw = key.shape[2]
            self.mem_k_bwd = self.mem_k_bwd[:, :, hw:]
            self.mem_v_bwd = self.mem_v_bwd[:, :, hw:]

            if self.mem_text_bwd is not None:
                self.mem_text_bwd = self.mem_text_bwd[:, :, 1:]

            self.mem_cnt_bwd -= 1

        # -----------------------------
        # Commit k/v
        # -----------------------------
        self.mem_k_bwd = torch.cat([self.mem_k_bwd, key], dim=2)
        self.mem_v_bwd = torch.cat([self.mem_v_bwd, value], dim=2)

        # -----------------------------
        # Commit text in sync with committed frames
        # -----------------------------
        if self.mem_text_bwd is not None:
            # Text memory already enabled: must append one token every commit
            if text_feature is None:
                B, C, _ = self.mem_text_bwd.shape
                text_feature = torch.zeros(
                    (B, C, 1),
                    device=self.mem_text_bwd.device,
                    dtype=self.mem_text_bwd.dtype
                )
            self.mem_text_bwd = torch.cat([self.mem_text_bwd, text_feature], dim=2)
        else:
            # Text memory not enabled yet.
            # If this commit has a text_feature, enable it now and backfill zeros for past frames.
            if text_feature is not None:
                B, C, _ = text_feature.shape
                zeros_hist = torch.zeros(
                    (B, C, self.mem_cnt_bwd),
                    device=text_feature.device,
                    dtype=text_feature.dtype
                )
                self.mem_text_bwd = torch.cat([zeros_hist, text_feature], dim=2)

        # -----------------------------
        # Update committed frame count (ONLY here)
        # -----------------------------
        self.mem_cnt_bwd += 1

        # -----------------------------
        # Debug safety: ensure strict synchronization if text memory is enabled
        # -----------------------------
        if self.mem_text_bwd is not None:
            assert self.mem_text_bwd.shape[2] == self.mem_cnt_bwd, \
                (self.mem_text_bwd.shape, self.mem_cnt_bwd)

    # def add_bwd_memory(self, key, value, is_temp=False, searchRegionImg=None, text_feature=None):
    #     if searchRegionImg is not None:
    #         if self.imgs_bwd is None:
    #             self.imgs_bwd = torch.from_numpy(searchRegionImg[:, :, ::-1].copy())
    #         else:
    #             img = torch.from_numpy(searchRegionImg[:, :, ::-1].copy())
    #             self.imgs_bwd = torch.cat([self.imgs_bwd, img], dim=1)
    #     self.temp_k = None
    #     self.temp_v = None
    #     key = key.flatten(start_dim=2)
    #     value = value.flatten(start_dim=2)
    #
    #     # [New] Handle Text Feature
    #     if text_feature is not None:
    #         # text_feature: [B, C_text] -> [B, C_text, 1] to match time dimension logic
    #         if text_feature.dim() == 2:
    #             text_feature = text_feature.unsqueeze(2)
    #
    #     if self.mem_k_bwd is None:
    #         # First frame, just shove it in
    #         self.mem_k_bwd = key
    #         self.mem_v_bwd = value
    #         self.mem_text_bwd = text_feature # [New]
    #         self.CK_bwd = key.shape[1]
    #         self.CV_bwd = value.shape[1]
    #         self.mem_cnt_bwd = 1
    #     else:
    #         if is_temp:
    #             self.temp_k = key
    #             self.temp_v = value
    #         else:
    #             if self.mem_cnt_bwd == self.mem_max:
    #                 self.mem_k_bwd = self.mem_k_bwd[:, :, key.shape[2]:]
    #                 self.mem_v_bwd = self.mem_v_bwd[:, :, value.shape[2]:]
    #                 if self.mem_text_bwd is not None: # [New]
    #                      self.mem_text_bwd = self.mem_text_bwd[:, :, 1:]
    #                 self.mem_cnt_bwd -= 1
    #
    #             self.mem_k_bwd = torch.cat([self.mem_k_bwd, key], 2)
    #             self.mem_v_bwd = torch.cat([self.mem_v_bwd, value], 2)
    #     if self.mem_text_bwd is not None:
    #                 if text_feature is None:
    #                     # Pad with zeros to maintain synchronization
    #                     B, C, _ = self.mem_text_bwd.shape
    #                     text_feature = torch.zeros((B, C, 1), device=self.mem_text_bwd.device, dtype=self.mem_text_bwd.dtype)
    #                 self.mem_text_bwd = torch.cat([self.mem_text_bwd, text_feature], 2)
    #                 self.mem_cnt_bwd += 1

    def add_memory(self, key, value, is_temp=False, searchRegionImg=None, text_feature=None):
        if searchRegionImg is not None:
            if self.imgs is None:
                self.imgs = torch.from_numpy(searchRegionImg[:,:,::-1].copy())
            else:
                img = torch.from_numpy(searchRegionImg[:,:,::-1].copy())
                self.imgs = torch.cat([self.imgs, img], dim=1)

        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)
        
        # [New] Handle Text Feature
        if text_feature is not None:
            # text_feature: [B, C_text] -> [B, C_text, 1]
            if text_feature.dim() == 2:
                text_feature = text_feature.unsqueeze(2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.mem_text = text_feature # [New]
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.mem_cnt = 1
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                if self.mem_cnt == self.mem_max:
                    self.mem_k = self.mem_k[:, :, key.shape[2]:]
                    self.mem_v = self.mem_v[:, :, value.shape[2]:]
                    if self.mem_text is not None: # [New]
                        self.mem_text = self.mem_text[:, :, 1:]
                    self.mem_cnt -= 1
                
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
                if self.mem_text is not None: # [New]
                    if text_feature is None:
                        # Pad with zeros to maintain synchronization
                        B, C, _ = self.mem_text.shape
                        text_feature = torch.zeros((B, C, 1), device=self.mem_text.device, dtype=self.mem_text.dtype)
                    self.mem_text = torch.cat([self.mem_text, text_feature], 2)
                self.mem_cnt += 1
 
    def match_memory(self, qk):
        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)

        # One affinity for all
        readout_mem = torch.bmm(affinity, mv)

        return readout_mem.view(qk.shape[0], self.CV, -1)
    
    def _global_matching(self, mk, qk):
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity

    def get_affinity(self, mk, qk, eval=False, visualize=False):
        """
         @Attention : mk [B C T H W]
        """
        #B, CK, THW = mk.shape
        if not eval:
            mk = mk.flatten(start_dim=2)
        B, CK, THW = mk.shape
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum # Normalize
        return affinity

    def readout(self, affinity, mv, qv, eval=False, mem_text=None):
        if not eval:
            mv = mv.flatten(start_dim=2)
        B, CV, THW = mv.shape
        _, _, H, W = qv.shape
        #mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mv, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)
        
        # [Fusion Fix: Residual + Norm]
        # Replaces scale_factor and concat with robust transformer fusion
        
        # 1. Dropout on memory response
        mem = self.fusion_dropout(mem)
        
        # 2. Residual (Query + Memory)
        mem_fused = qv + mem
        
        # 3. LayerNorm (channel-last expectation)
        mem_fused_flat = mem_fused.flatten(2).transpose(1, 2) # [B, HW, C]
        mem_fused_flat = self.fusion_norm(mem_fused_flat)
        mem_fused = mem_fused_flat.transpose(1, 2).view(B, CV, H, W)
        
        # 4. Project 512->1024
        mem_out = self.output_proj(mem_fused)
        
        # [New] Readout Text Memory using the SAME affinity
        text_out = None
        if mem_text is not None:
            # affinity: [B, THW, HW] -> We need to aggregate over THW (time-space)
            # mem_text: [B, C_text, T]
            # But affinity is spatial-temporal flattened.
            # We need to pool affinity spatially to get temporal weights?
            # Or if mem_text is [B, C_text, T], we need weights [B, T]
            
            # Strategy: Since affinity is [B, T*H*W, H*W], it represents how much each pixel in memory contributes to each pixel in query.
            # To get a global text feature for the query frame, we can average the affinity over the query spatial dim (H*W)
            # and sum over the memory spatial dim (H*W) to get weights for each Time step T.
            
            # 1. Sum over query spatial dim (H*W) -> [B, T*H*W, 1]
            # This tells us "how much total attention does this memory pixel get from the whole query image?"
            attn_sum = affinity.sum(dim=2) # [B, T*H*W]
            print("THW=", affinity.shape[1], "HW=", H * W, "T_real=", affinity.shape[1] // (H * W), "T_text=",
                  mem_text.shape[2])
            # 2. Reshape to [B, T, H*W]
            T = mem_text.shape[2]
            attn_sum = attn_sum.view(B, T, -1)
            
            # 3. Sum over memory spatial dim (H*W) -> [B, T]
            # This tells us "how much total attention does this Time Step get?"
            temporal_weights = attn_sum.sum(dim=2) # [B, T]
            
            # 4. Normalize weights (scale down by query spatial size to avoid gradient saturation in Softmax)
            # The sum of temporal_weights is H*W (total query mass). 
            # We divide by H*W so the logits are roughly in [0, 1] range, preserving soft attention.
            scale_factor = H * W
            temporal_weights = temporal_weights / (scale_factor + 1e-6)

            # temporal_weights = F.softmax(temporal_weights, dim=1).unsqueeze(1) # [B, 1, T]
            # [Correction] temporal_weights already sums to 1 (approx). Softmax would dilute it.
            # Just shape it.
            temporal_weights = temporal_weights.unsqueeze(1) # [B, 1, T]

            # 5. Weighted sum of text memory
            text_out = torch.bmm(mem_text, temporal_weights.transpose(1, 2)) # [B, C_text, 1]
            text_out = text_out.squeeze(2) # [B, C_text]

        return mem_out, text_out # Return tuple


class HistoricalPromptNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HistoricalPromptEncoder() 
        self.decoder = HistoricalPromptDecoder()

    def set_eval(self, mem_max):
        self.decoder.set_eval(mem_max)

    def addMemory(self, addMemKey, addMemValue, searchRegionImg, text_feature=None):
        self.decoder.add_memory(addMemKey, addMemValue, searchRegionImg=searchRegionImg, text_feature=text_feature)

    def addbwdMemory(self, addMemKey, addMemValue, searchRegionImg, text_feature=None):
        self.decoder.add_bwd_memory(addMemKey, addMemValue, searchRegionImg=searchRegionImg, text_feature=text_feature)

    def encode(self, frame, kf16, mask, other_mask=None): 
        f16 = self.encoder(frame, kf16, mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def eval_decode(self, queryFrame_key, queryFrame_value):
        affinity = self.decoder.get_affinity(self.decoder.mem_k, queryFrame_key, eval=True, visualize=True)
        return self.decoder.readout(affinity=affinity, mv=self.decoder.mem_v, qv=queryFrame_value, eval=True, mem_text=self.decoder.mem_text)

    def eval_bwd_decode(self, queryFrame_key, queryFrame_value):
        affinity = self.decoder.get_affinity(self.decoder.mem_k_bwd, queryFrame_key, eval=True, visualize=True)
        return self.decoder.readout(affinity=affinity, mv=self.decoder.mem_v_bwd, qv=queryFrame_value, eval=True, mem_text=self.decoder.mem_text_bwd)

    def eval_bwd_decode_clear(self):
        self.decoder.mem_k_bwd = None
        self.decoder.mem_v_bwd = None
        self.decoder.mem_text_bwd = None
        self.decoder.mem_cnt_bwd = 0

    def decode(self, queryFrame_key, queryFrame_value, memory_key, memory_value, memory_text=None):
        """
            queryFrame_key : [B Ck H W]
            queryFrame_value : [B C_v H W]
            memory_key : [B Ck T H W]
            memory_value : [B C_v T H W]
            memory_text : [B C_text T]
        """
        affinity = self.decoder.get_affinity(memory_key, queryFrame_key, visualize=False)
        return self.decoder.readout(affinity=affinity, mv=memory_value, qv=queryFrame_value, mem_text=memory_text)

    def bwd_decode(self, queryFrame_key, queryFrame_value, memory_key, memory_value, memory_text=None):
        """
            queryFrame_key : [B Ck H W]
            queryFrame_value : [B C_v H W]
            memory_key : [B Ck T H W]
            memory_value : [B C_v T H W]
            memory_text : [B C_text T]
        """
        affinity = self.decoder.get_affinity(memory_key, queryFrame_key, visualize=False)
        return self.decoder.readout(affinity=affinity, mv=memory_value, qv=queryFrame_value, mem_text=memory_text)

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode':
            return self.encode(*args, **kwargs)
        elif mode == 'train_decode':
            return self.decode(*args, **kwargs)
        elif mode == 'train_bwd_decode':
            return self.bwd_decode(*args, **kwargs)
        elif mode == 'eval_decode':
            return self.eval_decode(*args, **kwargs)
        elif mode == 'eval_bwd_decode':
            return self.eval_bwd_decode(*args, **kwargs)
        else:
            raise NotImplementedError