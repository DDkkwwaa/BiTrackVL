import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class CEN_Cosine(nn.Module):
    def __init__(self, feature_dim=1024, roi_size=7):
        super(CEN_Cosine, self).__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size

    def forward(self, t, s):
        B, C, H, W = t.shape
        t, s = t.view(B, C, -1), s.view(B, C, -1)
        t, s = F.normalize(t, p=2, dim=1), F.normalize(s, p=2, dim=1)
        return (t * s).mean(dim=1).mean(dim=1, keepdim=True)


class CEN_Dot(nn.Module):
    def __init__(self, feature_dim=1024, roi_size=7):
        super(CEN_Dot, self).__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size

    def forward(self, t, s):
        B, C, H, W = t.shape
        t, s = t.view(B, C, -1), s.view(B, C, -1)
        return (t * s).mean(dim=1).mean(dim=1, keepdim=True)


class CEN_Corr(nn.Module):
    def __init__(self, feature_dim=1024, roi_size=7):
        super(CEN_Corr, self).__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size

    def forward(self, t, s):
        B, C, H, W = t.shape
        corr = F.conv2d(s, t, groups=B)
        return corr.mean(dim=[1, 2, 3], keepdim=True)

class CEN_Attn(nn.Module):
    def __init__(self, feature_dim=1024, roi_size=7):
        super(CEN_Attn, self).__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size

    def forward(self, t, s):
        B, C, H, W = t.shape
        t, s = F.normalize(t, dim=1), F.normalize(s, dim=1)

        # 计算点乘相似度
        sim = (t * s).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        # 🔹 在空间维度上执行 softmax（flatten 再恢复）
        attn = torch.softmax(sim.flatten(2), dim=-1).view(B, 1, H, W)

        # 加权求和得到 IoU 预测
        sim_weighted = (attn * sim).sum(dim=(2, 3), keepdim=True)
        return sim_weighted

class CEN(nn.Module):
    def __init__(self, feature_dim=1024, roi_size=7):
        super(CEN, self).__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size

        # 使用1x1卷积压缩通道维度到feature_dim/4
        self.compressed_dim = feature_dim // 4
        self.compress = nn.Conv2d(feature_dim, self.compressed_dim, kernel_size=1)

        # 展平维度
        self.flattened_dim = self.compressed_dim * roi_size * roi_size

        # 调制向量生成网络 - 使用原来的线性层方式
        self.modulation_fc = nn.Sequential(
            nn.Linear(self.flattened_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.compressed_dim),
            nn.Sigmoid()
        )

        self.iou_predictor = nn.Sequential(
            nn.Linear(self.flattened_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        # scaled sigmoid
        self.margin = 0.25
        self.scale = 1 + self.margin * 2

    def forward(self, template_roi_feature, search_roi_feature):
        b = template_roi_feature.size(0)

        # 压缩特征
        template_comp = self.compress(template_roi_feature)  # [B, feature_dim/4, 7, 7]
        search_comp = self.compress(search_roi_feature)  # [B, feature_dim/4, 7, 7]

        # 特征扁平化
        template_flat = template_comp.view(b, -1)  # [B, (feature_dim/4)*7*7]

        # 生成调制向量
        modulation_vector = self.modulation_fc(template_flat)  # [B, feature_dim/4]
        modulation_vector = modulation_vector.view(b, self.compressed_dim, 1, 1)  # [B, feature_dim/4, 1, 1]

        # 调制搜索特征
        modulated_search = search_comp * modulation_vector  # [B, feature_dim/4, 7, 7]

        # 预测IoU
        modulated_flat = modulated_search.view(b, -1)  # [B, (feature_dim/4)*7*7]
        raw_prediction = self.iou_predictor(modulated_flat)  # [B, 1]

        # 缩放sigmoid
        iou_prediction = self.scale * torch.sigmoid(raw_prediction) - self.margin
        return iou_prediction
