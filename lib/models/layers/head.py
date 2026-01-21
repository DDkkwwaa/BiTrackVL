import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from lib.models.layers.frozen_bn import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()

        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        self.init_index = None
        self.last_return = False
        self.prev_target_coord = None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def detect_peaks_with_validation(self, score_map, ref_vectors, angle_threshold=30, distance_threshold=10.0):
        """
        带目标验证的峰值检测（新增距离约束）
        :param distance_threshold: 允许的最大欧氏距离（单位：像素）
        :其他参数同原函数
        """
        if ref_vectors == []:
            return None, []
        processed_map = score_map.clone()
        cos_threshold = math.cos(math.radians(angle_threshold))
        last_resort_peak = None
        last_resort_vectors = []

        h, w = score_map.shape[2], score_map.shape[3]
        device = score_map.device

        # 提前构建坐标网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )

        while True:
            self.last_return = False
            current_max, current_idx = torch.max(processed_map.flatten(1), dim=1, keepdim=True)
            current_max_value = current_max.item()

            if current_max_value <= 0.18:
                if last_resort_peak is not None:
                    self.last_return = True
                    return last_resort_peak, last_resort_vectors
                return None, []

            # 获取当前峰值坐标
            feat_w = self.feat_sz
            idx_y = torch.div(current_idx, feat_w, rounding_mode='trunc')
            idx_x = current_idx % feat_w
            current_coord = (idx_y.item(), idx_x.item())

            # 高置信度直接返回（原逻辑不变）
            if current_max_value > 0.7:
                self.last_return = True
                response = processed_map.clone()
                dist = torch.sqrt((y_grid - idx_y.item()) ** 2 + (x_grid - idx_x.item()) ** 2)
                mask = (dist <= 0.5).unsqueeze(0).unsqueeze(0)
                response = response.masked_fill(mask, 0)

                current_vectors = []
                while True:
                    sub_max, sub_idx = torch.max(response.flatten(1), dim=1, keepdim=True)
                    if sub_max <= 0.3 * current_max:
                        break

                    sub_y = torch.div(sub_idx, feat_w, rounding_mode='trunc')
                    sub_x = sub_idx % feat_w
                    delta_x = sub_x.item() - idx_x.item()
                    delta_y = sub_y.item() - idx_y.item()
                    current_vectors.append((delta_x, delta_y))

                    dist = torch.sqrt((y_grid - sub_y.item()) ** 2 + (x_grid - sub_x.item()) ** 2)
                    mask = (dist <= 1.5).unsqueeze(0).unsqueeze(0)
                    response = response.masked_fill(mask, 0)

                return current_coord, current_vectors

            # 记录最后一个低置信度峰值（保底用）
            last_resort_peak = current_coord
            last_resort_vectors = []

            # 提取当前峰值的所有子峰向量
            current_vectors = []
            response = processed_map.clone()
            while True:
                sub_max, sub_idx = torch.max(response.flatten(1), dim=1, keepdim=True)
                if sub_max <= 0.3 * current_max :
                    break

                sub_y = torch.div(sub_idx, feat_w, rounding_mode='trunc')
                sub_x = sub_idx % feat_w
                delta_x = sub_x.item() - idx_x.item()
                delta_y = sub_y.item() - idx_y.item()
                current_vectors.append((delta_x, delta_y))
                last_resort_vectors = current_vectors.copy()

                dist = torch.sqrt((y_grid - sub_y.item()) ** 2 + (x_grid - sub_x.item()) ** 2)
                mask = (dist <= 2).unsqueeze(0).unsqueeze(0)
                response = response.masked_fill(mask, 0)

            if ref_vectors:
                # if sub_max  / current_max < 0.35 :
                #     return current_coord, current_vectors
                tensor_ref = torch.tensor(ref_vectors, dtype=torch.float32)  # 形状 [n_ref, 2]
                tensor_curr = torch.tensor(current_vectors, dtype=torch.float32)  # 形状 [n_curr, 2]

                # 确保至少有一个参考向量和当前向量
                if tensor_ref.shape[0] == 0 or tensor_curr.shape[0] == 0:
                    return None, []

                # 1. 计算余弦相似度矩阵 [n_curr, n_ref]
                norm_curr = torch.norm(tensor_curr, dim=1, keepdim=True) + 1e-8  # [n_curr, 1]
                norm_ref = torch.norm(tensor_ref, dim=1, keepdim=True) + 1e-8  # [n_ref, 1]
                unit_curr = tensor_curr / norm_curr
                unit_ref = tensor_ref / norm_ref
                similarity = torch.mm(unit_curr, unit_ref.T)  # [n_curr, n_ref]

                # 2. 计算距离验证矩阵 [n_curr, n_ref]
                curr_norms = torch.norm(tensor_curr, dim=1, keepdim=True)  # [n_curr, 1]
                ref_norms = torch.norm(tensor_ref, dim=1, keepdim=True).T  # [1, n_ref] (转置)
                dist_ratio = torch.abs(curr_norms - ref_norms)  # 广播为 [n_curr, n_ref]
                dist_valid = (dist_ratio <= distance_threshold)  # [n_curr, n_ref]

                # 3. 联合验证 (维度自动对齐)
                valid_pairs = (similarity >= cos_threshold) & dist_valid  # [n_curr, n_ref]

                # 4. 判断是否存在有效对
                if torch.any(valid_pairs):
                    return current_coord, current_vectors

            # 抑制当前峰值区域
            dist = torch.sqrt((y_grid - current_coord[0]) ** 2 + (x_grid - current_coord[1]) ** 2)
            mask = (dist <= 1.5).unsqueeze(0).unsqueeze(0)
            processed_map = processed_map.masked_fill(mask, 0)

    def suppress_target(
            self,
            score_map,
            target_coord,
            suppress_factor=1,
            neighbor_radius=2,
            base_boost=0.2
    ):
        h, w = score_map.shape[-2:]
        mask = torch.full((h, w), suppress_factor, device=score_map.device, dtype=torch.float32)

        y, x = int(target_coord[0]), int(target_coord[1])
        if 0 <= y < h and 0 <= x < w:
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                for dx in range(-neighbor_radius, neighbor_radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        dist = math.sqrt(dy ** 2 + dx ** 2)
                        if dist <= neighbor_radius:
                            gain = 1.0 + base_boost * (1 - dist / neighbor_radius)
                            mask[ny, nx] = gain

        weighted_map = score_map * mask.unsqueeze(0).unsqueeze(0)
        return weighted_map

    def suppress_non_target(
            self,
            score_map,
            target_coord,
            target_gain=0.3,
            suppress_factor=1,
            auto_boost=True,
            min_boost_margin=0.1  # 最小超出幅度（避免刚好等于 max_score）
    ):
        """
        自适应增强目标位置，确保其得分超过当前最大值
        :param min_boost_margin: 增强后目标得分至少超过 max_score 的幅度（默认 0.1）
        """
        h, w = score_map.shape[-2:]
        mask = torch.full((h, w), suppress_factor, device=score_map.device, dtype=torch.float32)

        y, x = int(target_coord[0]), int(target_coord[1])
        if 0 <= y < h and 0 <= x < w:
            target_score = score_map[0, 0, y, x]
            max_score = score_map.max()

            if auto_boost and target_score < max_score:
                # 计算最小需要的增益，使得 target_score * gain >= max_score + min_boost_margin
                required_gain = (max_score + min_boost_margin) / target_score
                # 结合用户指定的 target_gain，取较大值（确保至少增强到超过最大值）
                gain = max(1.0 + target_gain, required_gain)
            else:
                gain = 1.0  # 不增强

            mask[y, x] = gain

        weighted_map = score_map * mask.unsqueeze(0).unsqueeze(0)
        return weighted_map

    def forward(self, x, gt_score_map=None, return_topk_boxes=False,index=None, error_idx=None, vectors=None,target_to_distractors_vectors_forward=None):
        """ Forward pass with input x. """
        target_coord = None
        self.last_return = False
        if target_to_distractors_vectors_forward and target_to_distractors_vectors_forward[-1][1] == []:
            self.prev_target_coord = None
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        if self.prev_target_coord is not None:
            score_map_ctr = self.suppress_target(score_map_ctr, self.prev_target_coord)
        filtered_score_map = score_map_ctr.clone()
        if error_idx is not None and len(error_idx) > 0 and self.init_index is None:
            self.init_index = error_idx[-1][0]
        if error_idx is None:
            self.init_index = None
        if gt_score_map is None:
            if self.init_index is not None:
                handle_length = 20 - (self.init_index % 20)  # 计算需要处理的长度
                if self.init_index <= index <= self.init_index + handle_length:
                    if error_idx is not None and len(error_idx) > 0 and index == error_idx[-1][0]:
                        error_idx_element = error_idx.pop()
                        if error_idx_element[0] == self.init_index:
                            target_index = index - 1
                            for idx, elem in target_to_distractors_vectors_forward:
                                if idx == target_index:
                                    target_coord, valid_vectors = self.detect_peaks_with_validation(
                                        score_map_ctr, elem, angle_threshold=50
                                    )
                                    break
                        else:
                            target_coord, valid_vectors = self.detect_peaks_with_validation(
                                    score_map_ctr, target_to_distractors_vectors_forward[-1][1], angle_threshold=50
                                )
            elif target_to_distractors_vectors_forward :
                target_coord, valid_vectors = self.detect_peaks_with_validation(
                    score_map_ctr, target_to_distractors_vectors_forward[-1][1], angle_threshold=35
                )

            if target_coord is not None:
                self.prev_target_coord = target_coord
                print("processing", index, target_coord)
                filtered_score_map = self.suppress_non_target(score_map_ctr, target_coord)
            bbox = self.cal_bbox(filtered_score_map, size_map, offset_map)
            if return_topk_boxes:
                topkBbox = self.cal_topk_bbox(50, score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)
            if return_topk_boxes:
                topkBbox = self.cal_topk_bbox(50, gt_score_map.unsqueeze(1), size_map, offset_map)
        if not return_topk_boxes:
            topkBbox = None
        return filtered_score_map, bbox, size_map, offset_map, topkBbox ,self.last_return

    def cal_topk_bbox(self, k, score_map_ctr, size_map, offset_map):
        topScores, idx = score_map_ctr.flatten(2).topk(k, dim=2)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.expand(idx.shape[0], 2, k)
        #for i in range(k):
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        bboxes = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)
        return bboxes.permute(0, 2, 1)

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        #import pdb
        #pdb.set_trace()
        #idx_y = idx // self.feat_sz
        idx_y = torch.div(idx, self.feat_sz, rounding_mode="trunc")
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if cfg.MODEL.HEAD.TYPE == "MLP":
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)


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
