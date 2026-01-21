from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import numpy as np
import cv2
import torch.nn.functional as F
class HIPTrackActor(BaseActor):
    """ Actor for training HIPTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None, multiFrame=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multiFrame = multiFrame
        self.epoch = 0
        self.ce_keep_rate = 1.0

    def compute_frame_consistency(self, idx, fwd_score, bwd_score, first_frame_gt):
        """
        精确到帧级的计算逻辑
        idx: 当前帧索引 (0到seq_len-1)
        fwd_score: 当前帧前向响应图
        bwd_score: 当前帧后向响应图
        first_frame_gt: 第一帧真实响应图
        """
        loss = 0

        # === 基础一致性损失 (仅在idx=0计算) ===
        if idx == 0:
            # 后向重建的第一帧响应图
            reconstructed_init = bwd_score
            # 与原始第一帧真值比较
            base_loss = F.mse_loss(reconstructed_init, first_frame_gt)
            loss += base_loss

        # === 帧间一致性损失 (从idx=1开始) ===
        if idx >= 1:
            # 1. 前向帧间一致性 (当前帧与前帧)
            fwd_consist = F.mse_loss(fwd_score, self.prev_fwd_score)

            # 2. 后向帧间一致性 (当前帧与前帧)
            bwd_consist = F.mse_loss(bwd_score, self.prev_bwd_score)

            # 3. 加权组合
            loss += 0.3 * (fwd_consist + bwd_consist)

        # 保存当前状态供下一帧使用
        self.prev_fwd_score = fwd_score.detach()
        self.prev_bwd_score = bwd_score.detach()

        return loss

    def compute_motion_weight(self, current_boxes, prev_boxes, img_size):
        """
        计算基于运动的权重系数
        公式: w = 1 + λ * displacement (论文中的公式9)
        """

        # 1. 转换为中心点坐标
        def boxes_to_centers(boxes):
            # 输入: [batch, 4] (x1, y1, w, h)
            # 返回中心点坐标 (x_center, y_center)
            x = boxes[:, 0] + boxes[:, 2] / 2
            y = boxes[:, 1] + boxes[:, 3] / 2
            return torch.stack((x, y), dim=1)

        # 2. 计算位移量
        with torch.no_grad():
            current_centers = boxes_to_centers(current_boxes)
            prev_centers = boxes_to_centers(prev_boxes)

            # 计算欧氏距离
            displacement = torch.norm(current_centers - prev_centers, dim=1)

            # 归一化位移（相对于图像尺寸）
            normalized_disp = displacement / img_size

            # 应用公式
            lambda_motion = self.cfg.TRAIN.LAMBDA
            weights = 1.0 + lambda_motion * normalized_disp

            # 防止极端值
            weights = torch.clamp(weights, min=0.5, max=3.0)

        return weights.unsqueeze(1)  # 保持维度一致性

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        fwd_search_bbox_in_crop, bwd_search_bbox_in_crop, fwd_out_list, bwd_out_list = self.forward_pass(data)

        if isinstance(fwd_out_list, list):
            losses = None
            statuses = {
                "Loss/total": 0.0,
                "Loss/giou": 0.0,
                "Loss/l1": 0.0,
                "Loss/location": 0.0,
                "Loss/iou_pred": 0.0,
                "IoU": 0.0,
                "BWD_Loss/total": 0.0,
                "BWD_Loss/giou": 0.0,
                "BWD_Loss/l1": 0.0,
                "BWD_Loss/location": 0.0,
                "BWD_IoU": 0.0,
                # 新增一致性相关状态
                "Loss/consistency": 0.0,
                # 新增主体词相关状态
                "Loss/subject": 0.0,
                "Metric/subject_ratio": 0.0,
                "Metric/write_weight": 0.0,
            }

            # 初始化历史状态
            self.prev_fwd_score = None
            self.prev_bwd_score = None
            self.last_fwd_boxes = None
            self.last_bwd_boxes = None
            all_consistency_losses = []
            frame_indices = []

            num_frames = len(fwd_out_list)
            for idx, (fwd_out, bwd_out) in enumerate(zip(fwd_out_list, bwd_out_list)):

                # 准备帧数据
                partData = {"search_anno": data['search_anno'].permute(1, 0, 2)[idx].unsqueeze(0)}
                bwd_partData_1 = {"search_anno": data['search_anno'].permute(1, 0, 2).flip(0)[idx].unsqueeze(0)}
                fwd_partData = {"search_anno": fwd_search_bbox_in_crop[idx].unsqueeze(0)}
                bwd_partData = {"search_anno": bwd_search_bbox_in_crop[idx].unsqueeze(0)}

                # 计算损失（包含新增的一致性损失）
                loss, status = self.compute_losses(
                    fwd_out, partData, fwd_partData,
                    bwd_out, bwd_partData_1, bwd_partData,
                    data, idx
                )

                # 收集一致性损失用于丢弃机制
                if "Loss/consistency" in status:
                    all_consistency_losses.append(status["Loss/consistency"])
                    frame_indices.append(idx)

                # 累加损失和状态
                losses = loss if losses is None else losses + loss
                for key in status:
                    if key in statuses:
                        statuses[key] += status[key]
                    # 确保新添加的 Metric 也能被累加，即使它们不在初始 statuses 中（虽然我们已经加了）
                    elif key.startswith("Metric/") or key == "Loss/subject":
                         statuses[key] = status[key] if key not in statuses else statuses[key] + status[key]

            # ========== 平均状态信息 ==========
            statuses['IoU'] /= num_frames
            statuses['BWD_IoU'] /= num_frames
            statuses['Loss/consistency'] /= num_frames
            statuses['Loss/subject'] /= num_frames
            statuses['Metric/subject_ratio'] /= num_frames
            statuses['Metric/write_weight'] /= num_frames
            # statuses['Loss/weighted_consistency'] /= num_frames
            # statuses['Consistency/motion_weight'] /= num_frames

            return losses, statuses

        # fwd_search_bbox_in_crop, bwd_search_bbox_in_crop,fwd_out_list, bwd_out_list = self.forward_pass(data)
        # # 打印前向和后向的第一帧GT框和预测框
        # if isinstance(fwd_out_list, list):
        #
        #     losses = None
        #     statuses = {
        #         "Loss/total": 0.0,
        #         "Loss/giou": 0.0,
        #         "Loss/l1": 0.0,
        #         "Loss/location": 0.0,
        #         "Loss/iou_pred": 0.0,
        #         "IoU": 0.0,
        #         "BWD_Loss/total": 0.0,
        #         "BWD_Loss/giou": 0.0,
        #         "BWD_Loss/l1": 0.0,
        #         "BWD_Loss/location": 0.0,
        #         "BWD_IoU": 0.0
        #     }
        #     num_frames = len(fwd_out_list)
        #     for idx, (fwd_out, bwd_out) in enumerate(zip(fwd_out_list, bwd_out_list)):
        #         # 使用当前帧索引提取标签
        #         partData = {"search_anno":data['search_anno'].permute(1, 0, 2)[idx].unsqueeze(0)}
        #         bwd_partData_1 = {"search_anno": data['search_anno'].permute(1, 0, 2).flip(0)[idx].unsqueeze(0)}
        #
        #         fwd_partData = {"search_anno": fwd_search_bbox_in_crop[idx].unsqueeze(0)}
        #         bwd_partData= {"search_anno": bwd_search_bbox_in_crop[idx].unsqueeze(0)}
        #         # 计算前向损失
        #         loss, status = self.compute_losses(fwd_out, partData, fwd_partData, bwd_out, bwd_partData_1, bwd_partData, data)
        #         # 计算后向损失
        #         total =loss
        #         losses = total if losses is None else losses + total
        #
        #         # 累加状态信息
        #         for key in status:
        #             statuses[key] += status[key]
        #
        #     statuses['IoU'] /= num_frames
        #     statuses['BWD_IoU'] /= num_frames
        #
        #     # 若需平均总损失，可添加
        #     losses = losses
        #
        #     return losses, statuses


    def deNorm(self, image):
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        cv2.imwrite("imgDeNorm.jpg", img=img[:,:,::-1])
        return img

    def forward_pass(self, data):

        if self.epoch != data['epoch']:
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            self.ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # Extract captions if available (for VLT)
        captions = None
        if 'object_meta' in data:
            meta = data['object_meta']
            # Case 1: meta is a dict (batch collation of dicts -> dict of lists/tensors)
            if isinstance(meta, dict):
                # Try 'caption' first (VastTrack), then 'nlp' (LaSOT)
                cap_val = meta.get('caption', None)
                if cap_val is None:
                    cap_val = meta.get('nlp', None)
                
                if cap_val is not None:
                    if isinstance(cap_val, (list, tuple)):
                        captions = list(cap_val)
                    elif isinstance(cap_val, str):
                        captions = [cap_val]
            # Case 2: meta is a list (batch collation of OrderedDicts -> list of OrderedDicts)
            # Note: PyTorch default_collate usually converts list of dicts to dict of lists, 
            # but custom collate might behave differently.
            elif isinstance(meta, list) and len(meta) > 0:
                # Try to extract caption from each element
                try:
                    extracted = []
                    for m in meta:
                        if isinstance(m, dict):
                            val = m.get('caption')
                            if val is None:
                                val = m.get('nlp')
                            extracted.append(val)
                        else:
                            extracted.append(None)
                    # If any caption found, use it
                    if any(c is not None for c in extracted):
                        captions = extracted
                except:
                    pass

        # Fallback: direct caption key
        if captions is None:
             captions = data.get('caption', None)
             if isinstance(captions, str):
                 captions = [captions]
             elif isinstance(captions, tuple):
                 captions = list(captions)
        
        # Explicit Text Encoding & Logging (Restored from VLT)
        text_outputs = None
        # Handle DDP
        net_module = self.net.module if hasattr(self.net, 'module') else self.net
        
        # Check STAGE
        stage = self.cfg.TRAIN.get('STAGE', 'stage2') if self.cfg else 'stage2'

        if stage == 'stage2' and captions and hasattr(net_module, 'text_encoder') and net_module.text_encoder is not None:
            try:
                # Sanitize captions: ensure all are strings and handle None
                if isinstance(captions, (list, tuple)):
                    captions = [str(c) if c is not None else "" for c in captions]

                # Use a tensor from data to get device
                # data['template_images'] is a list of tensors
                # [Fix] Use model parameters to determine device safely, avoiding data structure mismatch
                device = next(net_module.parameters()).device
                text_outputs = net_module.text_encoder(captions, device=device,
                                                     fp16=bool(net_module.text_cfg.get('FP16', False)))
                if text_outputs and 'embeddings' in text_outputs:
                    emb_shape = tuple(text_outputs['embeddings'].shape)
                    if not hasattr(self.settings, '_text_log_printed') or not self.settings._text_log_printed:
                        # Safe printing for list of captions
                        cap_print = captions[0][:60] if isinstance(captions, list) and len(captions)>0 else str(captions)[:60]
                        print(f"[Text] Caption 加载成功: '{cap_print}' | embeddings shape={emb_shape}")
                        self.settings._text_log_printed = True
            except Exception as e:
                if not hasattr(self.settings, '_text_log_printed_err'):
                    print(f"[Text][Warning] 编码 caption 失败: {e}")
                    self.settings._text_log_printed_err = True
        else:
             # Debug print if captions failed
             if not hasattr(self.settings, '_text_debug_printed'):
                 print(f"[Text][Debug] Captions extraction failed. data keys: {list(data.keys())}")
                 if 'object_meta' in data:
                     print(f"[Text][Debug] object_meta type: {type(data['object_meta'])}")
                     if isinstance(data['object_meta'], dict):
                         print(f"[Text][Debug] object_meta keys: {list(data['object_meta'].keys())}")
                         if 'caption' in data['object_meta']:
                             print(f"[Text][Debug] caption type: {type(data['object_meta']['caption'])}")
                             print(f"[Text][Debug] caption val: {data['object_meta']['caption']}")
                 self.settings._text_debug_printed = True

        out_dict = self.net.forward(data, self.ce_keep_rate, return_last_attn=False, captions=captions, text_outputs=text_outputs)  # gt [15, B, 4], pred [B, 1, 4]

        # self.visualizeCE(out_dict['removed_indexes_s'], out_dict['pred_boxes'].squeeze(1), data['search_images'][0], data['search_anno'][0])
        return out_dict

    def visualizeCE(self, ceMasks, predBoxes, imgs, gtBoxes):
        import pdb
        pdb.set_trace()
        for i in range(16):
            mask = np.ones((24, 24), dtype=np.uint8)
            img = imgs[i]
            img = self.deNorm(img)
            ce1 = ceMasks[0][i]
            ce2 = ceMasks[1][i]
            ce3 = ceMasks[2][i]
            ce = torch.cat([ce1, ce2, ce3], axis=0)
            for num in ce:
                x = int(num) // 24
                y = int(num) % 24
                mask[x][y] = 0
            box = (box_cxcywh_to_xyxy((predBoxes[i])) * 384).int()
            gtBox = (box_xywh_to_xyxy(gtBoxes[i]) * 384).int()
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            mask = np.stack([mask, mask, mask], axis=2) * 255
            mask = cv2.resize(mask, (384, 384))
            cv2.rectangle(mask, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
            cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
            cv2.rectangle(mask, (gtBox[0].item(), gtBox[1].item()), (gtBox[2].item(), gtBox[3].item()), (0, 255, 0), 2)
            cv2.rectangle(img, (gtBox[0].item(), gtBox[1].item()), (gtBox[2].item(), gtBox[3].item()), (0, 255, 0), 2)
            cv2.imwrite(f"maskCE_{i}.jpg", mask[:,:,::-1])
            cv2.imwrite(f"img_{i}.jpg", img[:,:,::-1])

    def compute_bwd_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        # import pdb
        # pdb.set_trace()
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:

            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:

            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        if 'score_map' in pred_dict:

            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:

            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # box_head layer2
        loss = (
                self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss
        )

        if return_status:
            # status for log

            mean_iou = iou.detach().mean()
            status = {"BWD_Loss/total": loss.item(),
                      "BWD_Loss/giou": giou_loss.item(),
                      "BWD_Loss/l1": l1_loss.item(),
                      "BWD_Loss/location": location_loss.item(),
                      "BWD_IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_losses(self, fwd_pred_dict, fwd_gt_dict, fwd, bwd_pred_dict, bwd_gt_dict, bwd, data, idx, return_status=True, L1_ON_NORMALIZED=True):
        # gt gaussian map
        # import pdb
        # pdb.set_trace()
        fwd_gt_bbox = fwd_gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        fwd_gt_gaussian_maps = generate_heatmap(fwd['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        fwd_gt_gaussian_maps = fwd_gt_gaussian_maps[-1].unsqueeze(1)

        bwd_gt_bbox = bwd_gt_dict['search_anno'][-1]
        bwd_gt_gaussian_maps = generate_heatmap(bwd['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        bwd_gt_gaussian_maps = bwd_gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        bwd_pred_boxes = bwd_pred_dict['last_pred_anno_gt']
        fwd_pred_boxes = fwd_pred_dict['last_pred_anno_gt']
        if torch.isnan(fwd_pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = fwd_pred_boxes.size(1)
        bwd_pred_boxes_vec = box_xywh_to_xyxy(bwd_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        bwd_gt_boxes_vec = box_xywh_to_xyxy(bwd_gt_bbox).cuda()

        fwd_pred_boxes_vec = box_xywh_to_xyxy(fwd_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        fwd_gt_boxes_vec = box_xywh_to_xyxy(fwd_gt_bbox).cuda()

        if L1_ON_NORMALIZED:
            img_size = self.cfg.DATA.SEARCH.SIZE
            bwd_pred_boxes_norm = bwd_pred_boxes_vec.clone()
            bwd_gt_boxes_norm = bwd_gt_boxes_vec.clone()

            fwd_pred_boxes_norm = fwd_pred_boxes_vec.clone()
            fwd_gt_boxes_norm = fwd_gt_boxes_vec.clone()

            # 归一化坐标（x轴除宽，y轴除高）
            bwd_pred_boxes_norm[:, [0, 2]] /= img_size
            bwd_pred_boxes_norm[:, [1, 3]] /= img_size
            bwd_gt_boxes_norm[:, [0, 2]] /= img_size
            bwd_gt_boxes_norm[:, [1, 3]] /= img_size

            fwd_pred_boxes_norm[:, [0, 2]] /= img_size
            fwd_pred_boxes_norm[:, [1, 3]] /= img_size
            fwd_gt_boxes_norm[:, [0, 2]] /= img_size
            fwd_gt_boxes_norm[:, [1, 3]] /= img_size

            fwd_l1_loss = self.objective['l1'](fwd_pred_boxes_norm, fwd_gt_boxes_norm)
            bwd_l1_loss = self.objective['l1'](bwd_pred_boxes_norm, bwd_gt_boxes_norm)
        # compute giou and iou
        try:

            fwd_giou_loss, fwd_iou = self.objective['giou'](fwd_pred_boxes_vec, fwd_gt_boxes_vec.cuda())  # (BN,4) (BN,4)
            bwd_giou_loss, bwd_iou = self.objective['giou'](bwd_pred_boxes_vec,
                                                            bwd_gt_boxes_vec.cuda())  # (BN,4) (BN,4)
            consistency_giou_loss, consistency_iou = self.objective['giou'](fwd_pred_boxes_vec, bwd_pred_boxes_vec)
        except:
            bwd_giou_loss, bwd_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            fwd_giou_loss, fwd_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            consistency_giou_loss, consistency_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
        if 'score_map' in fwd_pred_dict:
            bwd_location_loss = self.objective['focal'](bwd_pred_dict['score_map'], bwd_gt_gaussian_maps)
            fwd_location_loss = self.objective['focal'](fwd_pred_dict['score_map'], fwd_gt_gaussian_maps)
        else:
            bwd_location_loss = torch.tensor(0.0, device=bwd_l1_loss.device)
            fwd_location_loss = torch.tensor(0.0, device=fwd_l1_loss.device)

        if 'pred_iou' in fwd_pred_dict:
            pred_iou = fwd_pred_dict['pred_iou'].view(-1)
            gt_iou = fwd_iou.detach()
            iou_pred_loss = self.objective['iou_pred'](pred_iou, gt_iou)

        loss = (
                self.loss_weight['giou'] * fwd_giou_loss +
                self.loss_weight['l1'] * fwd_l1_loss +
                self.loss_weight['focal'] * fwd_location_loss +
                self.loss_weight['giou'] * bwd_giou_loss +
                self.loss_weight['l1'] * bwd_l1_loss +
                self.loss_weight['focal'] * bwd_location_loss
        )
        loss = loss + self.loss_weight['iou_pred'] * iou_pred_loss

        # motion_weight = torch.tensor(1.0, device=loss.device)  # 默认值为1.0

        # 计算基础一致性损失
        base_loss = consistency_giou_loss * 0.5
        consistency_loss = base_loss

        # 帧间一致性损失
        frame_consist_loss = torch.tensor(0.0, device=loss.device)
        if idx >= 1 and self.last_fwd_boxes is not None and self.last_bwd_boxes is not None:

            fwd_consist_loss, fwd_consist_iou = self.objective['giou'](fwd_pred_boxes_vec, self.last_fwd_boxes)

            bwd_consist_loss, bwd_consist_iou = self.objective['giou'](bwd_pred_boxes_vec, self.last_bwd_boxes)
            frame_consist_loss = fwd_consist_loss * 0.3 + bwd_consist_loss * 0.7

            # 累加帧间一致性到总一致性损失
        consistency_loss = consistency_loss + 0.3 * frame_consist_loss

        self.last_fwd_boxes = fwd_pred_boxes_vec
        self.last_bwd_boxes = bwd_pred_boxes_vec

        loss = loss + self.loss_weight['consistency'] * consistency_loss

        # Subject Loss
        loss_subject = torch.tensor(0.0, device=loss.device)
        subject_mask_pred = fwd_pred_dict.get('subject_mask_pred')
        attn_weights = fwd_pred_dict.get('text_attn_weights')
        
        if subject_mask_pred is not None and attn_weights is not None:
            # attn_weights: [B, N_search, N_text]
            # Average over search tokens -> [B, N_text]
            attn_mean = attn_weights.mean(dim=1)
            
            # N_text might be 2*L (due to f_XL = [f_L_T; f_L_C])
            # We only care about the first half (f_L_T) which corresponds to text tokens
            L_text = subject_mask_pred.shape[1]
            if attn_mean.shape[1] >= L_text:
                attn_score = attn_mean[:, :L_text]
            else:
                attn_score = attn_mean
            
            # Top-K pseudo labels
            k = 3 
            # Get top-k indices
            # Fix: Detach attn_score to prevent gradient flow from pseudo-labels back to attention weights
            # This breaks the positive feedback loop that causes training collapse
            attn_score_detached = attn_score.detach()
            _, topk_indices = torch.topk(attn_score_detached, k=min(k, attn_score.shape[1]), dim=1)
            pseudo_mask = torch.zeros_like(subject_mask_pred)
            pseudo_mask.scatter_(1, topk_indices, 1.0)
            
            # BCE Loss
            loss_subject = F.binary_cross_entropy(subject_mask_pred, pseudo_mask)
            
            loss = loss + self.loss_weight.get('subject', 0.2) * loss_subject

        loss = loss.mean()
        if return_status:
            # status for log
            fwd_mean_iou = fwd_iou.detach().mean()
            bwd_mean_iou = bwd_iou.detach().mean()
            
            # Metrics
            subject_ratio = 0.0
            if subject_mask_pred is not None:
                subject_ratio = (subject_mask_pred > 0.5).float().mean().item()
                
            write_weight_val = 0.0
            w_t = fwd_pred_dict.get('write_weight')
            if w_t is not None:
                write_weight_val = w_t.mean().item()

            status = {"Loss/total": loss.item(),
                      "Loss/giou": fwd_giou_loss.item(),
                      "Loss/l1": fwd_l1_loss.item(),
                      "Loss/location": fwd_location_loss.item(),
                      "Loss/iou_pred": iou_pred_loss.item(),
                      "Loss/subject": loss_subject.item(),
                      "Metric/subject_ratio": subject_ratio,
                      "Metric/write_weight": write_weight_val,
                      "IoU": fwd_mean_iou.item(),
                      "BWD_Loss/total": loss.item(),
                      "BWD_Loss/giou": bwd_giou_loss.item(),
                      "BWD_Loss/l1": bwd_l1_loss.item(),
                      "BWD_Loss/location": bwd_location_loss.item(),
                      "BWD_IoU": bwd_mean_iou.item(),
                      "Loss/consistency": consistency_loss.mean().item(),
                      # # "Loss/weighted_consistency": weighted_consistency_loss.mean().item(),
                      # "Consistency/motion_weight": motion_weight.mean().item() if not isinstance(motion_weight,
                      #                                                                            float) else 1.0,
                      }
            return loss, status
        else:
            return loss


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