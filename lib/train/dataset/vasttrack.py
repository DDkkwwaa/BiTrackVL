import os
import torch
import random
import numpy as np
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin.environment import env_settings


class VastTrack(BaseVideoDataset):
    """ VastTrack training dataset wrapper for HIPTrack training pipeline.

    Assumptions (based on user specification):
    - Root directory structure:
        <root>/<class_name>/<sequence_name>/
            Groundtruth.txt
            imgs/00000001.jpg ...
            (optional) nlp.txt  (ignored for now)
    - Groundtruth.txt format: each line 'x,y,w,h' (comma separated). Lines with invalid w/h (<=0) are skipped and
      replaced with previous valid box (or first valid) to keep length consistency.
    - No official train/val split required now (entire dataset used for training). Validation can be another dataset.
    - Visibility flag not provided: we set visible = valid = (w>0 & h>0).
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None,
                 frame_dir_candidates=None, gt_name_candidates=None, verbose=False):
        root = env_settings().vasttrack_dir if root is None else root
        super().__init__('VastTrack', root, image_loader)

        # Candidate names for robustness
        self.frame_dir_candidates = frame_dir_candidates or [
            'imgs', 'img', 'images', 'image', 'JPEGImages'
        ]
        self.gt_name_candidates = gt_name_candidates or [
            'Groundtruth.txt', 'groundtruth.txt', 'GroundTruth.txt', 'groundTruth.txt'
        ]
        self.verbose = verbose

        # Build sequence list (all sequences under class folders) with auto-detection
        self.sequence_list = self._discover_sequences()
        # 预先读取所有 caption (nlp.txt 第一行)，缓存避免重复 IO
        self.captions = {}
        for rel in self.sequence_list:
            seq_path = os.path.join(self.root, rel)
            self.captions[rel] = self._read_caption(seq_path)

        if data_fraction is not None and 0 < data_fraction < 1:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _discover_sequences(self):
        seqs = []
        if not os.path.isdir(self.root):
            raise RuntimeError(f"VastTrack root not found: {self.root}")
        for cls_name in sorted(os.listdir(self.root)):
            cls_path = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for seq_name in sorted(os.listdir(cls_path)):
                seq_path = os.path.join(cls_path, seq_name)
                if not os.path.isdir(seq_path):
                    continue
                gt_file = self._find_gt_file(seq_path)
                frame_dir = self._find_frame_dir(seq_path)
                if gt_file and frame_dir:
                    seqs.append(f"{cls_name}/{seq_name}")
                elif self.verbose:
                    miss = []
                    if not gt_file:
                        miss.append('groundtruth')
                    if not frame_dir:
                        miss.append('frames')
                    print(f"[VastTrack] Skip sequence {cls_name}/{seq_name} (missing {','.join(miss)})")
        if len(seqs) == 0:
            raise RuntimeError(f"No valid VastTrack sequences found in {self.root}")
        return seqs

    def get_name(self):
        return 'vasttrack'

    def _sequence_path(self, seq_id):
        rel = self.sequence_list[seq_id]
        return os.path.join(self.root, rel)

    def _find_gt_file(self, seq_path):
        for cand in self.gt_name_candidates:
            p = os.path.join(seq_path, cand)
            if os.path.isfile(p):
                return p
        # fallback: scan for a txt file whose first non-empty line has 4 numeric parts
        for f in os.listdir(seq_path):
            if f.lower().endswith('.txt') and 'nlp' not in f.lower():
                p = os.path.join(seq_path, f)
                try:
                    with open(p, 'r') as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            parts = [pp for pp in line.replace('\t', ',').replace(' ', ',').split(',') if pp]
                            if len(parts) == 4:
                                # rough numeric check
                                _ = [float(pp) for pp in parts]
                                return p
                            break
                except Exception:
                    continue
        return None

    def _find_frame_dir(self, seq_path):
        for cand in self.frame_dir_candidates:
            p = os.path.join(seq_path, cand)
            if os.path.isdir(p):
                return cand
        return None

    def _read_bb_anno(self, seq_path):
        gt_file = self._find_gt_file(seq_path)
        if gt_file is None:
            raise RuntimeError(f"Groundtruth file not found under {seq_path}")
        boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.replace('\t', ',').replace(' ', ',').split(',') if p.strip()]
                if len(parts) != 4:
                    continue
                try:
                    x, y, w, h = map(float, parts)
                except ValueError:
                    continue
                if w <= 0 or h <= 0:
                    # invalid size; skip (will repair later)
                    boxes.append(None)
                else:
                    boxes.append([x, y, w, h])
        # repair Nones with nearest previous valid (fallback to first valid)
        last_valid = None
        for i, b in enumerate(boxes):
            if b is not None:
                last_valid = b
            else:
                if last_valid is None:
                    # find first future valid
                    future = next((fb for fb in boxes if fb is not None), [0.0, 0.0, 10.0, 10.0])
                    last_valid = future
                boxes[i] = last_valid
        if len(boxes) == 0:
            boxes = [[0.0, 0.0, 10.0, 10.0]]
        return torch.tensor(np.array(boxes), dtype=torch.float32)

    def _read_caption(self, seq_path):
        """读取序列的自然语言描述。
        规则:
          - 文件名固定 "nlp.txt" (你确认的格式)
          - 取第一行非空行作为 caption
          - 若不存在或空，用占位 'an object in the scene'
        """
        nlp_file = os.path.join(seq_path, 'nlp.txt')
        if not os.path.isfile(nlp_file):
            return 'an object in the scene'
        try:
            with open(nlp_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
        except Exception:
            pass
        return 'an object in the scene'

    def get_caption(self, seq_id):
        rel = self.sequence_list[seq_id]
        return self.captions.get(rel, 'an object in the scene')

    def get_sequence_info(self, seq_id):
        seq_path = self._sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone()  # no extra visibility meta available for now
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _frame_path(self, seq_path, frame_id):
        # 根据用户反馈：文件名格式为 00001.jpg 5 位零填充，从 1 开始
        frame_dir = self._find_frame_dir(seq_path)
        if frame_dir is None:
            raise RuntimeError(f"Frame directory not found in {seq_path}")
        base_id = frame_id + 1  # 内部使用 0-based -> 文件 1-based
        name_5 = f"{base_id:05d}.jpg"
        path_5 = os.path.join(seq_path, frame_dir, name_5)
        if os.path.isfile(path_5):
            return path_5
        # 回退尝试 8 位（兼容之前假设）
        name_8 = f"{base_id:08d}.jpg"
        path_8 = os.path.join(seq_path, frame_dir, name_8)
        if os.path.isfile(path_8):
            return path_8
        # 再回退尝试不补零
        name_plain = f"{base_id}.jpg"
        path_plain = os.path.join(seq_path, frame_dir, name_plain)
        if os.path.isfile(path_plain):
            return path_plain
        raise FileNotFoundError(f"Frame not found (tried 5/8/no pad): {path_5} | {path_8} | {path_plain}")

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._sequence_path(seq_id)
        cls_name = seq_path.split(os.sep)[-2]
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: [v[f_id, ...].clone() if torch.is_tensor(v) else v[f_id]
                           for f_id in frame_ids] for k, v in anno.items()}
        object_meta = OrderedDict({'object_class_name': cls_name,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'caption': self.get_caption(seq_id)})
        return frame_list, anno_frames, object_meta
