import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VastTrackDataset(BaseDataset):
    """
    VastTrack test set.
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vasttrack_dir
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_info):
        # sequence_info is (class_name, sequence_name, relative_path)
        class_name, sequence_name, rel_path = sequence_info
        seq_path = os.path.join(self.base_path, rel_path)

        # Find GT
        gt_path = self._find_gt_file(seq_path)
        ground_truth_rect = load_text(str(gt_path), delimiter=',', dtype=np.float64)

        # Find Frames
        frame_dir = self._find_frame_dir(seq_path)
        frames_path = os.path.join(seq_path, frame_dir)

        # Construct frame list
        # We list all jpg files and sort them.
        # Note: vasttrack.py has complex logic for 5-digit vs 8-digit vs no-pad.
        # Here we assume they are sortable by name.
        frames_list = sorted(
            [os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.lower().endswith('.jpg')])

        # Read Caption
        caption = self._read_caption(seq_path)

        # Create Sequence object
        # Note: We pass caption in object_meta
        return Sequence(sequence_name, frames_list, 'vasttrack', ground_truth_rect.reshape(-1, 4),
                        object_class=class_name, target_visible=np.ones(ground_truth_rect.shape[0]),
                        object_meta={'caption': caption})

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seqs = []
        if not os.path.isdir(self.base_path):
            # If not found, return empty or raise error? BaseDataset usually assumes path exists.
            # But for safety let's return empty list or print warning.
            print(f"Warning: VastTrack root not found: {self.base_path}")
            return []

        for cls_name in sorted(os.listdir(self.base_path)):
            cls_path = os.path.join(self.base_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for seq_name in sorted(os.listdir(cls_path)):
                seq_path = os.path.join(cls_path, seq_name)
                if not os.path.isdir(seq_path):
                    continue
                # Check if valid
                if self._find_gt_file(seq_path) and self._find_frame_dir(seq_path):
                    seqs.append((cls_name, seq_name, f"{cls_name}/{seq_name}"))
        return seqs

    def _find_gt_file(self, seq_path):
        candidates = ['Groundtruth.txt', 'groundtruth.txt', 'GroundTruth.txt', 'groundTruth.txt']
        for cand in candidates:
            p = os.path.join(seq_path, cand)
            if os.path.isfile(p):
                return p
        return None

    def _find_frame_dir(self, seq_path):
        candidates = ['imgs', 'img', 'images', 'image', 'JPEGImages']
        for cand in candidates:
            p = os.path.join(seq_path, cand)
            if os.path.isdir(p):
                return cand
        return None

    def _read_caption(self, seq_path):
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
