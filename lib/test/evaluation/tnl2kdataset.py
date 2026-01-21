import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)

        # Ground Truth
        anno_path = os.path.join(seq_path, 'groundtruth.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # Language / Caption
        caption = self._read_caption(seq_path)

        # Images
        frames_path = os.path.join(seq_path, 'imgs')
        # Robustly list and sort image files
        frames_list = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        object_meta={'caption': caption})

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if not os.path.exists(self.base_path):
            print(f"Warning: TNL2k path not found: {self.base_path}")
            return []

        sequence_list = []
        for seq in sorted(os.listdir(self.base_path)):
            if os.path.isdir(os.path.join(self.base_path, seq)):
                sequence_list.append(seq)

        return sequence_list

    def _read_caption(self, seq_path):
        language_file = os.path.join(seq_path, 'language.txt')
        if os.path.exists(language_file):
            try:
                with open(language_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.readline().strip()
                    if text:
                        return text
            except Exception:
                pass
        return "an object in the scene"
