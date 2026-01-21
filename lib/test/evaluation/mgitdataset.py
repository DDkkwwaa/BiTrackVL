import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import json


class MGITDataset(BaseDataset):
    """
    MGIT test set.
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.mgit_test_path
        self.info_path = self.env_settings.mgit_info_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # sequence_name is like "001", "006", etc.

        # 1. Construct Frames Path
        # MGIT-Test/{sequence_name}/frame_{sequence_name}/*.jpg
        # Example: MGIT-Test/007/frame_007/000000.jpg
        # Note: The folder structure seems to be MGIT-Test/{seq_id}/frame_{seq_id}/
        # But let's verify based on user description: "MGIT-Test点进去就是一个一个的数字命名的序列...每一个序列点进去都是frame_+命名序列的数字"
        # So path is: self.base_path / sequence_name / f"frame_{sequence_name}"

        seq_path = os.path.join(self.base_path, sequence_name)
        frame_folder_name = f"frame_{sequence_name}"
        frames_path = os.path.join(seq_path, frame_folder_name)

        if not os.path.isdir(frames_path):
            # Fallback or error handling if structure is slightly different
            # Try finding any folder starting with frame_
            if os.path.isdir(seq_path):
                subdirs = [d for d in os.listdir(seq_path) if
                           os.path.isdir(os.path.join(seq_path, d)) and d.startswith('frame_')]
                if subdirs:
                    frames_path = os.path.join(seq_path, subdirs[0])

        frames_list = sorted(
            [os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.lower().endswith('.jpg')])

        # 2. Construct GT Path
        # MGIT-Info/attribute/groundtruth/{sequence_name}.txt
        gt_path = os.path.join(self.info_path, 'attribute', 'groundtruth', f"{sequence_name}.txt")

        if not os.path.isfile(gt_path):
            # Try without leading zeros if not found, or handle potential mismatch
            # But user says "文件的数字就是对应着刚才数字命名的测试序列"
            pass

        ground_truth_rect = load_text(str(gt_path), delimiter=',', dtype=np.float64)

        # 3. Read Caption from JSON
        # MGIT-Info/attribute/description/{sequence_name}.json
        json_path = os.path.join(self.info_path, 'attribute', 'description', f"{sequence_name}.json")
        caption = self._read_caption_from_json(json_path)

        # Create Sequence object
        return Sequence(sequence_name, frames_list, 'mgit', ground_truth_rect.reshape(-1, 4),
                        object_class='mgit_object', target_visible=np.ones(ground_truth_rect.shape[0]),
                        object_meta={'caption': caption})

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seqs = []
        if not os.path.isdir(self.base_path):
            print(f"Warning: MGIT Test root not found: {self.base_path}")
            return []

        # List all directories in MGIT-Test
        # They are named "001", "006", etc.
        for seq_name in sorted(os.listdir(self.base_path)):
            seq_path = os.path.join(self.base_path, seq_name)
            if os.path.isdir(seq_path):
                # Verify it has corresponding GT
                gt_path = os.path.join(self.info_path, 'attribute', 'groundtruth', f"{seq_name}.txt")
                if os.path.isfile(gt_path):
                    seqs.append(seq_name)
        return seqs

    def _read_caption_from_json(self, json_path):
        if not os.path.isfile(json_path):
            return 'an object in the scene'

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # User logic: "有很多description，我感觉只需要读取最后一个end_frame的description就行，那个描述应该最详尽"
            # The JSON structure seems to be a dictionary with keys like "action_1", "action_2", ..., "activity", "story_1"
            # Or maybe a list of actions.
            # Based on the screenshot (图九), the root object has keys like "action_1", "action_2"... "action_11", "activity", "story_1".
            # Each action has "description".
            # "story_1" seems to cover the whole range (start_frame: 0, end_frame: 14295).
            # "story_1" description: "A green clothes actor meets a group people and has a fight..."

            # Strategy 1: Look for "story_1" or similar global summary.
            if "story_1" in data and "description" in data["story_1"]:
                return data["story_1"]["description"]

            # Strategy 2: If no story, find the action with the largest end_frame (as user suggested "最后一个end_frame").
            max_end_frame = -1
            best_desc = 'an object in the scene'

            for key, value in data.items():
                if isinstance(value, dict) and "end_frame" in value and "description" in value:
                    try:
                        end_frame = int(value["end_frame"])
                        if end_frame > max_end_frame:
                            max_end_frame = end_frame
                            best_desc = value["description"]
                    except:
                        continue

            return best_desc

        except Exception as e:
            print(f"Error reading caption from {json_path}: {e}")
            pass

        return 'an object in the scene'
