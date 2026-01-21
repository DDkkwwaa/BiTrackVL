import importlib
import os
from collections import OrderedDict

import torch

from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)
        self.re_track_count = 0
        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.error_idx = None
        self.vctors = None
        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        #import pdb; pdb.set_trace()
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def safe_to_list(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return x.tolist()
        return x

    # def _track_sequence(self, tracker, seq, init_info):
    #     total_frames = len(seq.frames)  # 总帧数（如80）
    #
    #     # 预初始化输出结构
    #     output = {
    #         'target_bbox': [None] * total_frames,
    #         'time': [None] * total_frames
    #     }
    #     if tracker.params.save_all_boxes:
    #         output['all_boxes'] = [None] * total_frames
    #         output['all_scores'] = [None] * total_frames
    #
    #     def _store_outputs(frame_idx, tracker_out: dict, defaults=None):
    #         """按帧号覆盖存储结果"""
    #         defaults = defaults or {}
    #         for key in output:
    #             val = tracker_out.get(key, defaults.get(key))
    #             if val is not None:
    #                 output[key][frame_idx] = val
    #
    #     # 初始化第0帧
    #     image = self._read_image(seq.frames[0])
    #     start_time = time.time()
    #     out = tracker.initialize(image, init_info, seq.name)
    #     if out is None:
    #         out = {}
    #     prev_output = OrderedDict(out)
    #     prev_gt_box = np.array(init_info['init_bbox'])
    #
    #     # 存储第0帧结果到索引0
    #     init_default = {
    #         'target_bbox': init_info.get('init_bbox'),
    #         'time': time.time() - start_time
    #     }
    #     if tracker.params.save_all_boxes:
    #         init_default['all_boxes'] = out.get('all_boxes')
    #         init_default['all_scores'] = out.get('all_scores')
    #     _store_outputs(0, out, init_default)
    #
    #     # 主循环处理1~79帧
    #     frame_num = 1
    #     max_retries = 3
    #     retry_count = 0
    #
    #     while frame_num < total_frames:
    #         image = self._read_image(seq.frames[frame_num])
    #         start_time = time.time()
    #         info = seq.frame_info(frame_num)
    #         info['previous_output'] = prev_output
    #
    #         # 添加当前帧的GT（如果存在）
    #         if frame_num < len(seq.ground_truth_rect):
    #             info['gt_bbox'] = seq.ground_truth_rect[frame_num]
    #
    #         # 跟踪当前帧
    #         out = tracker.track(image, self.error_idx, info)
    #         out_o = {'target_bbox': out.get('target_bbox')}
    #         self.error_idx = out['needy_rectify_info'][1]
    #
    #         # 处理回溯
    #         reinit_frame = out['needy_rectify_info'][0]
    #         if reinit_frame is not None:
    #             if 0 <= reinit_frame < total_frames and retry_count < max_retries:
    #                 # 跳转到回溯帧的前一帧（后续frame_num+1会指向回溯帧）
    #                 frame_num = reinit_frame - 1
    #                 retry_count += 1
    #             else:
    #                 # 终止或重置
    #                 break
    #         else:
    #             retry_count = 0  # 重置计数器
    #
    #         # 存储当前帧结果（覆盖旧值）
    #         _store_outputs(frame_num, out_o, {'time': time.time() - start_time})
    #         prev_output = OrderedDict(out_o)
    #         frame_num += 1
    #
    #     # 清理未启用的键
    #     for key in ['all_boxes', 'all_scores']:
    #         if key in output and all(x is None for x in output[key]):
    #             del output[key]
    #
    #     return output

    # def _track_sequence(self, tracker, seq, init_info):
    #     total_frames = len(seq.frames)  # 总帧数（如80）
    #     # output = {'target_bbox': [],
    #     #           'time': []}
    #     # if tracker.params.save_all_boxes:
    #     #     output['all_boxes'] = []
    #     #     output['all_scores'] = []
    #     output = {
    #         'target_bbox': [None] * total_frames,
    #         'time': [None] * total_frames
    #     }
    #     if tracker.params.save_all_boxes:
    #         output['all_boxes'] = [None] * total_frames
    #         output['all_scores'] = [None] * total_frames
    #
    #     # def _store_outputs(tracker_out: dict, defaults=None):
    #     #     defaults = {} if defaults is None else defaults
    #     #     for key in output.keys():
    #     #         val = tracker_out.get(key, defaults.get(key, None))
    #     #         if key in tracker_out or val is not None:
    #     #             output[key].append(val)
    #     def _store_outputs(frame_idx, tracker_out: dict, defaults=None):
    #         """按帧号覆盖存储结果"""
    #         defaults = defaults or {}
    #         for key in output:
    #             val = tracker_out.get(key, defaults.get(key))
    #             if val is not None:
    #                 output[key][frame_idx] = val
    #
    #     image = self._read_image(seq.frames[0])
    #
    #     start_time = time.time()
    #
    #     out = tracker.initialize(image, init_info, seq.name)
    #     if out is None:
    #         out = {}
    #
    #     prev_output = OrderedDict(out)
    #     prev_gt_box = np.array(init_info['init_bbox'])
    #     init_default = {'target_bbox': init_info.get('init_bbox'),
    #                     'time': time.time() - start_time}
    #     if tracker.params.save_all_boxes:
    #         init_default['all_boxes'] = out['all_boxes']
    #         init_default['all_scores'] = out['all_scores']
    #
    #     _store_outputs(0, out, init_default)
    #
    #     frame_num = 1
    #     total_num = len(seq.frames)
    #     count = 0
    #     while frame_num < total_num:
    #
    #         image = self._read_image(seq.frames[frame_num])
    #
    #         start_time = time.time()
    #
    #         info = seq.frame_info(frame_num)
    #         info['previous_output'] = prev_output
    #
    #         if len(seq.ground_truth_rect) > 1:
    #             info['gt_bbox'] = seq.ground_truth_rect[frame_num]  # 当前帧的gt
    #         out = tracker.track(image, self.error_idx, info)
    #         out_o = {'target_bbox': out['target_bbox']}
    #         if self.error_idx is None:
    #             if out['needy_rectify_info'] is not None:
    #                 self.error_idx = out['needy_rectify_info']
    #                 count = len(self.error_idx)
    #         else:
    #             count = len(self.error_idx)
    #             if count == 0:
    #                 self.error_idx = None
    #         if out['needy_rectify_info']!= None:
    #             out_o = tracker.reinitialize(out['needy_rectify_info'][len(out['needy_rectify_info'])-1][0], out['needy_rectify_info'][len(out['needy_rectify_info'])-1][3],
    #                                          self._read_image(seq.frames[out['needy_rectify_info'][len(out['needy_rectify_info'])-1][0]]), init_info, seq.name)
    #             self.re_track_count += 1
    #             if self.re_track_count >= 2:
    #                 break
    #             if out_o is None:
    #                 out_o = {}
    #             frame_num = out['needy_rectify_info'][len(out['needy_rectify_info'])-1][0]
    #             continue
    #
    #         prev_output = OrderedDict(out_o)
    #         print("output['time'] =", output.get('time', 'Missing'))
    #
    #         _store_outputs(frame_num, out_o, {'time': time.time() - start_time})
    #         # _store_outputs(out_o, {'time': time.time() - start_time})
    #         frame_num += 1
    #
    #     for key in ['target_bbox', 'all_boxes', 'all_scores']:
    #         if key in output and len(output[key]) <= 1:
    #             output.pop(key)
    #
    #     return output
    def _track_sequence(self, tracker, seq, init_info):
        total_frames = len(seq.frames)
        output = {
            'target_bbox': [None] * total_frames,
            'time': [None] * total_frames
        }
        if tracker.params.save_all_boxes:
            output['all_boxes'] = [None] * total_frames
            output['all_scores'] = [None] * total_frames

        def _store_outputs(frame_idx, tracker_out: dict, defaults=None):
            defaults = defaults or {}
            for key in output:
                val = tracker_out.get(key, defaults.get(key))
                if val is not None:
                    output[key][frame_idx] = val

        image = self._read_image(seq.frames[0])
        start_time = time.time()

        out = tracker.initialize(image, init_info, seq.name)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'), 'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out.get('all_boxes')
            init_default['all_scores'] = out.get('all_scores')

        _store_outputs(0, out, init_default)

        frame_num = 1
        count = 0

        while frame_num < total_frames:
            # print(f"\n[Track] Processing frame {frame_num}/{total_frames}")
            image = self._read_image(seq.frames[frame_num])
            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]

            out = tracker.track(image, self.error_idx, self.vctors, info)

            if out is None:
                # print(f"[Warning] tracker.track returned None at frame {frame_num}")
                frame_num += 1
                continue

            # 默认输出
            out_o = {'target_bbox': out.get('target_bbox')}

            # 更新 error_idx 状态
            if self.error_idx is None:
                if out.get('needy_rectify_info') is not None:
                    self.error_idx = out['needy_rectify_info']
                    self.vctors = out['vectors']
                    count = len(self.error_idx)
            else:
                count = len(self.error_idx)
                if count == 0:
                    self.error_idx = None

            # 需要 reinitialize
            if out.get('needy_rectify_info') is not None:
                rectify_info = out['needy_rectify_info'][-1]
                reinit_frame = rectify_info[0] + 1
                if reinit_frame >= total_frames:
                    # print(f"[Error] Reinit frame {reinit_frame} out of range. Breaking.")
                    break

                # print(f"[ReTrack] Reinitializing at frame {reinit_frame}")
                out_o = tracker.reinitialize(
                    reinit_frame,
                    self.safe_to_list(rectify_info[1]),
                    # rectify_info[1].tolist(),
                    self._read_image(seq.frames[reinit_frame]),
                    init_info,
                    seq.name
                )
                if out_o is None:
                    # print("[Warning] reinitialize returned None.")
                    _store_outputs(frame_num, {'target_bbox': [0, 0, 0, 0], 'time': time.time() - start_time})
                    frame_num += 1
                    out_o = {}
                prev_output = OrderedDict(out_o)
                _store_outputs(reinit_frame, out_o, {'time': time.time() - start_time})
                frame_num = reinit_frame
                continue

            prev_output = OrderedDict(out_o)
            _store_outputs(frame_num, out_o, {'time': time.time() - start_time})
            frame_num += 1

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



