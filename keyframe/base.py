import numpy as np


class BaseKeyframer(object):

    def __init__(self, min_diff=float('inf'), max_diff=float('-inf'), min_interval=.5):
        self._min_diff = min_diff
        self._max_diff = max_diff
        self._min_interval = min_interval
        self.scores = []

    def get_scores(self):
        return np.array(self.scores)

    def __call__(self, frame_iter):
        prev_vec = None
        prev_time = None
        for frame_num, frame_time, frame in frame_iter:
            height, width = frame.shape[:2]
            cur_vec = self.feat_func(frame)
            if prev_vec is not None:
                score = self.diff_func(prev_vec, cur_vec)
                print('%d: Score[%f]' % (frame_num, score))
                self.scores.append(score)
                if (self._min_diff <= score or score <= self._max_diff) and self._min_interval < frame_time - prev_time:
                    iskeyframe = True
                    prev_time = frame_time
                else:
                    iskeyframe = False
            else:
                prev_time = frame_time
                iskeyframe = False
            prev_vec = cur_vec
            yield (frame_num, frame_time, frame), iskeyframe
