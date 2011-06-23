import vidfeat
import numpy as np
import impoint


class SURF(object):

    def __init__(self, match_thresh=10, skip_mod=1, min_interval=3.0):
        self._match_thresh = match_thresh
        self._skip_mod = skip_mod
        self._min_interval = min_interval
        self.scores = []

    def get_scores(self):
        return np.array(self.scores)

    def __call__(self, video):
        prev_points = None
        prev_time = None        
        surf = impoint.SURF()
        for frame_num, frame_time, frame in vidfeat.convert_video(video, ('frameiterskip', [('opencv', 'bgr', 8)], self._skip_mod)):
            points = surf(frame)
            if prev_points is not None:
                score = len(surf.match(prev_points, points))
                self.scores.append(score)
                if score < self._match_thresh and frame_time - prev_time > self._min_interval:
                    iskeyframe = True
                    prev_time = frame_time
                else:
                    iskeyframe = False
            else:
                prev_time = frame_time
                iskeyframe = False
            prev_points = points
            yield (frame_num, frame_time, frame), iskeyframe
