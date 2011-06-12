import vidfeat
import numpy as np
import impoint


class SURF(object):

    def __init__(self, match_thresh=10):
        self._match_thresh = match_thresh
        self.scores = []

    def get_scores(self):
        return np.array(self.scores)

    def __call__(self, video):
        prev_points = None
        surf = impoint.SURF()
        for frame_num, frame_time, frame in vidfeat.convert_video(video,
                                                                  ('frameiter',
                                                                   [('opencv', 'bgr', 8)])):
            points = surf(frame)
            if prev_points is not None:
                score = len(surf.match(prev_points, points))
                self.scores.append(score)
            iskeyframe = prev_points is not None and score < self._match_thresh
            prev_points = points
            yield (frame_num, frame_time, frame), iskeyframe
