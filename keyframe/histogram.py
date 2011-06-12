import imfeat
import vidfeat
import numpy as np


class Histogram(object):

    def __init__(self, diff_thresh=10, skip_mod=1, min_interval=3.0):
        self._diff_thresh = diff_thresh
        self._skip_mod = skip_mod
        self._min_interval = min_interval
        self.scores = []

    def get_scores(self):
        return np.array(self.scores)

    def __call__(self, video):
        prev_vec = None
        prev_time = None
        for frame_num, frame_time, frame in vidfeat.convert_video(video, ('frameiterskip', [('opencv', 'bgr', 8)], self._skip_mod)):

            width, height = frame.width, frame.height
            cgr = imfeat.CoordGeneratorRect
            out = []
            # TODO: This uses LAB to convert to Gray, fix that in imfeat
            for block, trans in imfeat.BlockGenerator(frame, cgr, output_size=(50, 50), step_delta=(50, 50)):
                feat = imfeat.Histogram('lab', num_bins=(8, 8, 8), style='planar')
                out.append(imfeat.compute(feat, block)[:8])
            cur_vec = np.hstack(out)
            if prev_vec is not None:
                score = np.sum(np.abs(prev_vec - cur_vec))
                self.scores.append(score)
                if score > self._diff_thresh and \
                       frame_time - prev_time > self._min_interval:
                    iskeyframe = True
                    prev_time = frame_time
                else:
                    iskeyframe = False
            else:
                prev_time = frame_time
                iskeyframe = False
            prev_vec = cur_vec
            yield (frame_num, frame_time, frame), iskeyframe
