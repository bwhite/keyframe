import imfeat
import vidfeat
import numpy as np


class Histogram(object):

    def __init__(self, diff_thresh=10, skip_mod=1):
        self._diff_thresh = diff_thresh
        self._skip_mod = skip_mod

    def __call__(self, video):
        prev_vec = None
        for frame_num, frame_time, frame in vidfeat.convert_video(video, ('frameiterskip', [('opencv', 'bgr', 8)], self._skip_mod)):
            width, height = frame.width, frame.height
            cgr = imfeat.CoordGeneratorRect
            out = []
            # TODO: This uses LAB to convert to Gray, fix that in imfeat
            for block, trans in imfeat.BlockGenerator(frame, cgr, output_size=(50, 50), step_delta=(50, 50)):
                out.append(imfeat.Histogram('lab', num_bins=(8, 8, 8), style='planar')(block)[:8])
            cur_vec = np.hstack(out)
            iskeyframe = prev_vec is not None and np.sum(np.abs(prev_vec - cur_vec)) > self._diff_thresh
            prev_vec = cur_vec
            yield (frame_num, frame_time, frame), iskeyframe
