import imfeat
import vidfeat
import pyffmpeg
import numpy as np
import cv


class Histogram(object):

    def __init__(self):
        pass

    def compute(self, video):
        prev_vec = None
        prev_frame = None
        for frame_num, frame_time, frame in vidfeat.convert_video(video, ('frameiter', [('opencv', 'bgr', 8)])):
            width, height = frame.width, frame.height
            cgr = imfeat.CoordGeneratorRect
            out = []
            for block, trans in imfeat.BlockGenerator(frame, cgr, output_size=(50, 50), step_delta=(50, 50)):
                out.append(imfeat.Histogram('lab', num_bins=(8, 8, 8), style='planar')(block)[:8])
            cur_vec = np.hstack(out)
            if prev_vec is not None:
                diff = np.sum(np.abs(prev_vec - cur_vec))
                print(diff)
                if diff > 10:
                    cv.SaveImage('%.8d.jpg' % (frame_num - 1), prev_frame)
                    cv.SaveImage('%.8d.jpg' % frame_num, frame)
            prev_vec = cur_vec
            prev_frame = frame

stream = pyffmpeg.VideoStream()
stream.open('/home/tp/data/videos/HVC006045-compressed.avi')
Histogram().compute(stream)
