import keyframe
import pyffmpeg
import cv
import sys


def main(path):
    kf = keyframe.Histogram()
    stream = pyffmpeg.VideoStream()
    stream.open(path)
    prev_frame = None
    for (frame_num, frame_time, frame), iskeyframe in kf(stream):
        if iskeyframe and prev_frame:
            cv.SaveImage('%.8d.jpg' % (frame_num - 1), prev_frame)
            cv.SaveImage('%.8d.jpg' % frame_num, frame)
        prev_frame = frame

if __name__ == '__main__':
    path = '/home/tp/data/videos/HVC006045-compressed.avi'
    if len(sys.argv) == 2:
        path = sys.argv[1]
    main(path)
