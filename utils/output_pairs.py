import keyframe
import pyffmpeg
import cv
import sys
import impoint
import matplotlib
matplotlib.use('agg')
import pylab


def main(path):
    if 0:
        kf = keyframe.SURF()
        stream = pyffmpeg.VideoStream()
        stream.open(path)
        prev_frame = None
        for (frame_num, frame_time, frame), iskeyframe in kf(stream):
            if prev_frame:
                if iskeyframe:
                    cv.SaveImage('pairs/%.8d-.jpg' % (frame_num - 1), prev_frame)
                    cv.SaveImage('pairs/%.8d+.jpg' % frame_num, frame)
            prev_frame = frame

        pylab.figure(1)
        pylab.clf()
        pylab.plot(-kf.get_scores())
        pylab.savefig('matches_fig.png')

    kf = keyframe.Histogram()
    stream = pyffmpeg.VideoStream()
    stream.open(path)
    prev_frame = None
    for (frame_num, frame_time, frame), iskeyframe in kf(stream):
        if prev_frame:
            if iskeyframe:
                cv.SaveImage('pairs/%.8d-.jpg' % (frame_num - 1), prev_frame)
                cv.SaveImage('pairs/%.8d+.jpg' % frame_num, frame)
        prev_frame = frame

    pylab.figure(1)
    pylab.clf()
    pylab.plot(kf.get_scores())
    pylab.savefig('scores_fig.png')


if __name__ == '__main__':
    path = '/mnt/nfsdrives/data/trecvid/TRECVID/events/E001/HVC006045-compressed.avi'
    if len(sys.argv) == 2:
        path = sys.argv[1]
    main(path)
