import keyframe
import cv2
import os
import viderator


def main(path, output_dir):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    kf = keyframe.Meta([keyframe.SURF(min_diff=10), keyframe.Histogram(min_diff=10)], min_diff=2)
    prev_frame = None
    for (frame_num, frame_time, frame), iskeyframe in kf(viderator.frame_iter(path)):
        if prev_frame is not None:
            if iskeyframe:
                cv2.imwrite(output_dir + '/%.8d-.jpg' % (frame_num - 1), prev_frame)
                cv2.imwrite(output_dir + '/%.8d+.jpg' % frame_num, frame)
        prev_frame = frame

    import matplotlib
    matplotlib.use('agg')
    import pylab
    pylab.figure(1)
    pylab.clf()
    pylab.plot(kf.get_scores())
    pylab.savefig('matches_fig.png')


def _parser():
    import argparse
    parser = argparse.ArgumentParser('Output pairs bordering on keyframes')
    parser.add_argument('input_video', help='Input video path')
    parser.add_argument('output_dir', help='Output dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parser()
    main(args.input_video, args.output_dir)
