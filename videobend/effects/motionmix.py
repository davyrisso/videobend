import sys

import numpy
import cv2

from videobend.utils import video

# TODO(davyrisso): Expose parameters (Optical Flow, Frame opacity, etc.)


def GenerateFrames(input_video, motion_video, interactive=False):
    """Applies the effect and returns a generator over the generated frames."""

    # Creates the initial previous frame as the first affected frame.
    # Note: We convert to grayscale as it is what optical flow uses.
    previous_motion_frame_grayscale = cv2.cvtColor(
        motion_video.ReadFrameAt(0), cv2.COLOR_BGR2GRAY)
    motion_video_frames = motion_video.ReadFrames(
        start_frame=1)

    previous_frame_grayscale = cv2.cvtColor(
        input_video.ReadFrameAt(0), cv2.COLOR_BGR2GRAY)

    # Places the first frame of the video in an image buffer.
    image_buffer = input_video.ReadFrameAt(0)

    sum_motion_flows = numpy.zeros(
        (input_video.frame_height, input_video.frame_width, 2),
        dtype=numpy.float32)

    sum_frames_flows = numpy.zeros(
        (input_video.frame_height, input_video.frame_width, 2),
        dtype=numpy.float32)

    for frame, position in input_video.ReadFrames(interactive=interactive):
        motion_frame, motion_position = motion_video_frames.next()

        motion_frame_grayscale = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Performs optical flow between previous and current frame.
        # Note: the numeric parameters will impact the look of the effect.
        motion_optical_flow = cv2.calcOpticalFlowFarneback(
            prev=previous_motion_frame_grayscale,
            next=motion_frame_grayscale,
            flow=None,
            pyr_scale=.25,
            levels=5,
            winsize=15,
            iterations=20,
            poly_n=5,
            poly_sigma=1.1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        sum_motion_flows = numpy.add(motion_optical_flow, sum_motion_flows)

        optical_flow = cv2.calcOpticalFlowFarneback(
            prev=previous_frame_grayscale,
            next=frame_grayscale,
            flow=None,
            pyr_scale=.25,
            levels=5,
            winsize=15,
            iterations=20,
            poly_n=5,
            poly_sigma=1.1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        sum_frames_flows = numpy.add(optical_flow, sum_frames_flows)

        input_video_weight = 1.0
        motion_video_weight = 1.0

        total_flow = numpy.add(
            optical_flow * input_video_weight,
            motion_optical_flow * motion_video_weight)

        # Creates a matrix of pixel coordinates of the same size as the image,
        # this is used to convert the values of the optical flow to motion
        # vectors.
        coords_y, coords_x, = numpy.indices(
            (input_video.frame_height, input_video.frame_width),
            dtype=numpy.float32)

        # Creates motion vectors x & y values by adding the flow vectors to the
        # pixel coordinates above.
        # Note: We add the inverse of the optical flow because optical flow
        # values indicate where the motion vectors come from and we are looking
        # for the inverse, i.e where they are pointing to.
        motion_x = numpy.add(coords_x, -total_flow[..., 0])
        motion_y = numpy.add(coords_y, -total_flow[..., 1])

        sum_motion_x = numpy.add(coords_x, -sum_motion_flows[..., 0])
        sum_motion_y = numpy.add(coords_y, -sum_motion_flows[..., 1])

        # Applies the motion vectors to the image buffer that contains the
        # previous calculated image (or first frame for first iteration).
        # Note: The interpolation and borderMode values will impact the look of
        # the effect.
        cv2.remap(
            src=image_buffer,
            dst=image_buffer,
            map1=motion_x,
            map2=motion_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        # TODO(davyrisso): Add sum of frame optical flows to motion (so the
        # movement of the frame also impacts the motion of the motion video).
        # E.g: drop on face should follow the face too.

        # Calculates the transformed image for the current frame.
        frame_copy = frame.copy()
        cv2.remap(
            src=frame_copy,
            dst=frame_copy,
            map1=sum_motion_x,
            map2=sum_motion_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)

        # Blends in pixels of the transformed current frame.
        frame_weight = 0.25
        cv2.addWeighted(
            src1=image_buffer,
            src2=frame_copy,
            dst=image_buffer,
            alpha=1 - frame_weight,
            beta=frame_weight,
            gamma=0)

        # Blends in pixels of the motion video.
        motion_weight = 0.05
        cv2.addWeighted(
            src1=image_buffer,
            src2=motion_frame,
            dst=image_buffer,
            alpha=1 - motion_weight,
            beta=motion_weight,
            gamma=0)

        # Calculates edges with Canny method
        motion_edges_weight = .0
        motion_edges = cv2.Canny(motion_frame, 10, 50)
        motion_edges_rgb = cv2.cvtColor(motion_edges, cv2.COLOR_GRAY2BGR)
        # Blends in pixels of the edges.
        cv2.addWeighted(
            src1=image_buffer,
            src2=motion_edges_rgb,
            dst=image_buffer,
            alpha=1 - motion_edges_weight,
            beta=motion_edges_weight,
            gamma=0)

        yield image_buffer, position

        previous_motion_frame_grayscale = motion_frame_grayscale
        previous_frame_grayscale = frame_grayscale


def main(input_video_path, motion_video_path, output_video_path,
         preview=False):
    input_video = video.VideoReader(file_path=input_video_path)
    motion_video = video.VideoReader(file_path=motion_video_path)
    output_video = video.VideoWriter.FromReader(
        input_video, file_path=output_video_path)

    start_frame = 0
    end_frame = input_video.frame_count

    print('\n'.join(['Input video info:',
                     ' - size: %dx%d' % input_video.frame_size,
                     ' - fps: %d' % input_video.fps,
                     ' - frame count: %d' % input_video.frame_count,
                     ' - codec code: %s' % input_video.fourcc,
                     '\n']))

    print('\n'.join(['Motion video info:',
                     ' - size: %dx%d' % motion_video.frame_size,
                     ' - fps: %d' % motion_video.fps,
                     ' - frame count: %d' % motion_video.frame_count,
                     ' - codec code: %s' % motion_video.fourcc,
                     '\n']))

    print('Generating frames...')

    # Gets a generator of transformed frames.
    generated_frames = GenerateFrames(
        input_video, motion_video, interactive=preview)

    for frame, position in output_video.WriteFrames(generated_frames):
        cv2.imshow('Preview', frame)
        print(position)
        # sys.stdout.write(
        #     'Progress: %d/%d (%.2f)% \r' %
        #     (position[0], end_frame,
        #       float(position[0]) / float(end_frame) * 100))
        # sys.stdout.flush()

        # if preview:
        #     cv2.imshow('Preview', frame)

    print('\nDone.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=('Uses Optical Flow to reproduce a Datamosh effect.'))

    parser.add_argument(
        'input_video_path',
        type=str,
        help='Input video path')

    parser.add_argument(
        'motion_video_path',
        type=str,
        help='Motion video path'
    )

    parser.add_argument(
        'output_video_path',
        type=str,
        help='Output video file path')

    parser.add_argument(
        '--start_frame', '-s',
        metavar='<frame>',
        type=int,
        default=0,
        help='Video start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '--end_frame', '-e',
        metavar='<frame>',
        type=int,
        default=None,
        help='Video end frame number. Default: last frame.')

    parser.add_argument(
        '--effect_start_frame', '-fx_s',
        metavar='<frame>',
        type=int,
        default=0,
        help='Effect start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '--effect_end_frame', '-fx_e',
        metavar='<frame>',
        type=int,
        default=None,
        help='Effect end frame number. Default: last frame.')

    parser.add_argument(
        '--preview',
        action='count',
        default=0,
        help='If set, shows a preview of the generated frames')

    args = parser.parse_args()

    main(
        args.input_video_path,
        args.motion_video_path,
        args.output_video_path,
        preview=bool(args.preview))
