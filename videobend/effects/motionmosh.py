import sys

import numpy
import cv2

from videobend.utils import video

# TODO(davyrisso): Expose parameters (Optical Flow, Frame opacity, etc.)


def GenerateFrames(
        input_video,
        start_frame, end_frame,
        effect_start_frame, effect_end_frame,
        interactive=False):
    """Applies the effect and returns a generator over the generated frames."""

    # Creates the initial previous frame as the first affected frame.
    # Note: We convert to grayscale as it is what optical flow uses.
    previous_frame_grayscale = cv2.cvtColor(
        input_video.ReadFrameAt(effect_start_frame), cv2.COLOR_BGR2GRAY)

    # Places the first frame of the video in an image buffer.
    image_buffer = input_video.ReadFrameAt(effect_start_frame)

    for frame, position in input_video.ReadFrames(
            start_frame=start_frame, end_frame=end_frame,
            interactive=interactive):

        # Returns unaffected frames before effect_start_frame.
        if position[0] < effect_start_frame:
            yield frame, position
            continue

        # Returns unaffected frames after effect_end_frame.
        if position[0] > effect_end_frame:
            yield frame, position
            continue

        # Converts current frame to grayscale for optical flow.
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Performs optical flow between previous and current frame.
        # Note: the numeric parameters will impact the look of the effect.
        optical_flow = cv2.calcOpticalFlowFarneback(
            prev=previous_frame_grayscale,
            next=frame_grayscale,
            flow=None,
            pyr_scale=.9,
            levels=3,
            winsize=2,
            iterations=5,
            poly_n=2,
            poly_sigma=1.0,
            flags=0)

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
        motion_x = numpy.add(coords_x, -optical_flow[..., 1])
        motion_y = numpy.add(coords_y, -optical_flow[..., 0])

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
            borderMode=cv2.BORDER_DEFAULT)

        # We add the pixels of the current frame to the image buffer with a low
        # opacity, so as to preserve some of the details of the frame.
        # If we skip this step (or set the beta parameter to 0), then only the
        # pixels of the first frame will be present throughout the video.
        cv2.addWeighted(
            src1=image_buffer,
            src2=frame,
            dst=image_buffer,
            alpha=.95,
            beta=.05,
            gamma=0)

        yield image_buffer, position

        previous_frame_grayscale = frame_grayscale


def main(input_video_path, output_video_path, start_frame=0, end_frame=None,
         effect_start_frame=0, effect_end_frame=None, preview=False):
    input_video = video.VideoReader(file_path=input_video_path)
    output_video = video.VideoWriter.FromReader(
        input_video, file_path=output_video_path)

    if end_frame is None:
        end_frame = input_video.frame_count

    if effect_end_frame is None:
        effect_end_frame = input_video.frame_count

    effect_start_frame = max(effect_start_frame, start_frame)
    effect_end_frame = min(effect_end_frame, end_frame)

    print('\n'.join(['Starting motionmosh effect with parameters:',
                     ' - input: %s' % input_video_path,
                     ' - output: %s' % output_video_path,
                     ' - video:',
                     '   - start frame: %d' % start_frame,
                     '   - end frame: %d' % end_frame,
                     ' - effect:',
                     '   - start frame: %d' % effect_start_frame,
                     '   - effect end frame: %d' % effect_end_frame,
                     '\n']))

    print('\n'.join(['Input video info:',
                     ' - size: %dx%d' % input_video.frame_size,
                     ' - fps: %d' % input_video.fps,
                     ' - frame count: %d' % input_video.frame_count,
                     ' - codec code: %s' % input_video.fourcc,
                     '\n']))

    print('Generating frames...')

    # Gets a generator of transformed frames.
    generated_frames = GenerateFrames(
        input_video,
        start_frame=start_frame,
        end_frame=end_frame,
        effect_start_frame=effect_start_frame,
        effect_end_frame=effect_end_frame,
        interactive=preview)

    for frame, position in output_video.WriteFrames(generated_frames):
        progress = float(position[0]) / float(end_frame)
        sys.stdout.write(
            'Progress: %d/%d (%.2f)' %
            (position[0], end_frame, progress * 100) + '%\r')
        sys.stdout.flush()

        if preview:
            cv2.imshow('Preview', frame)

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
        args.output_video_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        effect_start_frame=args.effect_start_frame,
        effect_end_frame=args.effect_end_frame,
        preview=bool(args.preview))
