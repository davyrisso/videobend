"""Mosh effect. Uses OpticalFlow to reproduce a datamoshing effect.

We first calculate the dense optical flow between each frame of the input video
and its preceding frame. We then convert the obtained flow in remap vectors
and apply these vectors to an image buffer so as to apply to it the motion
approximated by the dense optical flow.

This file can be used as a library and a script.
"""
import itertools
import sys

import numpy
import cv2

from ..utils.constants import BORDER_MODES, DEFAULT_BORDER_VALUE, INTERPOLATION_METHODS
from ..utils.frame import Frame
from ..utils.optical_flow import OpticalFlow, OpticalFlowGenerator
from ..utils.video import VideoReader, VideoWriter


def GenerateFrames(
        input_video,
        start_frame, end_frame,
        effect_start_frame, effect_end_frame,
        motion_multiplier_x, motion_multiplier_y,
        motion_threshold_x, motion_threshold_y,
        frame_blend_weight, interpolation_method, border_mode):
    """Applies the effect and returns a generator over the resulting frames.

    Args:
        input_video: string. The path to the input video file.
        start_frame: int. The frame number of the first frame to include in the
            output video.
        end_frame: int. The frame number of the last frame to include in the
            output video.
        effect_start_frame: int. The frame number of the first frame to apply
            the effect to.
        effect_end_frame: int. The frame number of the last frame to apply
            the effect to.
        motion_multiplier_x: float. A scalar to multiply the estimated motion
            by on the x axis.
        motion_multiplier_y: float. A scalar to multiply the estimated motion
            by on the y axis.
        motion_threshold_x: float. Minimum value for an estimated motion vector
            to be taken into account on the x axis.
        motion_threshold_y: float. Minimum value for an estimated motion vector
            to be taken into account on the y axis.
        frame_blend_weight: float. How much the current transformed frame should
            be blended with the result frame.
        interpolation_method: int. Interpolation method (value from 0 to 4). See
            utils.constants.
        border_mode: int. Border mode (value from 0 to 16). See utils.constants.

    Returns:
        A generator over the resulting frames.
    """

    # Unaffected frames before effect.
    frames_before = input_video.GetStream().ReadFrames(
        start_frame=start_frame, end_frame=effect_start_frame)

    # Unaffected frames after effect.
    frames_after = input_video.GetStream().ReadFrames(
        start_frame=effect_end_frame, end_frame=end_frame)

    # Affected frames.
    frames = input_video.GetStream().ReadFrames(
        start_frame=effect_start_frame, end_frame=effect_end_frame)

    def GeneratedFrames():
        """A generator over the resulting frames."""
        # The image buffer will be initialized as the first affected frame.
        # It is the pixels of this buffer that will be transformed by applying
        # the calculated remap vectors.
        image_buffer = None

        for frame, optical_flow in OpticalFlowGenerator(frames).GenerateFlow():
            # Stores the pixels of the first frame in the image buffer.
            if image_buffer is None:
                image_buffer = frame.pixels.copy()

            # Retrieves the remap vectors and applies thresholds and
            # multipliers.
            remap_vectors = optical_flow.GetRemapVectors(
                optical_flow.flow,
                multiplier_x=motion_multiplier_x,
                multiplier_y=motion_multiplier_y,
                threshold_x=motion_threshold_x,
                threshold_y=motion_threshold_y)

            # Applies the estimated movement of the current frame to the
            # image buffer.
            cv2.remap(
                src=image_buffer,
                dst=image_buffer,
                map1=remap_vectors[0],
                map2=remap_vectors[1],
                interpolation=interpolation_method,
                borderMode=border_mode,
                borderValue=DEFAULT_BORDER_VALUE)

            # Blends in the current transformed frame if blend weight > 0.
            if frame_blend_weight > 0:
                # We first transform the frame by applying the remap vectors.
                cv2.remap(
                    src=frame.pixels,
                    dst=frame.pixels,
                    map1=remap_vectors[0],
                    map2=remap_vectors[1],
                    interpolation=interpolation_method,
                    borderMode=border_mode,
                    borderValue=DEFAULT_BORDER_VALUE)
                # We then blend in the resulting frame with the image buffer.
                cv2.addWeighted(
                    src1=image_buffer,
                    src2=frame.pixels,
                    dst=image_buffer,
                    alpha=1 - frame_blend_weight,
                    beta=frame_blend_weight,
                    gamma=0)

            yield Frame(image_buffer, frame.position)

    # We return a generator over of all unaffected and affected frames.
    return itertools.chain(frames_before, GeneratedFrames(), frames_after)


def main(input_video_path, output_video_path, start_frame=0, end_frame=None,
         effect_start_frame=0, effect_end_frame=None,
         motion_multiplier_x=1.0, motion_multiplier_y=1.0,
         motion_threshold=0.0, frame_blend_weight=0,
         interpolation_method=0, border_mode=1, preview=False):
    input_video = VideoReader(file_path=input_video_path)
    output_video = VideoWriter.FromReader(
        input_video, file_path=output_video_path)

    if end_frame is None:
        end_frame = input_video.frame_count

    if effect_end_frame is None:
        effect_end_frame = input_video.frame_count

    effect_start_frame = max(effect_start_frame, start_frame)
    effect_end_frame = min(effect_end_frame, end_frame)

    print('\n'.join(['Starting mosh effect with parameters:',
                     ' - input: %s' % input_video_path,
                     ' - output: %s' % output_video_path,
                     ' - video:',
                     '   - start frame: %d' % start_frame,
                     '   - end frame: %d' % end_frame,
                     ' - effect:',
                     '   - start frame: %d' % effect_start_frame,
                     '   - effect end frame: %d' % effect_end_frame,
                     '   - motion_multiplier: (%f, %f)' % (
                         motion_multiplier_x, motion_multiplier_y),
                     '   - motion threshold: %f' % motion_threshold,
                     '   - frame blend weight: %f' % frame_blend_weight,
                     '   - interpolation method: %s' % (
                         INTERPOLATION_METHODS[interpolation_method]),
                     '   - border mode: %s' % BORDER_MODES[border_mode],
                     '\n']))

    print('\n'.join(['Input video info:',
                     ' - size: %dx%d' % input_video.frame_size,
                     ' - fps: %d' % input_video.fps,
                     ' - frame count: %d' % input_video.frame_count,
                     ' - codec code: %s' % input_video.fourcc,
                     '\n']))

    # Gets a generator of transformed frames.
    generated_frames = GenerateFrames(
        input_video,
        start_frame=start_frame,
        end_frame=end_frame,
        effect_start_frame=effect_start_frame,
        effect_end_frame=effect_end_frame,
        frame_blend_weight=frame_blend_weight,
        motion_multiplier_x=motion_multiplier_x,
        motion_multiplier_y=motion_multiplier_y,
        motion_threshold_x=motion_threshold,
        motion_threshold_y=motion_threshold,
        interpolation_method=interpolation_method,
        border_mode=border_mode)

    # Writes frames to the output video file.
    print('Writing frames to %s...' % output_video_path)
    for frame in output_video.WriteFrames(generated_frames):
        sys.stdout.write(
            'Progress: %d/%d (%.2f%%) \r' %
            (frame.position[0], end_frame,
             float(frame.position[0]) / float(end_frame) * 100))
        sys.stdout.flush()

        # Shows a preview if requested.
        if preview:
            cv2.imshow('Preview', frame.pixels)

    print('\nDone writing %s.' % output_video_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=('Uses Optical Flow to reproduce a Datamosh effect.'))

    parser.add_argument(
        '-i', '--input_video_path',
        metavar='<video file path>',
        required=True,
        type=str,
        help='Input video path')

    parser.add_argument(
        '-o', '--output_video_path',
        metavar='<path>',
        required=True,
        type=str,
        help='Output video file path')

    parser.add_argument(
        '-s', '--start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Output video start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '-e', '--end_frame',
        metavar='<frame>',
        type=int,
        default=None,
        help='Output video end frame number. Default: last frame.')

    parser.add_argument(
        '-fx_s', '--effect_start_frame',
        metavar='<frame>',
        type=int,
        default=0,
        help='Effect start frame number. Default: 0 (first frame)')

    parser.add_argument(
        '-fx_e', '--effect_end_frame',
        metavar='<frame>',
        type=int,
        default=None,
        help='Effect end frame number. Default: last frame.')

    parser.add_argument(
        '--frame_blend_weight',
        metavar='<0.0-1.0>',
        type=float,
        default=0.0,
        help=('Blend weight for the transformed frame in the final image. ' +
              'Used to preserve detail from the frames throuhougt the video. ' +
              'If > 0 the transformed frames will be blended with the result ' +
              'images, otherwise only the first frame\'s pixels will be used.'))

    parser.add_argument(
        '-m_threshold', '--motion_threshold',
        metavar='<pixel_threshold>',
        type=float,
        default=0.0,
        help=('Motion pixel threshold. Motion below this value is ignored. ' +
              'Setting a value > 0 helps to keep slightly moving areas sharp.')
    )

    parser.add_argument(
        '-m_mult_x', '--motion_multiplier_x',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Multiplier to apply to the motion vectors along the x axis. ' +
              'This value will be multiplied by the global motion multiplier.')
    )

    parser.add_argument(
        '-m_mult_y', '--motion_multiplier_y',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help=('Multiplier to apply to the motion vectors along the y axis.' +
              'This value will be multiplied by the global motion multiplier.')
    )

    parser.add_argument(
        '-m_mult', '--motion_multiplier',
        metavar='<multiplier>',
        type=float,
        default=1.0,
        help='Multiplier to apply to the motion vectors.'
    )

    parser.add_argument(
        '--interpolation_method',
        metavar='<0-4>',
        type=int,
        default=0,
        help=(
            'Interpolation method. Values: %s. Default: 0 (%s).' % (
                INTERPOLATION_METHODS, INTERPOLATION_METHODS[0])))

    parser.add_argument(
        '--border_mode',
        metavar='<0-16>',
        type=int,
        default=1,
        help=(
            'Border mode. Values: %s. Default: 1 (%s).' % (
                BORDER_MODES, BORDER_MODES[1])))

    parser.add_argument(
        '--preview',
        action='count',
        default=0,
        help='Shows a preview of the generated frames during computation.')

    args = parser.parse_args()

    main(
        args.input_video_path,
        args.output_video_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        effect_start_frame=args.effect_start_frame,
        effect_end_frame=args.effect_end_frame,
        frame_blend_weight=args.frame_blend_weight,
        motion_multiplier_x=args.motion_multiplier_x * args.motion_multiplier,
        motion_multiplier_y=args.motion_multiplier_y * args.motion_multiplier,
        motion_threshold=args.motion_threshold,
        interpolation_method=args.interpolation_method,
        border_mode=args.border_mode,
        preview=bool(args.preview))
