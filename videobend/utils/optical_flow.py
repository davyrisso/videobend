
import collections

import numpy
import cv2

# https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
OpticalFlowConfig = collections.namedtuple(
    'OpticalFlowConfig', [
        'pyr_scale',
        'levels',
        'winsize',
        'iterations',
        'poly_n',
        'poly_sigma',
        'flags'])

OPTFLOW_DEFAULT_CONFIG_PYR_SCALE = .5
OPTFLOW_DEFAULT_CONFIG_LEVELS = 3
OPTFLOW_DEFAULT_CONFIG_WINSIZE = 5
OPTFLOW_DEFAULT_CONFIG_ITERATIONS = 1
OPTFLOW_DEFAULT_CONFIG_POLY_N = 5
OPTFLOW_DEFAULT_CONFIG_POLY_SIGMA = 1.1
OPTFLOW_DEFAULT_CONFIG_FLAGS = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

OPTFLOW_DEFAULT_CONFIG = OpticalFlowConfig(
    pyr_scale=OPTFLOW_DEFAULT_CONFIG_PYR_SCALE,
    levels=OPTFLOW_DEFAULT_CONFIG_LEVELS,
    winsize=OPTFLOW_DEFAULT_CONFIG_WINSIZE,
    iterations=OPTFLOW_DEFAULT_CONFIG_ITERATIONS,
    poly_n=OPTFLOW_DEFAULT_CONFIG_POLY_N,
    poly_sigma=OPTFLOW_DEFAULT_CONFIG_POLY_SIGMA,
    flags=OPTFLOW_DEFAULT_CONFIG_FLAGS)


class OpticalFlow():

    def __init__(self, flow, dtype=numpy.float32):
        self.__dtype = dtype
        self.__flow = flow
        self.__motion_vectors_x, self.__motion_vectors_y = (
            self.GetMotionVectors(flow))

    @staticmethod
    def GetMotionVectors(flow, dtype=numpy.float32):
        coords_y, coords_x, = numpy.indices(
            (flow.shape[0], flow.shape[1]), dtype=dtype)

        motion_vectors_x = numpy.add(coords_x, -flow[..., 0])
        motion_vectors_y = numpy.add(coords_y, -flow[..., 1])

        return (motion_vectors_x, motion_vectors_y)

    @classmethod
    def Like(cls, flow):
        return cls(numpy.zeros_like(flow.flow), dtype=flow.dtype)

    @classmethod
    def Add(cls, flow_1, flow_2, dtype=numpy.float32):
        return cls(numpy.add(flow_1.flow, flow_2.flow), dtype=dtype)

    @property
    def flow(self):
        return self.__flow

    @property
    def dtype(self):
        return self.__dtype

    @property
    def motion_vectors_x(self):
        return self.__motion_vectors_x

    @property
    def motion_vectors_y(self):
        return self.__motion_vectors_y

    @property
    def motion_vectors(self):
        return numpy.stack(
            (self.motion_vectors_x, self.motion_vectors_y), axis=1)


class OpticalFlowGenerator():

    def __init__(self, frames, config=OPTFLOW_DEFAULT_CONFIG):
        self.__frames = frames
        self.__config = config

    @staticmethod
    def GetFlow(frames, config=OPTFLOW_DEFAULT_CONFIG):
        previous_frame_pixels_grayscale = None

        for frame in frames:
            # TODO(davyrisso): Play on contrast of base frame.
            # E.g: frame_pixels_grayscale = frame_pixels_grayscale * 5.0.
            frame_pixels_grayscale = cv2.cvtColor(
                frame.pixels, cv2.COLOR_BGR2GRAY)
            if previous_frame_pixels_grayscale is None:
                previous_frame_pixels_grayscale = frame_pixels_grayscale

            optical_flow = cv2.calcOpticalFlowFarneback(
                prev=previous_frame_pixels_grayscale,
                next=frame_pixels_grayscale,
                flow=None,
                pyr_scale=config.pyr_scale,
                levels=config.levels,
                winsize=config.winsize,
                iterations=config.iterations,
                poly_n=config.poly_n,
                poly_sigma=config.poly_sigma,
                flags=config.flags)

            yield frame, OpticalFlow(optical_flow)

            previous_frame_pixels_grayscale = frame_pixels_grayscale

    def GenerateFlow(self):
        return self.GetFlow(self.__frames, self.__config)
