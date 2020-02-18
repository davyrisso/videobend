
import collections

import numpy
import cv2

# Configuration object for Farneback Optical Flow algorithm. See:
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

# Default values for Optical Flow algorithm.
OPTFLOW_DEFAULT_CONFIG_PYR_SCALE = .5
OPTFLOW_DEFAULT_CONFIG_LEVELS = 3
OPTFLOW_DEFAULT_CONFIG_WINSIZE = 5
OPTFLOW_DEFAULT_CONFIG_ITERATIONS = 1
OPTFLOW_DEFAULT_CONFIG_POLY_N = 5
OPTFLOW_DEFAULT_CONFIG_POLY_SIGMA = 1.1
OPTFLOW_DEFAULT_CONFIG_FLAGS = cv2.OPTFLOW_FARNEBACK_GAUSSIAN

# Default config for Optical Flow algorithm.
OPTFLOW_DEFAULT_CONFIG = OpticalFlowConfig(
    pyr_scale=OPTFLOW_DEFAULT_CONFIG_PYR_SCALE,
    levels=OPTFLOW_DEFAULT_CONFIG_LEVELS,
    winsize=OPTFLOW_DEFAULT_CONFIG_WINSIZE,
    iterations=OPTFLOW_DEFAULT_CONFIG_ITERATIONS,
    poly_n=OPTFLOW_DEFAULT_CONFIG_POLY_N,
    poly_sigma=OPTFLOW_DEFAULT_CONFIG_POLY_SIGMA,
    flags=OPTFLOW_DEFAULT_CONFIG_FLAGS)


class OpticalFlow():
    """A class that holds a calculated dense optical flow.

    Provides convenience methods such as GetRemapVectors.
    """

    def __init__(self, flow, dtype=numpy.float32):
        """Constructor.

        Args:
            flow: A numpy array of flow vectors as calculated with the Farneback
                algorithm.
            dtype: The numpy dtype for the array. Default: numpy.float32.
        """
        self.__dtype = dtype
        self.__flow = flow

    @staticmethod
    def GetRemapVectors(flow, threshold_x=0.0, threshold_y=0.0,
                        multiplier_x=1.0, multiplier_y=1.0,
                        dtype=numpy.float32):
        """Calculates remap vectors from a calculated dense optical flow.

        These vectors are used to remap (transform) a frame with the cv2.remap()
        function. We can't just use the vectors from the flow as remap() needs
        the coordinates from where a point comes from in order to transform it.

        Args:
            flow: A numpy array of flow vectors as calculated with the Farneback
                algorithm.
            threshold_x: float. A scalar that sets the minimal value in pixels
                for a flow vector to be taken into account along the x axis.
                This is used to ignore small movements and help keep static
                areas sharp when applying a mosh effect. Default: 0.0.
            threshold_y: float. A scalar that sets the minimal value in pixels
                for a flow vector to be taken into account along the y axis.
                This is used to ignore small movements and help keep static
                areas sharp when applying a mosh effect. Default: 0.0.
            multiplier_x: float. A scalar to multiply the flow vectors by on
                the x axis.
            multiplier_y: float. A scalar to multiply the flow vectors by on
                the y axis.
            dtype: The numpy dtype of the resulting vectors. Default: float32.

        Returns:
            A tuple of (vectors_x, vectors_y) of remap vectors ready to be used
            as map in cv2.remap().
        """
        # Creates an array of pixel coordinates ([[0,0], [0,1]...], ...).
        coords_y, coords_x, = numpy.indices(
            (flow.shape[0], flow.shape[1]), dtype=dtype)

        # Calculates remap vectors:
        # We apply the threshold and multipliers to the flow values and add
        # the inverse of the resulting value to the coordinates created above.
        # This converts the flow values in remap vectors.
        remap_vectors_x = numpy.add(
            coords_x,
            - (flow[..., 0] * (abs(flow[..., 0]) >= threshold_x)) * multiplier_x)
        remap_vectors_y = numpy.add(
            coords_y,
            -(flow[..., 1] * (abs(flow[..., 1]) >= threshold_y)) * multiplier_y)

        return (remap_vectors_x, remap_vectors_y)

    @classmethod
    def Like(cls, optical_flow):
        """Creates an empty flow (0 vectors) of the same shape and dtype
        as the passed OpticalFlow object.

        Args:
            optical_flow: An OpticalFlow object.

        Returns:
            A new OpticalFlow object of the same shape and dtype as the passed
            object.
        """
        return cls(numpy.zeros_like(optical_flow.flow),
                   dtype=optical_flow.dtype)

    @classmethod
    def Add(cls, dtype=numpy.float32, *flows):
        """Adds two OpticalFlow objects (i.e adds their flows).

        Args:
            flow_1: An OpticalFlow object.
            flow_2: An OpticalFlow object.

        Returns:
            A new OpticalFlow object which flow is the sum of the passed flows.
        """
        return cls(
            numpy.sum([flow.flow for flow in flows], axis=0), dtype=dtype)

    @property
    def flow(self):
        return self.__flow

    @property
    def dtype(self):
        return self.__dtype


class OpticalFlowGenerator():
    """A class that generates OpticalFlows based on an iterable of frames."""

    def __init__(self, frames, config=OPTFLOW_DEFAULT_CONFIG):
        """Constructor.

        Args:
            frames: An iterable of utils.Frame. The frames for which to
                calculate the optical flow.
        """
        self.__frames = frames
        self.__config = config

    @staticmethod
    def GetFlow(frames, config=OPTFLOW_DEFAULT_CONFIG):
        """Returns a generator of OpticalFlow objects for the passed frames.

        Args:
            frames: An iterable of utils.Frame
            config: An OpticalFlowConfig namedtuple. Configuration for the
                Farneback algorithm.

        Yields:
            OpticalFlow objects for each frame in the passed iterable.
        """
        previous_frame_pixels_grayscale = None

        for frame in frames:
            # Converts the frame to grayscale, as this is what the Farneback
            # alrogithm uses.
            # TODO(davyrisso): Play on contrast of base frame.
            # E.g: frame_pixels_grayscale = frame_pixels_grayscale * 5.0.
            frame_pixels_grayscale = cv2.cvtColor(
                frame.pixels, cv2.COLOR_BGR2GRAY)
            if previous_frame_pixels_grayscale is None:
                previous_frame_pixels_grayscale = frame_pixels_grayscale

            # Calculates dense optical flow with the Farneback alrgorithm
            # between the previous and current frames.
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
