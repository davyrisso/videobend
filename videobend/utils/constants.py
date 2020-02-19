import numpy
import cv2

INTERPOLATION_METHODS = {
    cv2.INTER_NEAREST: 'Nearest neighbor',
    cv2.INTER_LINEAR: 'Linear',
    cv2.INTER_CUBIC: 'Cubic',
    cv2.INTER_AREA: 'Area',
    cv2.INTER_LANCZOS4: 'Lanczos',
}

BORDER_MODES = {
    cv2.BORDER_CONSTANT: 'Constant',
    cv2.BORDER_REPLICATE: 'Replicate',
    cv2.BORDER_REFLECT: 'Reflect',
    cv2.BORDER_WRAP: 'Wrap',
    cv2.BORDER_REFLECT101: 'Reflect 101',
    cv2.BORDER_TRANSPARENT: 'Transparent',
    cv2.BORDER_ISOLATED: 'Isolated',
}

DEFAULT_BORDER_VALUE = (255, 0, 255)
