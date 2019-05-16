import numpy as np

GAMMA_DECODE = 2.2  # sensible constant for gamma decoding


def gamma_decode(img):
    return (img / 255.) ** GAMMA_DECODE


def gamma_encode(img):
    return ((img ** (1. / GAMMA_DECODE)) * 255).astype(np.int)


def gamma_adjust(bg, fg, gamma_const=0.9):
    """
    adjust fg brightness level according to bg brightness level with some constant factor. Note that
    fg and bg are gamma decoded values from cv2.imread()
    :param bg: background numpy array of shape (w, h, 3)
    :param fg: foreground numpy array of shape (w, h, 3)
    :return: new adjusted fg
    """
    bg_decoded = gamma_decode(bg)
    fg_decoded = gamma_decode(fg)
    gamma_adjust_lvl = np.mean(fg_decoded) - np.mean(bg_decoded)
    gamma_adjust_lvl *= gamma_const
    fg_decoded = np.clip(fg_decoded - gamma_adjust_lvl, 0, 1)
    return gamma_encode(fg_decoded)
