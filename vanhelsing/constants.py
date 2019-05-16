import os

IMAGE_EXTS = ('.jpg', '.png', '.jpeg')


def filter_image_exts(x):
    _, ext = os.path.splitext(x)
    return ext.lower() in IMAGE_EXTS
