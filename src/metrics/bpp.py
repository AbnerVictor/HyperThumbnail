from basicsr.utils.registry import METRIC_REGISTRY
import cv2
import os

@METRIC_REGISTRY.register()
def calculate_bpp_from_file(img):
    """Calculate file_size.
    """
    jpg_size_bit = os.path.getsize(img)*8
    h,w, _ = cv2.imread(img).shape
    pixel_number = h*w
    bpp = jpg_size_bit/pixel_number
    return bpp
