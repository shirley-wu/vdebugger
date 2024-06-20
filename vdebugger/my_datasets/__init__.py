from .gqa import GQADataset
from .vsr import VSRDataset
from .tallyqa import TallyQADataset
from .covr import COVRDataset
from .refcoco import RefCOCODataset
from .nlvr import NLVRDataset


def process_result(x):
    class ImagePatch:
        def __init__(self, left, right, upper, lower, *args, **kwargs):
            self.left = left
            self.right = right
            self.upper = upper
            self.lower = lower

            self.height = self.upper - self.lower
            self.width = self.right - self.left
            self.horizontal_center = (self.left + self.right) / 2
            self.vertical_center = (self.lower + self.upper) / 2

        def __repr__(self):
            return "ImagePatch(left={}, right={}, upper={}, lower={}, height={}, width={}, horizontal_center={}, vertical_center={})".format(
                self.left, self.right, self.upper, self.lower, self.height, self.width,
                self.horizontal_center, self.vertical_center,
            )

    # if x == 'None':  # that doesn't really make sense
    #     return None
    if isinstance(x, str) and x.startswith("ImagePatch"):
        try:
            return eval(x)
        except:
            print("Weird or invalid ImagePatch:", x)
    return x
