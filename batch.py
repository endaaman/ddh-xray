import os

import numpy as np
from tqdm import tqdm

from endaaman import Commander, pad_to_size

from datasets import XRBBDataset


class C(Commander):
    def arg_crop_roi(self, parser):
        parser.add_argument('--dest', '-d', default='data/rois/gt')

    def run_crop_roi(self):
        ds = XRBBDataset(target='all')
        os.makedirs(self.a.dest, exist_ok=True)
        for item in tqdm(ds.items):
            image = item.image
            bbs = np.round(item.bb.values[:, :4]).astype(np.int64)
            # top-left edge
            left, top = np.min(bbs[:, [0, 1]], axis=0)
            # bottom-right edge
            right, bottom = np.max(bbs[:, [2, 3]], axis=0)
            cropped = image.crop((left, top, right, bottom))
            cropped = pad_to_size(cropped, size=(512, 256))
            cropped.save(os.path.join(self.a.dest, f'{item.name}.jpg'))


if __name__ == '__main__':
    cmd = C()
    cmd.run()
