from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
# convert dataset annotation to semantic segmentation map

# define class and plaette for better visualization
classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

classes = ('STAS', 'background')
palette = [[1, 1, 1],[0, 0, 0]]
@DATASETS.register_module()
class BinaryDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', 
                         seg_map_suffix='.png', 
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None