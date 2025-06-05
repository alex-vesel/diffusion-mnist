import os
import re
import numpy as np


def recurse_dir_for_imgs(path, file_extension='jpg'):
    # get paths to image files with the specified extension
    img_paths = []
    for root, dirs, files in os.walk(path):
        # Filter files with the specified extension
        for file in files:
            if file.lower().endswith(f'.{file_extension}'):
                img_paths.append(os.path.join(root, file))

    return img_paths
