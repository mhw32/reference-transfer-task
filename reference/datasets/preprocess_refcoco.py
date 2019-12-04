import os
import sys

refer_dir = os.path.join(os.path.dirname(__file__), 'refer')
sys.path.append(refer_dir)
from refer import REFER


if __name__ == "__main__":
    dataset = REFER(
        '/mnt/fs5/wumike/datasets/refer_datasets', 
        'refcocog', 
        'google',
    )

