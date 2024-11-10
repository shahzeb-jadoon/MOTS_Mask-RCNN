"""
Data Preparation
data_prep.py

Parse the ground truth
"""

import numpy as np

def decode_rle(encoded_mask, height, width):
    
    mask = np.zeros(height * width, dtype=np.uint8)
    encoded_mask = encoded_mask.split()
    
    for i in range(0, len(encoded_mask), 2):
        
        start = int(encoded_mask[i]) - 1
        length = int(encoded_mask[i + 1])
        mask[start:start + length] = 1
        
    return mask.reshape((height, width), order='F')

def parse_gt_file(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        
        for line in f:
            
            image_id, height, width, encoded_pixels = line.strip().split(',')
            height = int(height)
            width = int(width)
            mask = decode_rle(encoded_pixels, height, width)
            data.append((image_id, mask))
            
    return data