"""
Mask R-CNN
mask_rcnn.py

Fine-tune a pre-trained Mask R-CNN model for instance segmentation
"""

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = maskrcnn_resnet50_fpn_v2(pretrained=True)


# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False
    
    
# Only fine-tune the heads for classification and mask prediction
params_to_optimize = [p for p in model.parameters() if p.requires_grad]