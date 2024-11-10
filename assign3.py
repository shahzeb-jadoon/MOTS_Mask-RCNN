"""
By: Shahzeb Jadoon & Prerit Mittal

assign3.py

MOTS Assignment
Robot Perception
"""

# Prerequisites
import numpy as np
from torchvision.transforms import v2
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Data Preparation
def decode_rle(encoded_mask, height, width):
    """
    Decodes a run-length encoded mask string into a binary mask array.

    Args:
        encoded_mask (str): Run-length encoded mask string.
        height (int): Height of the mask.
        width (int): Width of the mask.

    Returns:
        numpy.ndarray: Decoded binary mask as a NumPy array.
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    encoded_mask = encoded_mask.split()
    
    for i in range(0, len(encoded_mask), 2):
        
        start = int(encoded_mask[i]) - 1
        length = int(encoded_mask[i + 1])
        mask[start:start + length] = 1
        
    return mask.reshape((height, width), order='F')

def parse_gt_file(file_path):
    """
    Parses the ground truth file containing image IDs and their corresponding masks.
    This function is adapted to handle the MOTS dataset format, where each line contains:
    frame_id track_id class_id height width encoded_pixels

    Args:
        file_path (str): Path to the ground truth file.

    Returns:
        list: List of tuples, where each tuple contains the image ID and its corresponding mask.
    """
    data = []
    
    with open(file_path, 'r') as f:
        
        for line in f:
            # Split by spaces and unpack only the first 5 values
            parts = line.strip().split()
            # Extract the frame ID, track ID, class ID, height, and width
            frame_id, track_id, class_id, height, width = parts[:5]
            # Join the remaining parts as the encoded pixels
            encoded_pixels = ' '.join(parts[5:])
            height = int(height)
            width = int(width)
            # Create image_id from frame and track IDs
            image_id = f"{frame_id}_{track_id}"
            # Decode the run-length encoded mask
            mask = decode_rle(encoded_pixels, height, width)
            # Append the image ID and mask to the data list
            data.append((image_id, mask))
            
    return data

# Data Augmentation
class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading images and their corresponding masks and labels.
    """
    def __init__(self, data, transform=None):
        """
        Initializes the dataset.

        Args:
            data (list): List of tuples, where each tuple contains the image ID and its mask.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform if transform else v2.Compose([
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), # Apply color jitter
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Apply Gaussian blur
            v2.ToTensor(), # Convert image to PyTorch tensor
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize image
        ])

    def __getitem__(self, idx):
        """
        Loads and returns the image and its corresponding target at the given index.

        Args:
            idx (int): Index of the data point to load.

        Returns:
            tuple: A tuple containing the image and its target dictionary.
                   The target dictionary contains the following keys:
                       - 'masks': Tensor of masks for the objects in the image.
                       - 'labels': Tensor of labels for the objects in the image.
                       - 'boxes': Tensor of bounding boxes for the objects in the image.
        """
        image_id, mask = self.data[idx]
        image = cv2.imread(f"images/{image_id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        
        if self.transform:
            image = self.transform(image)
        
        target = {
            'masks': torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0),
            'labels': torch.ones((1,), dtype=torch.int64), # Assuming only one class (human)
            'boxes': self._get_bbox(mask)
        }
        
        return image, target

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
            int: Number of data points.
        """
        return len(self.data)
    
    def _get_bbox(self, mask):
        """
        Calculates the bounding box for a given mask.

        Args:
            mask (numpy.ndarray): Binary mask array.

        Returns:
            torch.Tensor: Bounding box coordinates as a tensor (xmin, ymin, xmax, ymax).
        """
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        
        return torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

class SiameseDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading image pairs and their corresponding labels for Siamese Network training.
    """
    def __init__(self, data, transform=None):
        """
        Initializes the dataset.

        Args:
            data (list): List of tuples, where each tuple contains image information (e.g., ID, mask).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform if transform else v2.Compose([
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), # Apply color jitter
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Apply Gaussian blur
            v2.ToTensor(), # Convert image to PyTorch tensor
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize image
        ])
        self.pairs = self._create_pairs() # Generate pairs of images for training

    def _create_pairs(self):
        """
        Creates pairs of images with labels indicating similarity (1 for similar, -1 for dissimilar).

        Returns:
            list: List of tuples, where each tuple contains indices of two images and their similarity label.
        """
        pairs = []
        
        # Create positive pairs (same person)
        for i in range(len(self.data)):
            
            for j in range(i + 1, len(self.data)):
                
                if self.data[i][0].split('_')[0] == self.data[j][0].split('_')[0]: # Check if same person ID
                    pairs.append((i, j, 1)) # Similar pair
                    
                else:
                    
                    if len(pairs) % 2 == 0: # Balance dataset with dissimilar pairs
                        pairs.append((i, j, -1)) # Dissimilar pair
                        
        return pairs

    def __getitem__(self, idx):
        """
        Loads and returns a pair of images and their corresponding label at the given index.

        Args:
            idx (int): Index of the data point to load.

        Returns:
            tuple: A tuple containing two image tensors and their similarity label tensor.
        """
        idx1, idx2, label = self.pairs[idx]
        
        # Load first image
        img1_id = self.data[idx1][0]
        img1 = cv2.imread(f"images/{img1_id}.jpg")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        
        # Load second image
        img2_id = self.data[idx2][0]
        img2 = cv2.imread(f"images/{img2_id}.jpg")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of image pairs in the dataset.

        Returns:
            int: Number of image pairs.
        """
        return len(self.pairs)

# Mask R-CNN
def get_model_instance_segmentation(num_classes):
    """
    Loads a pre-trained Mask R-CNN model and modifies its head for the given number of classes.

    Args:
        num_classes (int): Number of classes to be segmented.

    Returns:
        torchvision.models.detection.mask_rcnn.MaskRCNN: Modified Mask R-CNN model.
    """
    model = maskrcnn_resnet50_fpn_v2(pretrained=True)

    # Freeze all parameters in the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Replace the existing classifier and mask predictor heads with new ones
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # Get only trainable parameters (the heads)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    return model, params_to_optimize


# Tracker
class Siamese_Network(nn.Module):
    """
    Siamese Network for feature extraction and similarity comparison.
    """
    def __init__(self):
        """
        Initializes the Siamese Network with convolutional and fully connected layers.
        """
        super(Siamese_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 22 * 22, 256)  # Adjust input size if needed
        self.fc2 = nn.Linear(256, 256)

    def forward_one(self, x):
        """
        Forward pass for a single branch of the Siamese Network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the branch.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 22 * 22)  # Adjust input size if needed
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        """
        Forward pass for the Siamese Network with two input branches.

        Args:
            input1 (torch.Tensor): Input tensor for the first branch.
            input2 (torch.Tensor): Input tensor for the second branch.

        Returns:
            tuple: A tuple containing the output tensors of the two branches.
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Main
def train_mask_rcnn(model, dataloader, optimizer, device, num_epochs):
    """
    Trains the Mask R-CNN model.

    Args:
        model: The Mask R-CNN model to train.
        dataloader: DataLoader for the training dataset.
        optimizer: Optimizer to use for training.
        device: Device to use for training (CPU or GPU).
        num_epochs: Number of epochs to train for.
    """
    model.to(device) # Move model to the specified device
    
    for epoch in range(num_epochs):
        
        model.train() # Set the model to training mode
        
        for images, targets in dataloader:
            
            images = list(image.to(device) for image in images) # Move images to device
            
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # Move targets to device
            loss_dict = model(images, targets) # Calculate loss
            
            losses = sum(loss for loss in loss_dict.values()) # Sum all losses
            optimizer.zero_grad() # Reset gradients
            losses.backward() # Backpropagate loss
            optimizer.step() # Update model parameters
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}") # Print epoch loss

def train_siamese_network(model, dataloader, optimizer, device, num_epochs):
    """
    Trains the Siamese Network.

    Args:
        model: The Siamese Network model to train.
        dataloader: DataLoader for the training dataset.
        optimizer: Optimizer to use for training.
        device: Device to use for training (CPU or GPU).
        num_epochs: Number of epochs to train for.
    """
    model.to(device) # Move model to the specified device
    criterion = torch.nn.CosineEmbeddingLoss() # Use Cosine Embedding Loss
    
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        total_loss = 0.0 # Initialize total loss for the epoch
        num_batches = 0 # Initialize count of batches
        
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device) # Move data to device

            output1, output2 = model(img1, img2) # Get outputs from the Siamese Network
            loss = criterion(output1, output2, labels) # Calculate loss

            optimizer.zero_grad() # Reset gradients
            loss.backward() # Backpropagate loss
            optimizer.step() # Update model parameters
            
            total_loss += loss.item() # Accumulate loss
            num_batches += 1 # Increment batch count
        
        avg_loss = total_loss / num_batches # Calculate average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}") # Print epoch loss

def main():
    """
    Main function to load data, initialize models, and train them.
    """
    # Configuration
    gt_file = 'D:\\RIT\\Classes\\Fall_24\\Robot_Perception\\assign3\\MOTS\\train\\MOTS20-02\\gt\\gt.txt' # Path to the ground truth file
    num_classes = 2 # Number of classes (background + human)
    batch_size = 4 # Batch size for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA if available
    num_epochs_mask_rcnn = 10 # Number of epochs to train Mask R-CNN
    num_epochs_siamese = 10 # Number of epochs to train Siamese Network

    # Load and preprocess data
    data = parse_gt_file(gt_file) # Parse the ground truth file
    mask_rcnn_dataset = CustomDataset(data=data) # Create the custom dataset for Mask R-CNN
    siamese_dataset = SiameseDataset(data=data) # Create the custom dataset for Siamese Network
    
    # Create DataLoaders for both models
    mask_rcnn_dataloader = torch.utils.data.DataLoader(
        mask_rcnn_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Use 4 worker processes for data loading
        collate_fn=lambda x: tuple(zip(*x)) # Custom collate function for the DataLoader
    )
    
    siamese_dataloader = torch.utils.data.DataLoader(
        siamese_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Initialize models and optimizers
    mask_rcnn_model, params_to_optimize = get_model_instance_segmentation(num_classes)
    mask_rcnn_optimizer = torch.optim.SGD(
        params_to_optimize, # Optimize only the unfrozen parameters
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    siamese_model = Siamese_Network()
    siamese_optimizer = torch.optim.Adam(siamese_model.parameters())

    # Train models
    print("Training Mask R-CNN...")
    train_mask_rcnn(mask_rcnn_model, mask_rcnn_dataloader, mask_rcnn_optimizer, 
                   device, num_epochs_mask_rcnn)
    
    print("Training Siamese Network...")
    train_siamese_network(siamese_model, siamese_dataloader, siamese_optimizer, 
                          device, num_epochs_siamese)

if __name__ == "__main__":
    main()