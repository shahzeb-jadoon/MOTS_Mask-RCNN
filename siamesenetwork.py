import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import random
import cv2
from torch.amp import autocast, GradScaler
from PIL import Image
import os

class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        person_ids = {}
        max_pairs_per_person = 10

        # Group images by person ID
        for i, (image_id, _) in enumerate(self.data):
            person_id = image_id.split('_')[1]
            if person_id not in person_ids:
                person_ids[person_id] = []
            person_ids[person_id].append(i)

        # Create positive pairs (same person)
        positive_pairs = []
        for indices in person_ids.values():
            if len(indices) > 1:
                for i in range(min(len(indices), max_pairs_per_person)):
                    for j in range(i + 1, min(len(indices), max_pairs_per_person)):
                        positive_pairs.append((indices[i], indices[j], 1))

        # Create negative pairs (different persons)
        negative_pairs = []
        person_ids_list = list(person_ids.keys())
        for i in range(len(positive_pairs)):
            id1, id2 = random.sample(person_ids_list, 2)
            idx1 = random.choice(person_ids[id1])
            idx2 = random.choice(person_ids[id2])
            negative_pairs.append((idx1, idx2, -1))

        # Combine and shuffle pairs
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        print(f"Created {len(pairs)} total pairs")
        return pairs

    def __getitem__(self, idx):
        (idx1, idx2, label) = self.pairs[idx]

        # Load first image using PIL
        img1_id = self.data[idx1][0].split('_')[0].zfill(6)
        img1 = Image.open(f"D:\\Study\\RIT HW and Assignments\\Robot Perception (CMPE789)\\Homework\\HW3\\MOTS_Mask-RCNN\\MOTS\\train\\MOTS20-02\\img1\\{img1_id}.jpg")
        
        # Load second image using PIL
        img2_id = self.data[idx2][0].split('_')[0].zfill(6)
        img2 = Image.open(f"D:\\Study\\RIT HW and Assignments\\Robot Perception (CMPE789)\\Homework\\HW3\\MOTS_Mask-RCNN\\MOTS\\train\\MOTS20-02\\img1\\{img2_id}.jpg")

        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Output: 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output: 64 x 64 x 64
            
            # Second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output: 128 x 32 x 32
            
            # Third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output: 256 x 16 x 16
            
            # Fourth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Output: 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output: 512 x 8 x 8
        )
        
        # Calculate the flattened size
        self.fc_input_size = 512 * 8 * 8
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

def train_siamese_network(model, dataloader, optimizer, num_epochs, device):
    model.to(device)
    criterion = ContrastiveLoss()
    scaler = GradScaler()
    accumulation_steps = 4

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        running_loss = 0
        last_loss = 0

        for i, (img1, img2, labels) in enumerate(dataloader):
            try:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                # Mixed precision training
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output1, output2 = model(img1, img2)
                    loss = criterion(output1, output2, labels)
                    scaled_loss = loss / accumulation_steps

                # Backward pass
                scaler.scale(scaled_loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item()
                running_loss += loss.item()

                # Print progress
                if i % 100 == 99:    # Print every 100 mini-batches
                    last_loss = running_loss / 100
                    print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {last_loss:.4f}')
                    running_loss = 0.0

                # Clear cache periodically
                if i % 50 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error in batch {i}: {str(e)}")
                continue

        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed")
        print(f"Average Loss: {avg_loss:.4f}")
        print("-" * 60)

def parse_gt_file(file_path):
    """
    Parse the ground truth file containing tracking information.
    
    Args:
        file_path (str): Path to the ground truth file
        
    Returns:
        list: List of tuples containing (image_id, None) pairs
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure we have minimum required fields
                    frame_id, track_id = parts[:2]
                    image_id = f"{frame_id}_{track_id}"
                    data.append((image_id, None))
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error parsing ground truth file: {str(e)}")
        raise
    
    return data

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # File paths
        gt_file = "D:\\Study\\RIT HW and Assignments\\Robot Perception (CMPE789)\\Homework\\HW3\\MOTS_Mask-RCNN\\MOTS\\train\\MOTS20-02\\gt\\gt.txt"
        
        # Parse ground truth file
        data = parse_gt_file(gt_file)
        
        # Create dataset and dataloader
        siamese_dataset = SiameseDataset(data=data)
        siamese_dataloader = DataLoader(
            siamese_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model and optimizer
        siamese_model = SiameseNetwork()
        siamese_optimizer = optim.Adam(siamese_model.parameters(), lr=0.0001)
        
        # Train the network
        print("Starting Siamese Network training...")
        train_siamese_network(siamese_model, siamese_dataloader, siamese_optimizer, num_epochs=10, device=device)
        
        # Save the trained model
        torch.save(siamese_model.state_dict(), 'siamese_model.pth')
        print("Training completed. Model saved as 'siamese_model.pth'")
        
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()