import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PatchDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir (str): Path to the mock_pcam dataset.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # In PCam, classes are 0 (healthy) and 1 (tumor)
        for class_label in [0, 1]:
            class_dir = os.path.join(self.data_dir, str(class_label))
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_label)
                        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(config, train_transform, val_transform):
    """
    Returns train, val, and test dataloaders based on config settings.
    """
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    train_dataset = PatchDataset(data_dir, split="train", transform=train_transform)
    val_dataset = PatchDataset(data_dir, split="val", transform=val_transform)
    test_dataset = PatchDataset(data_dir, split="test", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
