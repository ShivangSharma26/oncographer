import torchvision.transforms as transforms

def get_transforms(image_size):
    """
    Returns train and val/test transforms.
    The training transform includes data augmentation to prevent overfitting
    and make the model robust to stain variations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Medical images can be rotated randomly as tissue has no fixed orientation
        transforms.RandomRotation(90), 
        # Stain variation is a real issue, color jitter helps
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        # Standard ImageNet normalization (often a good default even for medical if using transfer learning)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
