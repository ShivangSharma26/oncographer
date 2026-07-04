import os
import numpy as np
from PIL import Image

def generate_mock_data(base_dir="data/mock_pcam", num_train=100, num_val=20, num_test=20):
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    
    for split, count in splits.items():
        # Create directories for class 0 (healthy) and class 1 (tumor)
        for class_label in [0, 1]:
            dir_path = os.path.join(base_dir, split, str(class_label))
            os.makedirs(dir_path, exist_ok=True)
            
            num_images = count // 2
            for i in range(num_images):
                # Generate a random 96x96 RGB image
                if class_label == 1:
                    # Tumor: slightly red tinted
                    img_array = np.random.randint(100, 255, (96, 96, 3), dtype=np.uint8)
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.5, 0, 255)
                else:
                    # Healthy: slightly blue/pink tinted
                    img_array = np.random.randint(100, 255, (96, 96, 3), dtype=np.uint8)
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.5, 0, 255)
                
                img = Image.fromarray(img_array)
                img.save(os.path.join(dir_path, f"mock_patch_{split}_{class_label}_{i}.png"))

    print(f"Mock data successfully generated in {base_dir}")

if __name__ == "__main__":
    generate_mock_data()
