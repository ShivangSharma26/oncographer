import os
from datasets import load_dataset
from tqdm import tqdm

def download_subset(base_dir="data/subset_pcam", num_train=2000, num_val=400, num_test=400):
    print("Downloading actual PCam subset from Hugging Face...")
    
    # Load dataset with streaming to avoid downloading the whole 7GB if possible, 
    # or just load the splits with slice syntax.
    # The dataset on HF is 1aurent/PatchCamelyon
    
    splits_to_load = {
        "train": f"train[:{num_train}]",
        "valid": f"valid[:{num_val}]",
        "test": f"test[:{num_test}]"
    }

    dir_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    for hf_split, local_split in dir_mapping.items():
        print(f"Processing {local_split} split...")
        ds = load_dataset('1aurent/PatchCamelyon', split=splits_to_load[hf_split])
        
        for i, item in enumerate(tqdm(ds)):
            image = item['image']
            label = item['label']
            
            # Create directory for class
            dir_path = os.path.join(base_dir, local_split, str(label))
            os.makedirs(dir_path, exist_ok=True)
            
            # Save image
            img_path = os.path.join(dir_path, f"patch_{local_split}_{label}_{i}.png")
            image.save(img_path)

    print(f"Real subset data successfully downloaded and saved to {base_dir}")

if __name__ == "__main__":
    download_subset()
