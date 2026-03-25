"""
Download AbdomenAtlas3.0Mini dataset from Hugging Face
This script downloads the dataset and saves it to local cache for fast access
"""

from datasets import load_dataset
from pathlib import Path
import sys

def download_abdomen_atlas():
    """
    Download the AbdomenAtlas3.0Mini dataset from Hugging Face
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "ABDOMEN ATLAS 3.0 MINI - DATASET DOWNLOAD")
    print("=" * 80)
    
    # Create and verify cache directory
    cache_dir = Path("./dataset_cache")
    cache_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Cache Directory: {cache_dir.absolute()}")
    print(f"   (Dataset will be stored here for future use)\n")
    
    try:
        print("🔄 Downloading dataset... (This may take a few minutes)\n")
        
        # Load dataset with caching
        ds = load_dataset(
            "AbdomenAtlas/AbdomenAtlas3.0Mini",
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        
        print("\n" + "=" * 80)
        print("✅ DATASET DOWNLOADED SUCCESSFULLY!")
        print("=" * 80)
        
        # Display dataset information
        print("\n📊 DATASET INFORMATION\n")
        total_samples = 0
        
        for split_name, split_data in ds.items():
            num_samples = len(split_data)
            total_samples += num_samples
            
            print(f"  {split_name.upper()}:")
            print(f"    • Samples: {num_samples:,}")
            print(f"    • Features: {list(split_data.column_names)}")
            print(f"    • Sample: {split_data[0]}\n")
        
        print(f"  TOTAL SAMPLES: {total_samples:,}\n")
        
        print("=" * 80)
        print(f"\n✅ Dataset saved to: {cache_dir.absolute()}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   {str(e)}\n")
        
        print("📝 TROUBLESHOOTING:\n")
        print("  1️⃣  Install required packages:")
        print("      pip install datasets huggingface-hub\n")
        
        print("  2️⃣  If permission denied, login with:")
        print("      huggingface-cli login")
        print("      (Get token: https://huggingface.co/settings/tokens)\n")
        
        print("  3️⃣  Accept dataset license at:")
        print("      https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini\n")
        
        return False


if __name__ == "__main__":
    success = download_abdomen_atlas()
    sys.exit(0 if success else 1)
