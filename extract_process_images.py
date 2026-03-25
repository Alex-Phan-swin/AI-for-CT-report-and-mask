"""
Extract and Process AbdomenAtlas3.0Mini Dataset
Comprehensive script to load, analyze, and prepare data for medical imaging tasks
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json

class AbdomenAtlasProcessor:
    """Process AbdomenAtlas3.0Mini dataset"""
    
    def __init__(self, cache_dir="./dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.dataset = None
        self.metadata_df = None
        
    def load_dataset(self):
        """Load the dataset from cache"""
        print("\n📥 Loading dataset from cache...\n")
        
        try:
            self.dataset = load_dataset(
                "AbdomenAtlas/AbdomenAtlas3.0Mini",
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            print("✅ Dataset loaded successfully!\n")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}\n")
            return False
    
    def extract_metadata(self):
        """Extract metadata from all splits"""
        print("📊 Extracting metadata from all splits...\n")
        
        all_data = []
        
        for split_name, split_data in self.dataset.items():
            print(f"  Processing {split_name} split ({len(split_data)} samples)...")
            
            for idx, sample in enumerate(tqdm(split_data, desc=split_name, leave=False)):
                row = {
                    'split': split_name,
                    'index': idx,
                    'bdmap_id': sample.get('BDMAP ID', f'Unknown_{idx}')
                }
                all_data.append(row)
        
        self.metadata_df = pd.DataFrame(all_data)
        print(f"\n✅ Extracted {len(self.metadata_df)} metadata records\n")
        return self.metadata_df
    
    def analyze_dataset(self):
        """Analyze and display dataset statistics"""
        print("=" * 80)
        print("📈 DATASET ANALYSIS")
        print("=" * 80 + "\n")
        
        # Display dataset structure
        print("🔍 DATASET STRUCTURE:\n")
        for split_name, split_data in self.dataset.items():
            print(f"  {split_name.upper()}:")
            print(f"    • Samples: {len(split_data):,}")
            print(f"    • Features: {list(split_data.column_names)}")
            print(f"    • Feature Types: {split_data.features}\n")
        
        # Display metadata statistics
        if self.metadata_df is not None:
            print("📋 METADATA STATISTICS:\n")
            print(f"  Total Records: {len(self.metadata_df):,}")
            print(f"  Unique Splits: {self.metadata_df['split'].nunique()}")
            print(f"  Unique BDMAP IDs: {self.metadata_df['bdmap_id'].nunique()}\n")
            
            # Split distribution
            print("  Distribution by Split:")
            split_counts = self.metadata_df['split'].value_counts()
            for split, count in split_counts.items():
                percentage = (count / len(self.metadata_df)) * 100
                print(f"    • {split}: {count:,} ({percentage:.1f}%)")
            
            print("\n  Sample BDMAP IDs:")
            for bdmap_id in self.metadata_df['bdmap_id'].head(5).values:
                print(f"    • {bdmap_id}")
        
        print("\n" + "=" * 80)
    
    def save_metadata(self, output_file="dataset_metadata.csv"):
        """Save metadata to CSV file"""
        if self.metadata_df is None:
            print("⚠️  No metadata to save. Run extract_metadata() first.\n")
            return
        
        output_path = Path(output_file)
        self.metadata_df.to_csv(output_path, index=False)
        print(f"✅ Metadata saved to: {output_path.absolute()}\n")
        print(f"   Columns: {list(self.metadata_df.columns)}")
        print(f"   Total rows: {len(self.metadata_df)}\n")
    
    def save_config(self, output_file="dataset_config.json"):
        """Save dataset configuration"""
        config = {
            "dataset_name": "AbdomenAtlas3.0Mini",
            "total_samples": len(self.metadata_df) if self.metadata_df is not None else 0,
            "splits": {}
        }
        
        for split_name, split_data in self.dataset.items():
            config["splits"][split_name] = {
                "samples": len(split_data),
                "features": list(split_data.column_names)
            }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to: {output_path.absolute()}\n")


def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print(" " * 15 + "ABDOMEN ATLAS 3.0 MINI - EXTRACTION & PROCESSING")
    print("=" * 80)
    
    # Initialize processor
    processor = AbdomenAtlasProcessor()
    
    # Step 1: Load dataset
    if not processor.load_dataset():
        print("❌ Failed to load dataset. Exiting.\n")
        return False
    
    # Step 2: Extract metadata
    processor.extract_metadata()
    
    # Step 3: Analyze dataset
    processor.analyze_dataset()
    
    # Step 4: Save outputs
    print("💾 SAVING OUTPUTS\n")
    processor.save_metadata("dataset_metadata.csv")
    processor.save_config("dataset_config.json")
    
    # Summary
    print("=" * 80)
    print("✅ EXTRACTION AND PROCESSING COMPLETE!")
    print("=" * 80)
    print("\n📄 Generated Files:")
    print("   • dataset_metadata.csv - Full metadata of all samples")
    print("   • dataset_config.json - Dataset configuration\n")
    
    print("📝 Next Steps:")
    print("   1. Review dataset_metadata.csv to understand the data structure")
    print("   2. Use BDMAP IDs to access images from external source")
    print("   3. Implement your medical imaging analysis\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

