"""
Master Script: Download and Process AbdomenAtlas3.0Mini Dataset
Runs all steps in sequence with progress tracking
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """
    Run a Python script and track execution
    
    Args:
        script_name: Name of the script to run
        description: Description of what the script does
    """
    print(f"\n{'=' * 80}")
    print(f"🔄 STEP: {description}")
    print(f"{'=' * 80}")
    print(f"   Running: {script_name}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path.cwd(),
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - COMPLETED\n")
            return True
        else:
            print(f"\n❌ {description} - FAILED (Exit code: {result.returncode})\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}\n")
        return False


def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print(" " * 10 + "ABDOMEN ATLAS 3.0 MINI - COMPLETE PIPELINE")
    print("=" * 80)
    print("\n📋 This script will:")
    print("   1. Download the AbdomenAtlas3.0Mini dataset")
    print("   2. Extract and process metadata")
    print("   3. Generate analysis and configuration files\n")
    
    success_count = 0
    total_steps = 2
    
    # Step 1: Download dataset
    if run_script("download_dataset.py", "Download Dataset"):
        success_count += 1
    
    # Step 2: Extract and process
    if run_script("extract_process_images.py", "Extract and Process Data"):
        success_count += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 PIPELINE SUMMARY")
    print("=" * 80)
    print(f"\n✅ Completed: {success_count}/{total_steps} steps\n")
    
    if success_count == total_steps:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!\n")
        print("📁 Generated Files:")
        print("   • dataset_cache/ - Downloaded dataset")
        print("   • dataset_metadata.csv - Extracted metadata")
        print("   • dataset_config.json - Dataset configuration\n")
        
        print("📝 Next Actions:")
        print("   1. Open dataset_metadata.csv to view sample data")
        print("   2. Review dataset_config.json for structure")
        print("   3. Implement your medical imaging model\n")
        return True
    else:
        print(f"⚠️  {total_steps - success_count} step(s) failed. Please check the output above.\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
