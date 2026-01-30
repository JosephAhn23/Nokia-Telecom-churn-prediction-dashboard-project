"""
Complete Pipeline Runner
Runs the entire churn prediction pipeline: data generation → model training
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {description}")
        print(result.stderr)
        return False
    else:
        print(result.stdout)
        print(f"{description} completed successfully!")
        return True

def main():
    """Run the complete pipeline"""
    print("\n" + "="*70)
    print("TELECOM CHURN PREDICTION PIPELINE")
    print("="*70)
    
    steps = [
        ("python data_generator.py", "Generating telecom churn dataset"),
        ("python train_models.py", "Training ML models (Random Forest, XGBoost, Neural Network)"),
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            print(f"\nPipeline failed at: {description}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("   1. Launch dashboard: streamlit run dashboard.py")
    print("   2. Or use Docker: docker-compose up")
    print("\n")

if __name__ == "__main__":
    main()

