import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Orchestrate VIN Attribute Prediction Engine Pipeline")
    parser.add_argument("--sample_size", type=int, default=50000, help="Number of rows to sample for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    print("==========================================================")
    print("       VIN Attribute Prediction Pipeline Orchestrator     ")
    print("==========================================================")
    
    # 1. Check directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/attention_maps", exist_ok=True)
    
    # 2. Run target signal analysis and model training
    train_cmd = [
        sys.executable, "src/train.py",
        "--sample_size", str(args.sample_size),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr)
    ]
    
    print(f"Executing training command: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd)
    
    if result.returncode != 0:
        print("\n[ERROR] Pipeline training failed. Check stdout logs above.")
        sys.exit(result.returncode)
        
    print("\n==========================================================")
    print(" Pipeline completed successfully!")
    print(" - Target Signal Analysis Report: outputs/reports/target_signal_analysis_report.md")
    print(" - Evaluation Metrics: outputs/reports/model_evaluation_metrics.md")
    print(" - Transformer Checkpoint: models/transformer_best.pt")
    print(" - CatBoost Ensemble: models/catboost.pkl")
    print("==========================================================")
    print("\nYou can now launch the Streamlit verification app by running:")
    print("streamlit run app/streamlit_app.py")
    print("==========================================================")

if __name__ == "__main__":
    main()
