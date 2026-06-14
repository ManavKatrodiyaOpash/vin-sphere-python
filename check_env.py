import sys
import os

print("=========================================")
print("     VIN Engine Environment Diagnostics  ")
print("=========================================")

print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Current Working Directory: {os.getcwd()}")

# Check PyTorch
print("\nChecking PyTorch...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  [ERROR] PyTorch is not installed in this environment.")
except OSError as e:
    print(f"  [ERROR] PyTorch failed to load with OSError: {e}")
    if "1114" in str(e):
        print("\n>>> DIAGNOSIS: WinError 1114 (DLL initialization routine failed) <<<")
        print("This is a common compatibility issue when running PyTorch on an unsupported Python version on Windows.")
        print("Your current environment is running Python 3.14. PyTorch does not officially support Python 3.14 yet.")
        print("\nRECOMMENDED REMEDY:")
        print("1. Install Python 3.12 (standard stable release).")
        print("2. Create a virtual environment using Python 3.12: ")
        print("   python3.12 -m venv .venv")
        print("3. Activate the environment and install GPU-enabled PyTorch:")
        print("   .venv\\Scripts\\activate")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("4. This will resolve all DLL loading errors and enable GPU acceleration.")

# Check CatBoost
print("\nChecking CatBoost...")
try:
    import catboost
    print(f"  CatBoost version: {catboost.__version__}")
except ImportError:
    print("  [WARNING] CatBoost is not installed.")

# Check Streamlit
print("\nChecking Streamlit...")
try:
    import streamlit
    print(f"  Streamlit version: {streamlit.__version__}")
except ImportError:
    print("  [WARNING] Streamlit is not installed.")

print("=========================================")
