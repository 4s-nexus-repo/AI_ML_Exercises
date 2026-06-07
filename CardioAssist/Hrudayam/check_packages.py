packages = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "scipy",
    "torch",
    "torchvision",
    "tensorflow",
    "fastapi"
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"{pkg} ✅ installed")
    except ImportError:
        print(f"{pkg} ❌ NOT installed")



import torch
import tensorflow as tf

def print_hardware() -> None:
    print("\n--- Hardware backends ---")
    print(f"torch CUDA available : {torch.cuda.is_available()}")
    print(f"torch MPS  available : {torch.backends.mps.is_available()}")  # Apple Silicon GPU
    print(f"tensorflow devices   : {[d.device_type for d in tf.config.list_physical_devices()]}")

print_hardware()


print(f"tensorflow {tf.__version__}")