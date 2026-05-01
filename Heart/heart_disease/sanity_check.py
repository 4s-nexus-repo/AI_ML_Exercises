def main():
    import sys
    print("=" * 55)
    print("  Python & Library Sanity Check")
    print("=" * 55)

    print(f"\nPython            : {sys.version.split()[0]}")

    import numpy as np
    print(f"numpy             : {np.__version__}")

    import pandas as pd
    print(f"pandas            : {pd.__version__}")

    import sklearn
    print(f"scikit-learn      : {sklearn.__version__}")

    import matplotlib
    print(f"matplotlib        : {matplotlib.__version__}")

    import seaborn as sns
    print(f"seaborn           : {sns.__version__}")

    import scipy
    print(f"scipy             : {scipy.__version__}")

    import torch
    print(f"torch             : {torch.__version__}")

    import torchvision
    print(f"torchvision       : {torchvision.__version__}")

    import tensorflow as tf
    print(f"tensorflow        : {tf.__version__}")

    import keras
    print(f"keras             : {keras.__version__}")

    import fastapi
    print(f"fastapi           : {fastapi.__version__}")

    def print_hardware() -> None:
        print("\n" + "=" * 55)
        print("  Hardware Backends")
        print("=" * 55)
        print(f"\ntorch CUDA available : {torch.cuda.is_available()}")
        print(f"torch MPS  available : {torch.backends.mps.is_available()}")
        tf_devices = [d.device_type for d in tf.config.list_physical_devices()]
        print(f"tensorflow devices   : {tf_devices}")

        print("\n--- What this means for you ---")
        if torch.cuda.is_available():
            print("  CUDA GPU detected  → Windows/Linux with NVIDIA GPU")
            print(f"  GPU name           : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version       : {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            print("  MPS detected       → Apple Silicon Mac (M1/M2/M3)")
            print("  Neural networks will use Mac GPU automatically")
        else:
            print("  No GPU detected    → CPU only mode")

        if 'GPU' in tf_devices:
            print("  TensorFlow will use GPU automatically")
        else:
            print("  TensorFlow running on CPU")

    print_hardware()

    print("\n" + "=" * 55)
    print("  All checks passed!")
    print("=" * 55)


if __name__ == '__main__':
    main()