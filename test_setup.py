# test_setup.py

import sys

print("=" * 50)
print("ENVIRONMENT SETUP VERIFICATION")
print("=" * 50)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check numpy
try:
    import numpy as np
    print(f"2. NumPy Version: {np.__version__} ✓")
except ImportError:
    print("2. NumPy: NOT INSTALLED ✗")

# Check OpenCV
try:
    import cv2
    print(f"3. OpenCV Version: {cv2.__version__} ✓")
except ImportError:
    print("3. OpenCV: NOT INSTALLED ✗")

# Check Pillow
try:
    from PIL import Image
    import PIL
    print(f"4. Pillow Version: {PIL.__version__} ✓")
except ImportError:
    print("4. Pillow: NOT INSTALLED ✗")

# Check Matplotlib
try:
    import matplotlib
    print(f"5. Matplotlib Version: {matplotlib.__version__} ✓")
except ImportError:
    print("5. Matplotlib: NOT INSTALLED ✗")

# Check SciPy
try:
    import scipy
    print(f"6. SciPy Version: {scipy.__version__} ✓")
except ImportError:
    print("6. SciPy: NOT INSTALLED ✗")

print("\n" + "=" * 50)
print("Setup verification complete!")
print("=" * 50)