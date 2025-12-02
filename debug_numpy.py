import sys
print(f"Python version: {sys.version}")
try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
except Exception as e:
    print(f"Error importing numpy: {e}")
