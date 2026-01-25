import sys
import os

# Ensure current dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print(f"Checking imports from: {current_dir}")

try:
    print("Importing config...")
    import config
    print("Config imported.")
    
    print("Importing transforms...")
    import transforms
    print("Transforms imported.")
    
    print("Importing dataset...")
    import dataset
    print("Dataset imported.")
    
    print("Importing model...")
    import model
    print("Model imported.")
    
    print("Importing utils...")
    import utils
    print("Utils imported.")
    
    print("Importing explain...")
    import explain
    print("Explain imported.")
    
    print("Importing train...")
    import train
    print("Train imported.")
    
    print("ALL MODULES IMPORTED SUCCESSFULLY.")
    
except Exception as e:
    print(f"IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
