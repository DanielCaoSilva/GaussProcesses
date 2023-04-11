from pathlib import Path
import sys
import os
#sys.path.append(
sys.path.append(Path(os.getcwd()).parent.__str__())

print(sys.path)
