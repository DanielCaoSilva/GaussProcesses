from setuptools import setup
from pathlib import Path
import os, sys
print(sys.path)
sys.path.insert(0, Path(os.getcwd()).__str__())
print(sys.path)

# setup(
# 	name='GaussProcesses',
# 	version='',
# 	packages=['src'],
# 	url='',
# 	license='',
# 	author='dcaos & jrisk',
# 	author_email='',
# 	description=''
# )
