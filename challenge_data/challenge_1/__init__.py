
# Q. How to install custom python pip packages?

# A. Uncomment the below code to install the custom python packages.

import os
import subprocess
import sys
from pathlib import Path

def install(package):
    # Install a pip python package

    # Args:
    #     package ([str]): Package name with version

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_local_package(folder_name):
    # Install a local python package

    # Args:
    #     folder_name ([str]): name of the folder placed in evaluation_script/

    subprocess.check_output(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        os.path.join(str(Path(__file__).parent.absolute()) + folder_name),
    ]
)

# install("pytorch3d==0.7.2")
# install("numpy==1.23.5")
# install("scipy==1.9.3") # 1.13.1
# install("numba==0.56.4") # 0.60
#install("torch==1.12.1") # 1.12.1+cu113
#install("torchvision==0.13.1") #0.13.1+cu113
#install("torchaudio==0.12.1")

#install_local_package("package_folder_name")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "github/requirements.txt"])

from .main import evaluate
