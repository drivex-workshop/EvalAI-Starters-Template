
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

# for Python 3.7.5
install("numpy==1.21.0")
install("scipy==1.5.4")
install("requests==2.30.0")
install("numba==0.56.4")

#install_local_package("package_folder_name")

# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "github/requirements.txt"])

from .main import evaluate
