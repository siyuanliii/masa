
# Project Setup Instructions

## Step 1: Clone the Repository
Clone the project repository to your local machine using:
```bash
git clone https://github.com/siyuanliii/masa.git
```

## Step 2: Create and Activate the Conda Environment
Navigate to the project directory and create a Conda environment using:
```bash
conda env create -f environment.yml
conda activate masaenv
```
## Option 1: Automated Installation
### Step 3 : Run install_dependencies.sh
Run the `install_dependencies.sh` script to install the required dependencies:
```bash
sh install_dependencies.sh
```
If you encounter any issues, please refer to the manual installation instructions below.
Otherwise, you can skip to the next step.

## Option 2: Manual Installation

### Step 3: Install MMDetection 3.3.0
```bash
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install git+https://github.com/open-mmlab/mmdetection.git@v3.3.0
```

### Step 4: Install Additional Dependencies
Install the remaining Python packages using:
```bash
pip install -r requirements.txt
```
