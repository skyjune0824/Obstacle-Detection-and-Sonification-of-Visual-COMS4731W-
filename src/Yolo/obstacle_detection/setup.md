YOLOv8 RGB-D Fusion:

# Clone the repository
git clone https://github.com/safouaneelg/FusionVision.git
cd FusionVision/

# Create virtual environment (required - RealSense not compatible with conda)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install libxcb-cursor0
pip install -r requirements.txt


# Run complete pipeline with depth-aware detection
python FusionVisionV0.3.py \
    --yolo_weight yolov8x.pt \
    --fastsam_weight FastSAM-x.pt \
    --confidence_threshold 0.7 \
    --conf 0.4 \
    --iou 0.9 \
    --show_3dbbox

# For custom monocular depth input (instead of RealSense):
# Modify the depth input section in FusionVisionV0.3.py to load
# depth maps 


RFNet:

# Clone the repository
git clone https://github.com/AHupuJR/RFNet.git
cd RFNet/

# Install dependencies
pip install torch==1.1.0 torchvision==0.3.0
pip install opencv-python==3.3.1



