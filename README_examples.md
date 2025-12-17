# Obstacle Detection and Sonification For the Visually Impaired

The challenge of independent navigation is a critical area for technological innovation. While
traditional aids such as the white cane and guide dogs are invaluable, they often provide limited
information about the environment, specifically regarding the distance and the nature of obstacles
beyond immediate contact. Our project addresses this gap by developing a real-time obstacle
avoidance system that leverages modern computer vision technology.

Through our project, we explored three different designs: 

1. YOLO
2. Monocular Depth Estimation (MDE)
3. Monocular Visual Simultaneous Localization
and Mapping (SLAM)

The following text illustrates how one can utilize our final approach we decided on: YOLO. For exploring our previous attempts (MDE/SLAM), you can explore their respective test examples within their subdirectories inside the ``Test`` Folder.

## VODS External Examples: Datasets and Testing

The test folder included shows how someone **outside the project** can:
- Download COCO for YOLO training/finetuning.
- Set up a simple RGBD demo with YOLO + depth + TTS.
- Run the obstacle detection + sonification + text-to-speech pipeline on webcam.

## 1. Prerequisites

- Python 3.8+
- Git
- A working microphone/speaker setup (for audio and TTS)
- For GPU acceleration: a CUDA-capable GPU and proper drivers

From the project root, create and activate the virtual environment (if you have not already):

chmod +x setup.sh
./setup.sh

Then, in a new shell:
source cvproject/bin/activate # Linux/macOS

or
.\cvproject\Scripts\activate # Windows


This installs:
- `ultralytics` (YOLOv8)
- `pyttsx3` for text-to-speech
- `pyaudio` and other CV/audio dependencies (see root README).
  
## 2. Getting COCO for YOLO Training (Optional)

If you want to **train or fine-tune** the YOLO model yourself, you need the MS-COCO dataset.

Folder structure (from project root):

data/
coco/
images/
train2017/
val2017/
annotations/
instances_train2017.json
instances_val2017.json


You can download COCO manually from the official website:

- Images & annotations: https://cocodataset.org/#download

Or use the helper script from the project root:

cd examples
chmod +x download_coco.sh
./download_coco.sh


This will:
- Create `data/coco` under the project root.
- Download the 2017 train/val images and annotations.
- Extract them into the required structure.

> Note: COCO is large (~25–30 GB). Make sure you have enough disk space and a stable connection.

## 3. Training YOLOv8 on COCO (Optional)

Once COCO is downloaded and the environment is active:


From project root
python train_yolov8.py


The training script uses `coco.yaml` and saves weights under:

runs/detect/yolov8n_coco_exp_01/weights/best.pt


You can change model size, epochs, etc. inside `train_yolov8.py`.

If you don’t want to train, you can instead:
- Use the provided pretrained checkpoint (if included), or
- Use an official YOLOv8 COCO checkpoint, e.g. `yolov8n.pt`.

## 4. Running the YOLO + Depth + TTS Demo

This example uses:
- `RGBDObjectClassifier` (YOLO + depth validation)
- `ObjectTracker` and `ContextAwarePrioritizer` (from the main project)
- `ObjectAwareAudioMapper` for tonal cues
- `TTSFeedback` for spoken cues
- A simulated depth map (replace with real Depth Anything V2 / NYU / device depth if available)

### 4.1 Configure the demo

Edit `examples/sample_config.yaml` to set:

- Path to your YOLO model weights
- Whether to use webcam or a video file
- Whether to simulate depth or load it from file/device

### 4.2 Run the demo

From the project root (env activated):

python examples/run_rgbd_yolo_tts_demo.py


- Press `q` in the video window to quit.
- You should hear both:
  - Spatial tones indicating obstacle zones and urgency.
  - Voice prompts such as “Warning, pedestrian ahead, about one meter away.”

If you have a real depth sensor or precomputed depth maps, plug them into `run_rgbd_yolo_tts_demo.py` where indicated.

## 5. Expected Behavior

- The window “Object Classification” shows YOLO detections with depth labels.
- Every frame, the pipeline:
  - Validates detections using depth.
  - Tracks objects over time.
  - Prioritizes obstacles per zone.
  - Plays spatial audio tones (frequency/volume/panning).
  - Speaks short TTS messages about the most important obstacles.
