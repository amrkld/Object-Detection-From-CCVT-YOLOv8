# Object Detection and Tracking on CCTV Footage

This project demonstrates how to perform object detection and tracking on CCTV footage using YOLOv8 and Supervision libraries. The code is designed to be run in Google Colab and includes functionality for processing both individual video frames and entire videos. 

## Table of Contents

1. [Setup](#setup)
2. [Usage](#usage)
3. [Download Output](#download-output)
4. [Dependencies](#dependencies)

## Setup

1. **Open the Colab Notebook**:
   - You can run this code in Google Colab. Open the notebook from [this link](https://colab.research.google.com/drive/1tBQHVV6W4caKbdMwBAlRFcMzyASZ_uxO?usp=sharing).

2. **Install Required Libraries**:
   - The code installs necessary Python libraries for object detection and video processing.

   ```python
   !pip install ultralytics supervision

## Usage

### Run the Model on a Specific Frame

1. **Download the Video**:
   - The video is downloaded from a given URL and saved locally.

   ```python
   !gdown '10zzs49pm90lG5EqJpuf9X-N-sQcSSl43' -O CCTV_Input.mp4
   ```

2. **Load and Prepare the Model**:
   - The YOLOv8 model is loaded and configured.

   ```python
   from ultralytics import YOLO
   import supervision as sv
   import numpy as np
   model = YOLO('yolov8n.pt')
   ```

3. **Run Detection on a Specific Frame**:
   - Select a frame from the video, run detection, and annotate the frame.

   ```python
   generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
   corner_annotator = sv.BoxCornerAnnotator(thickness=2, color=color_to_use)
   frame_index = 600
   iterator = iter(generator)
   for i in range(frame_index + 1):
       frame = next(iterator)
   results = model(frame, verbose=False)[0]
   detections = sv.Detections.from_ultralytics(results)
   detections = detections[np.isin(detections.class_id, selected_classes)]
   annotated_frame = corner_annotator.annotate(scene=frame, detections=detections)
   sv.plot_image(annotated_frame, (16, 16))
   ```

### Run the Model on the Full Video

1. **Process the Entire Video**:
   - Track objects throughout the video and save the output.

   ```python
   TARGET_VIDEO_PATH = f"{HOME}/Halo_output_video.mp4"
   byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
   video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
   generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
   corner_annotator = sv.BoxCornerAnnotator(thickness=2)
   def callback(frame: np.ndarray, index: int) -> np.ndarray:
       results = model(frame, verbose=False)[0]
       detections = sv.Detections.from_ultralytics(results)
       detections = detections[np.isin(detections.class_id, selected_classes)]
       detections = byte_tracker.update_with_detections(detections)
       labels = [f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)]
       annotated_frame = corner_annotator.annotate(scene=frame.copy(), detections=detections)
       return annotated_frame
   sv.process_video(source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback)
   ```

## Download Output

1. **Download the Processed Video**:
   - Download the output video file from Colab.

   ```python
   from google.colab import files
   file_path = '/content/Halo_output_video.mp4'
   files.download(file_path)
   ```

## Dependencies

- **Python Libraries**:
  - `ultralytics`: For YOLOv8 model.
  - `supervision`: For video processing and annotation.
  - `opencv-python`: For image and video processing.
  - `numpy`: For numerical operations.

- **Google Colab**:
  - Recommended environment for running the code.

---

Feel free to modify the code and experiment with different configurations or models to suit your needs.
```

### Summary of the README File:

1. **Setup**: Instructions to run the code in Google Colab and install dependencies.
2. **Usage**: Detailed steps on how to use the code for running detections on specific frames and the entire video.
3. **Download Output**: Instructions for downloading the processed video file.
4. **Dependencies**: List of required Python libraries and recommended environment.

This README should help users understand how to set up and run the code, and how to work with the generated outputs.
