
import cv2
import numpy as np
import yaml
from pathlib import Path

# Import from the main project
from rgbd_classifier import RGBDObjectClassifier
from object_tracker import ObjectTracker
from prioritizer import ContextAwarePrioritizer
from audio_mapper import ObjectAwareAudioMapper
from tts_feedback import TTSFeedback


class RGBDYOLOTtsDemo:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        yolo_weights = self.config["yolo_weights"]
        print(f"[INFO] Loading YOLO model from: {yolo_weights}")

        self.classifier = RGBDObjectClassifier(yolo_weights)
        self.tracker = ObjectTracker()
        self.prioritizer = ContextAwarePrioritizer()
        self.audio_mapper = ObjectAwareAudioMapper()

        tts_cfg = self.config.get("tts", {})
        self.tts = TTSFeedback(
            rate=tts_cfg.get("rate", 180),
            volume=tts_cfg.get("volume", 1.0),
            voice_name_substring=tts_cfg.get("voice_hint"),
        ) if tts_cfg.get("enabled", True) else None

        self.frame_count = 0
        self.depth_cfg = self.config.get("depth", {})

    @staticmethod
    def _load_config(config_path: str):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_depth_map(self, frame):
        mode = self.depth_cfg.get("mode", "simulated")
        h, w = frame.shape[:2]

        if mode == "simulated":
            # Simple synthetic depth: closer at bottom center, farther at top edges
            max_depth = float(self.depth_cfg.get("max_depth_m", 5.0))
            yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
            center = np.exp(-((xx - 0.5) ** 2 + (yy - 0.8) ** 2) / 0.1)
            depth = max_depth * (1.0 - center)
            return depth.astype(np.float32)

        # Placeholder: integrate Depth Anything V2, RGBD sensor, or file-based depth here.
        # For now, fall back to simulated.
        print("[WARN] Depth mode not implemented, falling back to simulated.")
        max_depth = float(self.depth_cfg.get("max_depth_m", 5.0))
        return np.random.rand(h, w).astype(np.float32) * max_depth

    def run(self):
        input_cfg = self.config.get("input", {})
        source = input_cfg.get("source", "webcam")

        if source == "video":
            video_path = input_cfg.get("video_path")
            cap = cv2.VideoCapture(video_path)
            window_title = f"Object Classification (video: {video_path})"
        else:
            cap = cv2.VideoCapture(0)
            window_title = "Object Classification (webcam)"

        print("[INFO] Starting RGBD YOLO + TTS demo.")
        print("[INFO] Press 'q' to quit.")

        debug_cfg = self.config.get("debug", {})
        print_every = int(debug_cfg.get("print_every_n_frames", 30))

        try:
            while True:
                ret, rgb_frame = cap.read()
                if not ret:
                    print("[INFO] End of stream or camera error.")
                    break

                self.frame_count += 1

                # Get depth map
                depth_map = self._get_depth_map(rgb_frame)

                # Step 1: classify with depth validation
                detections = self.classifier.classify_with_depth_validation(
                    rgb_frame, depth_map
                )

                # Step 2: track across frames
                _ = self.tracker.update(detections, self.frame_count)
                tracked_objects = self.tracker.get_tracked_objects()

                # Step 3: prioritize obstacles into zones
                prioritized = self.prioritizer.prioritize_obstacles(
                    tracked_objects, rgb_frame.shape[1]
                )

                # Step 4: generate tonal audio cues
                audio_cues = self.audio_mapper.generate_audio_cues(prioritized)

                # Step 5: TTS spoken output
                if self.tts is not None:
                    # Directly from prioritized obstacles (preferred)
                    self.tts.speak_from_prioritized_obstacles(prioritized)
                    # Or: self.tts.speak_from_audio_cues(audio_cues)

                # Step 6: visualize detections
                vis_frame = self.classifier.visualize_detections(rgb_frame, detections)
                cv2.imshow(window_title, vis_frame)

                # Optional prints
                if self.frame_count % print_every == 0:
                    self.prioritizer.print_prioritized_obstacles(prioritized)
                    self.audio_mapper.print_audio_cues(audio_cues)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.tts is not None:
                self.tts.close()
            print("[INFO] Demo finished.")


if __name__ == "__main__":
    # Resolve config path relative to project root or examples/
    here = Path(__file__).resolve()
    config_path = here.parent / "sample_config.yaml"
    demo = RGBDYOLOTtsDemo(str(config_path))
    demo.run()
