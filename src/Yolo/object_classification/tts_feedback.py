# tts_feedback.py

import threading
from queue import Queue, Empty
from typing import Dict, List, Optional

import pyttsx3


class TTSFeedback:
    """
    Non-blocking text-to-speech for obstacle announcements.

    Can be driven either from prioritized obstacle dicts or from audio_cues.
    """

    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        voice_name_substring: Optional[str] = None,
    ):
        # Initialize engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", max(0.0, min(1.0, volume)))

        # Optionally pick a voice by name substring (e.g. "female", "Zira", "en")
        if voice_name_substring is not None:
            for v in self.engine.getProperty("voices"):
                if voice_name_substring.lower() in v.name.lower():
                    self.engine.setProperty("voice", v.id)
                    break

        # Queue + worker thread
        self.queue: "Queue[str]" = Queue()
        self._running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    # ------------------- worker loop -------------------

    def _worker_loop(self):
        """Background loop that speaks queued messages sequentially."""
        while self._running:
            try:
                text = self.queue.get(timeout=0.1)
            except Empty:
                continue

            if text is None:
                # Shutdown sentinel
                break

            self.engine.say(text)
            self.engine.runAndWait()
            self.queue.task_done()

    # ------------------- public API -------------------

    def enqueue_message(self, text: str):
        """Queue a message to be spoken (non-blocking)."""
        if not self._running or not text:
            return
        self.queue.put(text)

    # ------ helpers to map obstacles/audio_cues -> text ------

    @staticmethod
    def _zone_phrase(zone: str) -> str:
        if zone == "left":
            return "on your left"
        if zone == "right":
            return "on your right"
        if zone == "center":
            return "ahead"
        return zone

    @staticmethod
    def _distance_phrase(distance_m: float) -> str:
        if distance_m == float("inf"):
            return ""
        if distance_m < 0.7:
            return "very close, less than one meter"
        if distance_m < 1.5:
            return "about one meter away"
        if distance_m < 3.0:
            return f"about {distance_m:.1f} meters away"
        return "far ahead"

    def speak_from_prioritized_obstacles(
        self,
        prioritized_obstacles: List[Dict],
        max_zones: int = 2,
    ):
        """
        Build speech from prioritized obstacles.

        Each obstacle is expected to have keys:
            'zone', 'navigation_class', 'depth', optional 'is_urgent'
        which matches ContextAwarePrioritizer/ObjectAwareAudioMapper usage.
        """
        if not prioritized_obstacles:
            return

        # One critical obstacle per zone, closest first
        by_zone: Dict[str, Dict] = {}
        for obs in sorted(prioritized_obstacles, key=lambda o: o.get("depth", 1e9)):
            z = obs.get("zone", "center")
            if z not in by_zone:
                by_zone[z] = obs

        selected = sorted(by_zone.values(), key=lambda o: o.get("depth", 1e9))[:max_zones]

        for obs in selected:
            zone = obs.get("zone", "center")
            cls = obs.get("navigation_class", "obstacle").replace("_", " ")
            dist = float(obs.get("depth", float("inf")))
            urgent = bool(obs.get("is_urgent", False))

            zone_phrase = self._zone_phrase(zone)
            dist_phrase = self._distance_phrase(dist)
            prefix = "Warning" if urgent else "Obstacle"

            if dist_phrase:
                sentence = f"{prefix}, {cls} {zone_phrase}, {dist_phrase}."
            else:
                sentence = f"{prefix}, {cls} {zone_phrase}."

            self.enqueue_message(sentence)

    def speak_from_audio_cues(
        self,
        audio_cues: Dict[str, Dict],
        include_type: bool = True,
    ):
        """
        Alternate interface driven by ObjectAwareAudioMapper audio_cues.

        audio_cues[zone] should contain:
            'active', 'object_type', optional 'urgency'
        """
        for zone, cue in audio_cues.items():
            if not cue.get("active", False):
                continue

            zone_phrase = self._zone_phrase(zone)
            cls = cue.get("object_type", "obstacle").replace("_", " ")
            urgent = cue.get("urgency", "medium") == "high"
            prefix = "Warning" if urgent else "Obstacle"

            if include_type:
                text = f"{prefix}, {cls} {zone_phrase}."
            else:
                text = f"{prefix} {zone_phrase}."

            self.enqueue_message(text)

    # ------------------- shutdown -------------------

    def close(self):
        """Cleanly shut down the worker and TTS engine."""
        if not self._running:
            return
        self._running = False
        self.queue.put(None)
        self.thread.join(timeout=2.0)
        try:
            self.engine.stop()
        except Exception:
            pass
