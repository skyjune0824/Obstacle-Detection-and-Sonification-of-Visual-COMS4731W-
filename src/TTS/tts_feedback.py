import threading
from queue import Queue, Empty
from typing import Dict, List, Optional

import pyttsx3


class TTSFeedback:
    """
    Simple, non-blocking text-to-speech engine for obstacle announcements.

    Usage:
        tts = TTSFeedback()
        tts.enqueue_message("Obstacle ahead at 2 meters")
        ...
        tts.close()
    """

    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        voice_name_substring: Optional[str] = None,
    ):
        # Create engine in this thread, but speech will run on a worker thread
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", max(0.0, min(1.0, volume)))

        # Optionally pick a specific voice (e.g. female / en-US) by substring
        if voice_name_substring is not None:
            for v in self.engine.getProperty("voices"):
                if voice_name_substring.lower() in v.name.lower():
                    self.engine.setProperty("voice", v.id)
                    break

        # Message queue for non-blocking use
        self.queue: "Queue[str]" = Queue()
        self._running = True

        # Background worker that drains queue and speaks
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        """Background loop that speaks queued messages sequentially."""
        while self._running:
            try:
                # Small timeout so we can exit promptly when closing
                text = self.queue.get(timeout=0.1)
            except Empty:
                continue

            if text is None:
                # Sentinel for shutdown
                break

            # Speak synchronously in this worker thread
            self.engine.say(text)
            self.engine.runAndWait()
            self.queue.task_done()

    def enqueue_message(self, text: str):
        """
        Queue a message to be spoken.

        This call is non-blocking; the message is spoken by the worker thread.
        """
        if not self._running:
            return
        if not text:
            return
        self.queue.put(text)

    # Convenience for your obstacle representation -------------------------

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
        Generate and enqueue short messages from your prioritized obstacle list.

        Expected obstacle keys (as in ObjectAwareAudioMapper):
            - 'zone' (left/center/right)
            - 'navigation_class' (pedestrian, vehicle, furniture, ...)
            - 'depth' (meters)
            - optional 'is_urgent' (bool)
        """
        if not prioritized_obstacles:
            return

        # Keep one critical obstacle per zone, sorted by proximity
        by_zone: Dict[str, Dict] = {}
        for obs in sorted(prioritized_obstacles, key=lambda o: o.get("depth", 1e9)):
            z = obs.get("zone", "center")
            if z not in by_zone:
                by_zone[z] = obs

        # Choose at most max_zones zones to avoid overloading the user
        # Prefer closer obstacles overall
        selected = sorted(by_zone.values(), key=lambda o: o.get("depth", 1e9))[:max_zones]

        for obs in selected:
            zone = obs.get("zone", "center")
            cls = obs.get("navigation_class", "obstacle").replace("_", " ")
            dist = float(obs.get("depth", float("inf")))
            urgent = bool(obs.get("is_urgent", False))

            zone_phrase = self._zone_phrase(zone)
            dist_phrase = self._distance_phrase(dist)

            if urgent:
                prefix = "Warning"
            else:
                prefix = "Obstacle"

            # Example: "Warning, pedestrian ahead, about 1.2 meters away."
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
        Alternate interface if you want to drive TTS from ObjectAwareAudioMapper
        output instead of raw obstacles.

        Expected audio_cues[zone]:
            - 'active' (bool)
            - 'object_type' (string)
            - optional 'urgency' (high/medium)
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

    def close(self):
        """Shut down worker thread and TTS engine cleanly."""
        if not self._running:
            return
        self._running = False
        # Signal worker to exit
        self.queue.put(None)
        self.thread.join(timeout=2.0)
        # Flush any remaining speech
        try:
            self.engine.stop()
        except Exception:
            pass
