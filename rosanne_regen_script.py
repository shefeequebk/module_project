import argparse
import time
import pickle
import os

from threading import Thread

import cv2
import numpy as np
import face_recognition
import tflite_runtime.interpreter as tflite

# Use headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# Global configuration
# -----------------------

DEFAULT_RUN_TIME = 20.0           # seconds
FACE_DOWNSCALE = 4                # 4x smaller -> faster but less precise
DEFAULT_MAX_FPS = 5.0             # processing rate limit

FACE_INTERVAL_FRAMES = 2          # run face pipeline every N frames
OBJECT_INTERVAL_FRAMES = 1        # run object detector every N frames

LOG_PERIOD_SECONDS = 0.0          # 0 = log every processed frame

# Try to import Picamera2 if available
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


# -----------------------
# Camera handling
# -----------------------

class AsyncCamera:
    """Simple threaded camera wrapper for USB webcam or Picamera2."""

    def __init__(self, frame_size=(640, 480), fps=30, use_pi_camera=False):
        self._use_pi = bool(use_pi_camera)
        self._size = frame_size
        self._fps = fps
        self._stopped = False
        self._frame = None
        self._grabbed = False

        if self._use_pi:
            if Picamera2 is None:
                raise RuntimeError("Picamera2 is not available but use_pi_camera=True")

            self._cam = Picamera2()
            cfg = self._cam.create_preview_configuration(
                main={"format": "XRGB8888", "size": self._size}
            )
            self._cam.configure(cfg)
            self._cam.start()
            self._frame = self._cam.capture_array()
            self._grabbed = True
        else:
            self._cam = cv2.VideoCapture(0)
            self._cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
            self._grabbed, self._frame = self._cam.read()

    def start(self):
        Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while True:
            if self._stopped:
                if self._use_pi:
                    self._cam.stop()
                else:
                    self._cam.release()
                return

            if self._use_pi:
                self._frame = self._cam.capture_array()
                self._grabbed = True
            else:
                self._grabbed, self._frame = self._cam.read()

    def read(self):
        return self._frame

    def stop(self):
        self._stopped = True


# -----------------------
# Face encoding utilities
# -----------------------

def load_known_faces(enc_path):
    """Load known face encodings from pickle file."""
    print("[INFO] Loading known faces from:", enc_path)
    with open(enc_path, "rb") as fh:
        data = pickle.loads(fh.read())
    return {
        "encodings": data["encodings"],
        "names": data["names"],
    }


def distance_to_score(dist, max_dist=1.0):
    """Convert face distance to confidence-like score in [0, 1]."""
    d = min(dist, max_dist)
    return 1.0 - (d / max_dist)


def run_face_pipeline(frame_bgr, face_db, downscale=4):
    """
    Detect and recognize faces in a BGR frame.
    Returns list of dicts with {rect, name, score}.
    rect = (x1, y1, x2, y2) in original frame coordinates.
    """
    # Shrink frame for speed
    small = cv2.resize(
        frame_bgr,
        (0, 0),
        fx=1.0 / downscale,
        fy=1.0 / downscale,
    )
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations_small = face_recognition.face_locations(rgb_small)
    encodings = face_recognition.face_encodings(
        rgb_small, locations_small, model="small"  # use faster model
    )

    results = []
    for (top, right, bottom, left), enc in zip(locations_small, encodings):
        distances = face_recognition.face_distance(face_db["encodings"], enc)
        idx_best = int(np.argmin(distances))
        name = "Unknown"
        score = distance_to_score(distances[idx_best])

        if face_recognition.compare_faces(
            [face_db["encodings"][idx_best]], enc
        )[0]:
            name = face_db["names"][idx_best]

        # Map box back to original resolution
        top *= downscale
        right *= downscale
        bottom *= downscale
        left *= downscale

        results.append(
            {
                "rect": (left, top, right, bottom),
                "name": name,
                "score": float(score),
            }
        )

    return results


# -----------------------
# TFLite object detector
# -----------------------

class TFLiteDetector:
    def __init__(self, model_path, label_path):
        print("[INFO] Initializing TFLite detector...")
        t0 = time.time()

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.in_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()

        self.in_h = int(self.in_details[0]["shape"][1])
        self.in_w = int(self.in_details[0]["shape"][2])
        self.is_floating = self.in_details[0]["dtype"] == np.float32
        self.mean = 127.5
        self.std = 127.5

        with open(label_path, "r") as fh:
            self.labels = [ln.strip() for ln in fh.readlines()]

        print(f"[INFO] Detector ready in {time.time() - t0:.2f}s")

    def infer(self, frame_bgr, min_confidence, frame_w, frame_h, skip_person=True):
        """
        Run detector on BGR frame.
        Returns list of dicts: {rect, label, score}
        rect = (x1, y1, x2, y2) in original frame coordinates.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h))
        input_data = np.expand_dims(resized, axis=0)

        if self.is_floating:
            input_data = (np.float32(input_data) - self.mean) / self.std

        self.interpreter.set_tensor(self.in_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.out_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.out_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.out_details[2]["index"])[0]

        outputs = []
        for i, score in enumerate(scores):
            if score < min_confidence or score > 1.0:
                continue

            label = self.labels[int(classes[i])]
            if skip_person and label.lower() == "person":
                continue

            y_min = int(max(1, boxes[i][0] * frame_h))
            x_min = int(max(1, boxes[i][1] * frame_w))
            y_max = int(min(frame_h, boxes[i][2] * frame_h))
            x_max = int(min(frame_w, boxes[i][3] * frame_w))

            outputs.append(
                {
                    "rect": (x_min, y_min, x_max, y_max),
                    "label": label,
                    "score": float(score),
                }
            )

        return outputs


# -----------------------
# Visual helpers
# -----------------------

def draw_faces(canvas, faces):
    for f in faces:
        x1, y1, x2, y2 = f["rect"]
        name = f["name"]
        score = f["score"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)

        label_txt = f"{name}: {int(score * 100)}%"
        (tw, th), baseline = cv2.getTextSize(
            label_txt, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1
        )
        y_text = max(y1 - 10, th + 5)
        cv2.rectangle(
            canvas,
            (x1 - 3, y_text - th - 5),
            (x1 - 3 + tw + 6, y_text + baseline),
            (0, 255, 255),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label_txt,
            (x1, y_text),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 0, 0),
            1,
        )


def draw_objects(canvas, objects):
    count = 0
    for obj in objects:
        x1, y1, x2, y2 = obj["rect"]
        label = obj["label"]
        score = obj["score"]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 255, 0), 2)

        text = f"{label}: {int(score * 100)}%"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        y_text = max(y1, th + 10)
        cv2.rectangle(
            canvas,
            (x1, y_text - th - 10),
            (x1 + tw, y_text + baseline - 10),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            text,
            (x1, y_text - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        count += 1

    return count


# -----------------------
# Logging & analysis
# -----------------------

def record_events(event_list, t_stamp, faces, objects):
    """Append detection events to list for later analysis."""
    for f in faces:
        event_list.append(
            {
                "time": float(t_stamp),
                "label": f["name"],
                "kind": "face",
                "score": f["score"],
            }
        )
    for o in objects:
        event_list.append(
            {
                "time": float(t_stamp),
                "label": o["label"],
                "kind": "object",
                "score": o["score"],
            }
        )


def save_detection_plot(events, total_time, filename="detections_over_time.png"):
    if not events:
        print("[INFO] No events to plot.")
        return

    groups = sorted({(e["label"], e["kind"]) for e in events})
    plt.figure(figsize=(10, 6))

    for label, kind in groups:
        xs = [e["time"] for e in events if e["label"] == label and e["kind"] == kind]
        ys = [e["score"] for e in events if e["label"] == label and e["kind"] == kind]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", linestyle="-", label=f"{label} ({kind})")

    plt.xlabel("Time (s)")
    plt.ylabel("Confidence / Score")
    plt.title("Detections over time")
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, total_time)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot to {filename}")


def print_presence_spans(events, gap_threshold=1.0):
    if not events:
        print("[INFO] No detections to summarize.")
        return

    print("\n[INFO] Detection intervals:")
    groups = sorted({(e["label"], e["kind"]) for e in events})

    for label, kind in groups:
        times = sorted(
            e["time"] for e in events if e["label"] == label and e["kind"] == kind
        )
        if not times:
            continue

        spans = []
        start = times[0]
        prev = times[0]

        for t in times[1:]:
            if t - prev <= gap_threshold:
                prev = t
            else:
                spans.append((start, prev))
                start = t
                prev = t
        spans.append((start, prev))

        print(f" - {label} ({kind}):")
        for s, e in spans:
            print(f"     from {s:.1f}s to {e:.1f}s (duration {e - s:.1f}s)")


# -----------------------
# Main loop wrapper
# -----------------------

class DetectionRunner:
    def __init__(
        self,
        cam,
        face_db,
        detector,
        runtime,
        max_fps,
        conf_thresh,
        show_window=True,
        face_downscale=4,
        face_interval=2,
        obj_interval=1,
        log_period=0.0,
    ):
        self.cam = cam
        self.face_db = face_db
        self.detector = detector
        self.runtime = runtime
        self.max_fps = max_fps
        self.conf_thresh = conf_thresh
        self.show_window = show_window
        self.face_downscale = face_downscale
        self.face_interval = max(1, int(face_interval))
        self.obj_interval = max(1, int(obj_interval))
        self.log_period = max(0.0, float(log_period))

        self.events = []
        self._last_log_time = -1e9
        self._frame_idx = 0

    def run(self):
        start = time.time()
        last_frame_t = 0.0
        latest_faces = []
        latest_objs = []

        print("[INFO] Starting main loop...")
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= self.runtime:
                print("[INFO] Time limit reached, stopping loop.")
                break

            # FPS limiting
            if self.max_fps and self.max_fps > 0:
                interval = 1.0 / self.max_fps
                if (now - last_frame_t) < interval:
                    continue
                last_frame_t = now

            frame_start = time.time()
            frame = self.cam.read()
            if frame is None:
                continue

            # Picamera2 frames may arrive as RGB(A)
            if Picamera2 is not None and isinstance(self.cam._cam, Picamera2):
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w = frame.shape[:2]
            self._frame_idx += 1

            # Decide whether to run each pipeline this frame
            do_face = (self._frame_idx % self.face_interval == 0)
            do_obj = (self._frame_idx % self.obj_interval == 0)

            if do_face:
                latest_faces = run_face_pipeline(
                    frame, self.face_db, downscale=self.face_downscale
                )

            if do_obj:
                latest_objs = self.detector.infer(
                    frame,
                    min_confidence=self.conf_thresh,
                    frame_w=w,
                    frame_h=h,
                    skip_person=True,
                )

            # Log events
            should_log = False
            if self.log_period <= 0.0:
                should_log = True
            else:
                if (elapsed - self._last_log_time) >= self.log_period:
                    should_log = True
                    self._last_log_time = elapsed

            if should_log:
                record_events(self.events, elapsed, latest_faces, latest_objs)

            # Draw overlays
            canvas = frame.copy()
            draw_faces(canvas, latest_faces)
            count_objects = draw_objects(canvas, latest_objs)

            # FPS calculation (based on processing time)
            frame_time = time.time() - frame_start
            fps = 1.0 / frame_time if frame_time > 0 else 0.0

            cv2.putText(
                canvas,
                f"FPS: {fps:.2f}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 55),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Total Detection Count : {count_objects}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 55),
                2,
                cv2.LINE_AA,
            )

            if self.show_window:
                cv2.imshow("Face + Object Monitor", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 'q' pressed, exiting loop.")
                    break

        # Cleanup
        if self.show_window:
            cv2.destroyAllWindows()
        self.cam.stop()

        total_time = time.time() - start
        print_presence_spans(self.events, gap_threshold=1.0)
        save_detection_plot(self.events, total_time)


# -----------------------
# Argument parsing & entry
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Face + Object detector experiment")
    parser.add_argument(
        "--model",
        default="tfrpie/models/model.tflite",
        help="Path to TFLite model file",
    )
    parser.add_argument(
        "--labels",
        default="tfrpie/models/labels.txt",
        help="Path to labels text file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum object detection confidence",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Camera resolution as WxH (e.g. 640x480)",
    )
    parser.add_argument(
        "--encodings",
        default="face_rec/encodings.pickle",
        help="Pickle file containing known face encodings",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_RUN_TIME,
        help="Experiment duration in seconds",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=DEFAULT_MAX_FPS,
        help="Maximum processing FPS (0 or None = unlimited)",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use USB webcam instead of Raspberry Pi camera",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window (headless mode)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse resolution
    try:
        w_str, h_str = args.resolution.lower().split("x")
        cam_w, cam_h = int(w_str), int(h_str)
    except Exception:
        raise ValueError("Resolution must be in WxH format, e.g. 640x480")

    use_pi = not args.webcam   # default: Pi cam; --webcam → USB camera
    show = not args.no_display

    # Load resources
    face_db = load_known_faces(args.encodings)
    detector = TFLiteDetector(args.model, args.labels)

    # Start camera
    print("[INFO] Starting camera...")
    cam = AsyncCamera(frame_size=(cam_w, cam_h), fps=30, use_pi_camera=use_pi).start()
    time.sleep(1.0)

    runner = DetectionRunner(
        cam=cam,
        face_db=face_db,
        detector=detector,
        runtime=args.duration,
        max_fps=args.max_fps,
        conf_thresh=args.threshold,
        show_window=show,
        face_downscale=FACE_DOWNSCALE,
        face_interval=FACE_INTERVAL_FRAMES,
        obj_interval=OBJECT_INTERVAL_FRAMES,
        log_period=LOG_PERIOD_SECONDS,
    )
    runner.run()


if __name__ == "__main__":
    main()
