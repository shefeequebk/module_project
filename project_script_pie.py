import argparse
import time
import pickle

import cv2
import numpy as np
import face_recognition
import tflite_runtime.interpreter as tflite
from threading import Thread

# Set matplotlib to use non-GUI backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (no GUI required)
import matplotlib.pyplot as plt

# -----------------------
# Configuration
# -----------------------

DURATION_SECONDS = 20    # total time to run (can be changed with --duration)

# Processing resolution (NOT capture resolution)
# Full frame (Pi): 2592x1944, Processing: 1280x960 (same aspect ratio 4:3)
PROC_W = 1280
PROC_H = 720

CV_SCALER = 2            # extra downscale inside face recognizer
MAX_FPS = 15             # maximum frames per second to process (None = process all frames)

# How often to run each detector (in processed frames)
FACE_EVERY_N_FRAMES = 2  # run face recognition every 4 processed frames
OBJ_EVERY_N_FRAMES = 4   # run object detection every 4 processed frames

# Logging frequency (seconds). Set to 0 to log every processed frame.
LOG_EVERY_SECONDS = 0.0

# Parse PIE_CAM argument at module level (so importing this module won't fail)
parser_module = argparse.ArgumentParser(add_help=False)
parser_module.add_argument(
    '--webcam',
    action='store_false',
    dest='pie_cam',
    default=True,
    help='Use webcam instead of Raspberry Pi camera (default: use Pi camera)'
)
args_module, _ = parser_module.parse_known_args()
PIE_CAM = args_module.pie_cam

if PIE_CAM:
    from picamera2 import Picamera2


# -----------------------
# Video Stream
# -----------------------

# -----------------------
# Video Stream (Patched)
# -----------------------

class VideoStream:
    """Camera object that controls video streaming from webcam or Picamera2 in a separate thread."""

    def __init__(self, resolution=(1640, 1232), framerate=30, use_picamera=False):
        self.use_picamera = use_picamera
        self.resolution = resolution
        self.framerate = framerate

        if self.use_picamera:
            # --- CORRECT PICAMERA2 CONFIG YOU REQUESTED ---
            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={"format": "BGR888", "size": resolution},
                controls={"AwbMode": 1, "Saturation": 1.0}
            )

            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1)  # allow AWB to settle

            # First frame
            self.frame = self.picam2.capture_array()
            self.grabbed = True

        else:
            # USB webcam
            self.stream = cv2.VideoCapture(0)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.stream.set(3, resolution[0])
            self.stream.set(4, resolution[1])
            (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                if self.use_picamera:
                    self.picam2.stop()
                else:
                    self.stream.release()
                return

            if self.use_picamera:
                # Always return BGR888 full-resolution frame
                self.frame = self.picam2.capture_array()
                self.grabbed = True
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True



# -----------------------
# Face Recognition
# -----------------------

def load_face_encodings(path="encodings.pickle"):
    print("[INFO] loading face encodings...")
    with open(path, "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
    return known_face_encodings, known_face_names


def face_distance_to_confidence(face_distance, max_distance=1.0):
    """Convert face distance (0=perfect match) to a confidence score in [0, 1]."""
    confidence = 1.0 - min(face_distance, max_distance) / max_distance
    return confidence


def recognize_faces(frame_proc, known_face_encodings, known_face_names, cv_scaler=2):
    """
    Run face detection + recognition on a BGR *processed* frame (e.g., 1280x960).

    Returns:
        detections in PROC frame coordinates, box format: (top, left, bottom, right)
    """
    # Downscale for speed
    small_frame = cv2.resize(
        frame_proc, (0, 0), fx=1.0 / cv_scaler, fy=1.0 / cv_scaler
    )
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations_small = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(
        rgb_small, face_locations_small, model="small"
    )

    detections = []
    for (top, right, bottom, left), face_encoding in zip(
        face_locations_small, face_encodings
    ):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        confidence = face_distance_to_confidence(face_distances[best_match_index])

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale back to processed frame size (PROC_W x PROC_H)
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Box in processed frame coordinates
        detections.append(
            {
                "box": (top, left, bottom, right),  # (top, left, bottom, right)
                "label": name,
                "confidence": float(confidence),
            }
        )

    return detections


# -----------------------
# TFLite Object Detection
# -----------------------

def load_tflite_model(model_path, labels_path):
    """Load TFLite model and precompute I/O details once (faster per-frame)."""
    print("[INFO] loading TFLite object detection model...", end="")
    start_time = time.time()
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]["shape"][1]
    input_width = input_details[0]["shape"][2]
    floating_model = input_details[0]["dtype"] == np.float32

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    elapsed = time.time() - start_time
    print(f" done in {elapsed:.2f} seconds.")
    return (
        interpreter,
        labels,
        input_details,
        output_details,
        input_height,
        input_width,
        floating_model,
    )


def detect_objects(
    frame_proc,
    interpreter,
    labels,
    min_conf_thresh,
    input_details,
    output_details,
    input_height,
    input_width,
    floating_model,
):
    """
    Run TFLite object detection on a BGR *processed* frame.

    Returns:
        detections in PROC frame coordinates, box format: (top, left, bottom, right)
    """
    input_mean = 127.5
    input_std = 127.5

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]     # Bounding box
    classes = interpreter.get_tensor(output_details[1]["index"])[0]   # Class index
    scores = interpreter.get_tensor(output_details[2]["index"])[0]    # Confidence

    proc_h, proc_w = frame_proc.shape[:2]

    detections = []
    for i in range(len(scores)):
        if (scores[i] > min_conf_thresh) and (scores[i] <= 1.0):
            ymin = int(max(1, boxes[i][0] * proc_h))
            xmin = int(max(1, boxes[i][1] * proc_w))
            ymax = int(min(proc_h, boxes[i][2] * proc_h))
            xmax = int(min(proc_w, boxes[i][3] * proc_w))

            label = labels[int(classes[i])]  # Label for class index

            # Keep or remove this filter as per your preference
            # Currently: SKIP person class (since you're already tracking faces separately)
            if label.lower() == "person":
                continue

            # Store as (top, left, bottom, right) in PROC coordinates
            detections.append(
                {
                    "box": (ymin, xmin, ymax, xmax),
                    "label": label,
                    "confidence": float(scores[i]),
                }
            )

    return detections


# -----------------------
# Plotting & Interval Reporting
# -----------------------

def plot_detections_over_time(detections_log, total_duration):
    if not detections_log:
        print("No detections to plot.")
        return

    keys = sorted({(d["label"], d["kind"]) for d in detections_log})
    plt.figure(figsize=(10, 6))

    for (label, kind) in keys:
        times = [
            d["time"]
            for d in detections_log
            if d["label"] == label and d["kind"] == kind
        ]
        confs = [
            d["confidence"]
            for d in detections_log
            if d["label"] == label and d["kind"] == kind
        ]
        if not times:
            continue
        plt.plot(times, confs, marker="o", linestyle="-", label=f"{label} ({kind})")

    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.title("Detections over time")
    plt.ylim(0, 1.05)
    plt.xlim(0, total_duration)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_file = "detections_over_time.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to {output_file}")
    plt.close()


def print_detection_intervals(detections_log, max_gap=1.0):
    if not detections_log:
        print("No detections to summarize.")
        return

    keys = sorted({(d["label"], d["kind"]) for d in detections_log})
    print("\nDetection intervals:")
    for (label, kind) in keys:
        times = sorted(
            d["time"] for d in detections_log if d["label"] == label and d["kind"] == kind
        )
        if not times:
            continue
        intervals = []
        start = times[0]
        prev = times[0]
        for t in times[1:]:
            if t - prev <= max_gap:
                prev = t
            else:
                intervals.append((start, prev))
                start = t
                prev = t
        intervals.append((start, prev))

        print(f"- {label} ({kind}):")
        for s, e in intervals:
            print(f"    from {s:.1f}s to {e:.1f}s (duration {e - s:.1f}s)")


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Path to the TFLite model file",
        default="tfrpie/models/model.tflite",
    )
    parser.add_argument(
        "--labels",
        help="Path to the labels file",
        default="tfrpie/models/labels.txt",
    )
    parser.add_argument(
        "--threshold",
        help="Minimum confidence threshold for object detection",
        default=0.5,
    )
    parser.add_argument(
        "--resolution",
        help="Webcam resolution WxH (ignored for Pi camera, which uses 2592x1944)",
        default="1920x1080",
    )
    parser.add_argument(
        "--encodings",
        help="Path to face encodings pickle",
        default="face_rec/encodings.pickle",
    )
    parser.add_argument(
        "--duration",
        help="Duration of experiment in seconds",
        type=float,
        default=DURATION_SECONDS,
    )
    parser.add_argument(
        "--max-fps",
        help="Maximum frames per second to process (None = process all frames)",
        type=float,
        default=MAX_FPS,
    )
    parser.add_argument(
        "--webcam",
        action="store_false",
        dest="pie_cam",
        default=True,
        help="Use webcam instead of Raspberry Pi camera (default: use Pi camera)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without showing OpenCV window (for headless / speed)",
    )
    parser.add_argument(
        "--output-video",
        help="Path to save output video file (e.g., output.mp4).",
        default=None,
    )
    parser.add_argument(
        "--video-fps",
        help="FPS for output video (default: 30)",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help="Print per-frame timing for debugging (slower).",
    )

    args = parser.parse_args()

    # Update PIE_CAM from main parser (in case it was changed)
    global PIE_CAM
    PIE_CAM = args.pie_cam

    min_conf_thresh = float(args.threshold)
    show_window = not args.no_display
    debug_timing = args.debug_timing

    # Decide full capture resolution
    if PIE_CAM:
        # Full Pi camera resolution
        full_w, full_h = 1920, 1080
    else:
        resW, resH = args.resolution.split("x")
        full_w, full_h = int(resW), int(resH)

    # Load models
    known_face_encodings, known_face_names = load_face_encodings(args.encodings)
    (
        interpreter,
        labels,
        input_details,
        output_details,
        input_height,
        input_width,
        floating_model,
    ) = load_tflite_model(args.model, args.labels)

    # Start video stream at full resolution
    print("[INFO] starting video stream...")
    videostream = VideoStream(
    resolution=(1640, 1232), framerate=10, use_picamera=PIE_CAM
    ).start()
    time.sleep(1)

    detections_log = []  # list of dicts: time, label, kind, confidence

    experiment_start = time.time()
    max_fps = args.max_fps
    frame_interval = 1.0 / max_fps if max_fps else None
    next_allowed_time = experiment_start

    # Video recording setup (simple: write each processed frame at full resolution)
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.output_video, fourcc, args.video_fps, (full_w, full_h)
        )
        if not video_writer.isOpened():
            print(f"[WARNING] Failed to open video writer for {args.output_video}")
            video_writer = None
        else:
            print(f"[INFO] Recording video to {args.output_video} at {args.video_fps} FPS")

    # Detector scheduling & stats
    frame_idx = 0
    frames_processed = 0
    last_face_dets = []
    last_obj_dets = []
    last_log_time = -1e9

    face_times = []
    obj_times = []
    total_frame_times = []

    while True:
        now = time.time()
        elapsed = now - experiment_start
        if elapsed > args.duration:
            print("[INFO] experiment duration reached, stopping...")
            break

        # Frame rate limiting (skip if too soon)
        if max_fps is not None and frame_interval is not None:
            if now < next_allowed_time:
                continue
            next_allowed_time = now + frame_interval

        proc_start = time.time()

        frame_full = videostream.read()
        if frame_full is None:
            continue

        # Convert frame format if using Picamera2 (BGRA/RGBA to BGR)
        if PIE_CAM:
            if len(frame_full.shape) == 3 and frame_full.shape[2] == 4:
                frame_full = cv2.cvtColor(frame_full, cv2.COLOR_BGRA2BGR)
            elif len(frame_full.shape) == 3 and frame_full.shape[2] == 3:
                frame_full = cv2.cvtColor(frame_full, cv2.COLOR_RGB2BGR)

        # Ensure full_w, full_h match actual frame size
        full_h_actual, full_w_actual = frame_full.shape[:2]
        # If different from configured, update (safety)
        full_w, full_h = full_w_actual, full_h_actual

        # Create a smaller copy for processing (no crop, just scale)
        frame_proc = cv2.resize(frame_full, (PROC_W, PROC_H))

        frame_idx += 1

        # Decide whether to run detectors on this processed frame
        run_face = (frame_idx % FACE_EVERY_N_FRAMES == 0)
        run_obj = (frame_idx % OBJ_EVERY_N_FRAMES == 0)

        # Face recognition on processed frame
        face_time = 0.0
        if run_face:
            t_face_start = time.time()
            last_face_dets = recognize_faces(
                frame_proc, known_face_encodings, known_face_names, cv_scaler=CV_SCALER
            )
            face_time = (time.time() - t_face_start) * 1000.0  # ms
            if debug_timing:
                print(
                    f"[TIMING] Frame {frame_idx} - Face: {face_time:.2f}ms ({len(last_face_dets)} faces)"
                )

        # Object detection on processed frame
        obj_time = 0.0
        if run_obj:
            t_obj_start = time.time()
            last_obj_dets = detect_objects(
                frame_proc,
                interpreter,
                labels,
                min_conf_thresh,
                input_details,
                output_details,
                input_height,
                input_width,
                floating_model,
            )
            obj_time = (time.time() - t_obj_start) * 1000.0  # ms
            if debug_timing:
                print(
                    f"[TIMING] Frame {frame_idx} - Objects: {obj_time:.2f}ms ({len(last_obj_dets)} objects)"
                )

        face_dets_proc = last_face_dets
        obj_dets_proc = last_obj_dets

        # Scale factor from processed to full frame
        scale_x = full_w / float(PROC_W)
        scale_y = full_h / float(PROC_H)

        # Scale detections from processed coords to full frame coords
        face_dets_full = []
        for det in face_dets_proc:
            top_p, left_p, bottom_p, right_p = det["box"]
            top = int(top_p * scale_y)
            left = int(left_p * scale_x)
            bottom = int(bottom_p * scale_y)
            right = int(right_p * scale_x)
            face_dets_full.append(
                {
                    "box": (top, left, bottom, right),
                    "label": det["label"],
                    "confidence": det["confidence"],
                }
            )

        obj_dets_full = []
        for det in obj_dets_proc:
            top_p, left_p, bottom_p, right_p = det["box"]
            top = int(top_p * scale_y)
            left = int(left_p * scale_x)
            bottom = int(bottom_p * scale_y)
            right = int(right_p * scale_x)
            obj_dets_full.append(
                {
                    "box": (top, left, bottom, right),
                    "label": det["label"],
                    "confidence": det["confidence"],
                }
            )

        # Logging detections (no coordinates needed)
        log_this_frame = False
        if LOG_EVERY_SECONDS <= 0.0:
            log_this_frame = True
        else:
            if (elapsed - last_log_time) >= LOG_EVERY_SECONDS:
                log_this_frame = True
                last_log_time = elapsed

        if log_this_frame:
            for det in face_dets_full:
                detections_log.append(
                    {
                        "time": elapsed,
                        "label": det["label"],
                        "kind": "face",
                        "confidence": det["confidence"],
                    }
                )
            for det in obj_dets_full:
                detections_log.append(
                    {
                        "time": elapsed,
                        "label": det["label"],
                        "kind": "object",
                        "confidence": det["confidence"],
                    }
                )

        # Draw detections on FULL frame
        # Faces: yellow boxes
        for det in face_dets_full:
            top, left, bottom, right = det["box"]
            name = det["label"]
            conf = det["confidence"]
            cv2.rectangle(frame_full, (left, top), (right, bottom), (0, 255, 255), 2)
            label = f"{name}: {int(conf * 100)}%"
            cv2.rectangle(
                frame_full,
                (left - 3, top - 35),
                (right + 3, top),
                (0, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame_full,
                label,
                (left + 6, top - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 0),
                1,
            )

        # Objects: green boxes
        current_obj_count = 0
        for det in obj_dets_full:
            top, left, bottom, right = det["box"]
            label_text = det["label"]
            conf = det["confidence"]

            cv2.rectangle(frame_full, (left, top), (right, bottom), (10, 255, 0), 2)

            label = "%s: %d%%" % (label_text, int(conf * 100))
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            label_ymin = max(top, label_size[1] + 10)
            cv2.rectangle(
                frame_full,
                (left, label_ymin - label_size[1] - 10),
                (left + label_size[0], label_ymin + base_line - 10),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame_full,
                label,
                (left, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            current_obj_count += 1

        # Per-frame timing
        proc_time_ms = (time.time() - proc_start) * 1000.0
        total_frame_times.append(proc_time_ms)
        if run_face and face_time > 0:
            face_times.append(face_time)
        if run_obj and obj_time > 0:
            obj_times.append(obj_time)

        # Accurate global FPS
        frames_processed += 1
        elapsed_global = time.time() - experiment_start
        global_fps = frames_processed / elapsed_global if elapsed_global > 0 else 0.0

        # Overlay text on FULL frame
        cv2.putText(
            frame_full,
            f"FPS: {global_fps:.2f}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 55),
            2,
            cv2.LINE_AA,
        )


        # Write to video (if enabled)
        if video_writer:
            video_writer.write(frame_full)

        if show_window:
            cv2.imshow("Face + Object Detector", frame_full)
            if cv2.waitKey(1) == ord("q"):
                print("[INFO] 'q' pressed, exiting...")
                break

    # Clean up
    if show_window:
        cv2.destroyAllWindows()
    videostream.stop()

    if video_writer:
        video_writer.release()
        print(f"[INFO] Video saved to {args.output_video}")

    total_duration = time.time() - experiment_start

    # Timing statistics summary
    print("\n" + "=" * 60)
    print("PROCESSING TIME STATISTICS")
    print("=" * 60)
    if face_times:
        avg_face_time = sum(face_times) / len(face_times)
        print(f"Face Recognition:")
        print(f"  - Frames processed: {len(face_times)}")
        print(f"  - Average time: {avg_face_time:.2f}ms")
        print(f"  - Min time: {min(face_times):.2f}ms")
        print(f"  - Max time: {max(face_times):.2f}ms")

    if obj_times:
        avg_obj_time = sum(obj_times) / len(obj_times)
        print(f"Object Detection:")
        print(f"  - Frames processed: {len(obj_times)}")
        print(f"  - Average time: {avg_obj_time:.2f}ms")
        print(f"  - Min time: {min(obj_times):.2f}ms")
        print(f"  - Max time: {max(obj_times):.2f}ms")

    if total_frame_times:
        avg_frame_time = sum(total_frame_times) / len(total_frame_times)
        print(f"Total Frame Processing:")
        print(f"  - Frames processed: {len(total_frame_times)}")
        print(f"  - Average time: {avg_frame_time:.2f}ms")
        print(f"  - Min time: {min(total_frame_times):.2f}ms")
        print(f"  - Max time: {max(total_frame_times):.2f}ms")
        print(f"  - Average FPS (by time): {1000.0 / avg_frame_time:.2f}")
    print("=" * 60 + "\n")

    print_detection_intervals(detections_log, max_gap=1.0)
    plot_detections_over_time(detections_log, total_duration)


if __name__ == "__main__":
    main()