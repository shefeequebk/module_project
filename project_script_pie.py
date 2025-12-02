import argparse
import time
import pickle
import os

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

DURATION_SECONDS = 20   # total time to run (can be changed with --duration)
CV_SCALER = 4           # downscale factor for face detection (higher = faster, less accurate)
MAX_FPS = 5             # maximum frames per second to process (None = process all frames)

# How often to run each detector (in processed frames)
FACE_EVERY_N_FRAMES = 2   # run face recognition every 2 processed frames
OBJ_EVERY_N_FRAMES = 1    # run object detection every processed frame

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


class VideoStream:
    """Camera object that controls video streaming from webcam or Picamera2 in a separate thread."""

    def __init__(self, resolution=(640, 480), framerate=30, use_picamera=False):
        self.use_picamera = use_picamera
        self.resolution = resolution
        self.framerate = framerate

        if self.use_picamera:
            # Initialize Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": resolution}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.frame = self.picam2.capture_array()
            self.grabbed = True
        else:
            # Initialize the USB/webcam
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
                self.frame = self.picam2.capture_array()
                self.grabbed = True
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


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


def recognize_faces(frame, known_face_encodings, known_face_names, cv_scaler=4):
    """Run face detection + recognition on a BGR frame."""
    # Downscale for speed
    small_frame = cv2.resize(
        frame, (0, 0), fx=1.0 / cv_scaler, fy=1.0 / cv_scaler
    )
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations_small = face_recognition.face_locations(rgb_small)
    # OPTIMIZATION: use the "small" model instead of "large"
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

        # Scale back to original frame size
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        detections.append(
            {
                "box": (top, right, bottom, left),
                "label": name,
                "confidence": float(confidence),
            }
        )

    return detections


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
    frame,
    interpreter,
    labels,
    min_conf_thresh,
    imW,
    imH,
    input_details,
    output_details,
    input_height,
    input_width,
    floating_model,
):
    """Run TFLite object detection on a BGR frame."""
    input_mean = 127.5
    input_std = 127.5

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (input_width, input_height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]  # Bounding box coords
    classes = interpreter.get_tensor(output_details[1]["index"])[0]  # Class indices
    scores = interpreter.get_tensor(output_details[2]["index"])[0]  # Confidence scores

    detections = []
    # Loop over all detections and keep those above threshold
    for i in range(len(scores)):
        if (scores[i] > min_conf_thresh) and (scores[i] <= 1.0):
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))

            label = labels[int(classes[i])]  # Label for class index

            # Example filter: skip "person" if not needed
            if label.lower() == "person":
                continue

            # Store as (top, right, bottom, left)
            detections.append(
                {
                    "box": (ymin, xmax, ymax, xmin),
                    "label": label,
                    "confidence": float(scores[i]),
                }
            )

    return detections


def plot_detections_over_time(detections_log, total_duration):
    if not detections_log:
        print("No detections to plot.")
        return

    # Group by (label, kind)
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

    # Save plot to file instead of showing (works in headless environments)
    output_file = "detections_over_time.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to {output_file}")
    plt.close()  # Close the figure to free memory


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
        help="Webcam resolution WxH (must be supported by camera)",
        default="1280x720",
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

    args = parser.parse_args()
    # Update PIE_CAM from main parser (in case it was changed)
    global PIE_CAM
    PIE_CAM = args.pie_cam

    min_conf_thresh = float(args.threshold)
    resW, resH = args.resolution.split("x")
    imW, imH = int(resW), int(resH)

    show_window = not args.no_display

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

    # Start video stream
    print("[INFO] starting video stream...")
    videostream = VideoStream(
        resolution=(imW, imH), framerate=30, use_picamera=PIE_CAM
    ).start()
    time.sleep(1)

    detections_log = []  # list of dicts: time, label, kind, confidence
    freq = cv2.getTickFrequency()
    experiment_start = time.time()

    # Frame rate limiting
    max_fps = args.max_fps
    frame_interval = 1.0 / max_fps if max_fps else None
    last_frame_time = 0.0

    # Detector scheduling
    frame_idx = 0
    last_face_dets = []
    last_obj_dets = []

    # Logging schedule
    last_log_time = -1e9  # big negative so first log always allowed

    while True:
        now = time.time()
        elapsed = now - experiment_start
        if elapsed > args.duration:
            print("[INFO] experiment duration reached, stopping...")
            break

        # Frame skipping logic: only process if enough time has passed
        if max_fps is not None and frame_interval is not None:
            if (now - last_frame_time) < frame_interval:
                continue  # Skip this frame
            last_frame_time = now

        t1 = cv2.getTickCount()
        frame = videostream.read()
        if frame is None:
            continue

        # Convert frame format if using Picamera2 (BGRA/RGBA to BGR)
        if PIE_CAM:
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_idx += 1

        # Decide whether to run detectors on this processed frame
        run_face = (frame_idx % FACE_EVERY_N_FRAMES == 0)
        run_obj = (frame_idx % OBJ_EVERY_N_FRAMES == 0)

        # Face recognition
        if run_face:
            last_face_dets = recognize_faces(
                frame, known_face_encodings, known_face_names, cv_scaler=CV_SCALER
            )

        # Object detection
        if run_obj:
            last_obj_dets = detect_objects(
                frame,
                interpreter,
                labels,
                min_conf_thresh,
                imW,
                imH,
                input_details,
                output_details,
                input_height,
                input_width,
                floating_model,
            )

        face_dets = last_face_dets
        obj_dets = last_obj_dets

        # Decide whether to log this frame's detections
        log_this_frame = False
        if LOG_EVERY_SECONDS <= 0.0:
            log_this_frame = True
        else:
            if (elapsed - last_log_time) >= LOG_EVERY_SECONDS:
                log_this_frame = True
                last_log_time = elapsed

        if log_this_frame:
            # Log detections for plotting later
            for det in face_dets:
                detections_log.append(
                    {
                        "time": elapsed,
                        "label": det["label"],
                        "kind": "face",
                        "confidence": det["confidence"],
                    }
                )
            for det in obj_dets:
                detections_log.append(
                    {
                        "time": elapsed,
                        "label": det["label"],
                        "kind": "object",
                        "confidence": det["confidence"],
                    }
                )

        # Draw detections on frame
        # Faces: yellow boxes
        for det in face_dets:
            top, right, bottom, left = det["box"]
            name = det["label"]
            conf = det["confidence"]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            label = f"{name}: {int(conf * 100)}%"
            cv2.rectangle(
                frame,
                (left - 3, top - 35),
                (right + 3, top),
                (0, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (left + 6, top - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 0),
                1,
            )

        # Objects: green boxes
        current_obj_count = 0
        for det in obj_dets:
            top, right, bottom, left = det["box"]
            label_text = det["label"]
            conf = det["confidence"]

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (10, 255, 0), 2)

            # Draw label
            label = "%s: %d%%" % (label_text, int(conf * 100))
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            label_ymin = max(top, label_size[1] + 10)
            cv2.rectangle(
                frame,
                (left, label_ymin - label_size[1] - 10),
                (left + label_size[0], label_ymin + base_line - 10),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (left, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            current_obj_count += 1

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        fps = 1.0 / time1 if time1 > 0 else 0.0

        # Draw framerate and detection count in corner of frame
        cv2.putText(
            frame,
            "FPS: {0:.2f}".format(fps),
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 55),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Total Detection Count : " + str(current_obj_count),
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 55),
            2,
            cv2.LINE_AA,
        )

        if show_window:
            cv2.imshow("Face + Object Detector", frame)
            if cv2.waitKey(1) == ord("q"):
                print("[INFO] 'q' pressed, exiting...")
                break

    # Clean up
    if show_window:
        cv2.destroyAllWindows()
    videostream.stop()

    total_duration = time.time() - experiment_start
    print_detection_intervals(detections_log, max_gap=1.0)
    plot_detections_over_time(detections_log, total_duration)


if __name__ == "__main__":
    main()
