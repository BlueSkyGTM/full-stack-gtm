## Ship It

Now we package the frame-level extractor into a CLI tool that processes an entire video file. This mirrors the batch enrichment pattern: ingest raw assets (video frames at 1 fps), run the transform (pose estimation), and export structured records (JSONL lines). Each line is one timestamped observation, ready for downstream scoring, aggregation, or database insertion.

```python
import argparse
import cv2
import json
import mediapipe as mp

KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def process_video(video_path, output_path, min_confidence=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_interval = int(fps)
    if frame_interval < 1:
        frame_interval = 1

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_confidence
    )

    frame_idx = 0
    records_written = 0

    with open(output_path, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                results = pose.process(frame)

                if results.pose_landmarks:
                    keypoints = []
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"point_{idx}"
                        vis = round(lm.visibility, 4)
                        if vis >= min_confidence:
                            keypoints.append({
                                "name": name,
                                "x": round(lm.x, 4),
                                "y": round(lm.y, 4),
                                "confidence": vis
                            })

                    left_ear_vis = results.pose_landmarks.landmark[7].visibility
                    right_ear_vis = results.pose_landmarks.landmark[8].visibility
                    ear_delta = abs(left_ear_vis - right_ear_vis)
                    facing_camera = ear_delta < 0.3

                    record = {
                        "timestamp": round(timestamp, 2),
                        "keypoints": keypoints,
                        "keypoint_count": len(keypoints),
                        "facing_camera": facing_camera,
                        "ear_confidence_delta": round(ear_delta, 4)
                    }
                else:
                    record = {
                        "timestamp": round(timestamp, 2),
                        "keypoints": [],
                        "keypoint_count": 0,
                        "facing_camera": None,
                        "ear_confidence_delta": None
                    }

                f.write(json.dumps(record) + "\n")
                records_written += 1

            frame_idx += 1

    cap.release()
    pose.close()
    print(f"Processed {frame_idx} frames, wrote {records_written} records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose keypoints from video to JSONL")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("output", help="Path to output JSONL file")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum confidence threshold for keypoints (default: 0.3)")
    args = parser.parse_args()

    process_video(args.video, args.output, args.min_confidence)
```

Save this as `pose_cli.py` and run it against any video file:

```bash
python pose_cli.py recorded_zoom_call.mp4 pose_output.jsonl --min-confidence 0.5
```

To confirm the output, print the first few lines:

```bash
head -n 3 pose_output.jsonl | python -m json.tool
```

Each line is a self-contained record: timestamp, filtered keypoints, keypoint count, and the facing-camera flag. Changing `--min-confidence` from 0.3 to 0.5 will visibly reduce the `keypoint_count` field as low-confidence keypoints get filtered out, which is the observable proof that the threshold is doing work. This JSONL file is now a structured dataset — importable into any analytics tool, database, or conversation intelligence pipeline that expects per-frame pose observations as typed fields.