# Multi-Object Tracking & Video Memory

## Learning Objectives

1. Implement a frame-to-frame object association pipeline using the Hungarian algorithm on IoU cost matrices.
2. Configure track lifecycle states (tentative, confirmed, lost) and tune deletion thresholds.
3. Build a video memory buffer that maintains per-track feature histories for re-identification after occlusion.
4. Compare Kalman-filter-based motion prediction against appearance-based re-ID for handling identity switches.
5. Diagnose common MOT failure modes (ID fragmentation, track drift, duplicate tracks) from tracking logs.

---

## Beat 1: Hook

You have a detector that draws boxes around objects in single frames. The moment you need to count unique objects over time—or remember that "track #7 disappeared behind the shelf three seconds ago"—you need a tracker. Multi-object tracking is the engineering problem of maintaining object identity across frames when detections are noisy, objects occlude each other, and new objects enter the scene. Video memory is the persistence layer that lets you re-identify objects after they disappear and reappear.

---

## Beat 2: Concept

### The tracking-by-detection loop

Every frame produces detections. Every track has a predicted position. The core loop: predict where existing tracks will be, match incoming detections to those predictions, update matched tracks, spawn new tracks for unmatched detections, age out tracks that haven't been matched recently.

### The assignment problem

Given N tracks and M detections, which detection belongs to which track? This is a bipartite matching problem. You build a cost matrix (typically 1 - IoU or feature distance), then solve it with the Hungarian algorithm. Unmatched detections become new tracks. Unmatched tracks increment their "frames since last match" counter.

### Track state machine

```
Detection → Tentative → Confirmed → Lost → Deleted
              ↑                        │
              └────────────────────────┘
                   (if re-matched within window)
```

Tentative tracks haven't been seen enough times to trust. Confirmed tracks have been matched for N consecutive frames. Lost tracks are held in memory for a grace period before deletion.

### Motion prediction: Kalman filter

Tracks don't just sit still between frames. A constant-velocity Kalman filter predicts where a track will appear next frame based on its velocity. This prediction becomes the anchor for IoU matching.

### Re-identification with appearance features

IoU fails when objects move fast or occlude. Solution: attach a feature vector (from a ReID model or the detector's embedding head) to each track. When motion matching fails, fall back to cosine-distance matching between track feature history and detection features. This is the DeepSORT insight.

### Video memory

A track object is not just a bounding box. It's a data structure holding:
- Current and predicted bounding box
- Feature vector history (ring buffer of last K embeddings)
- Track ID
- State (tentative / confirmed / lost)
- Frame count since last match
- Class label and confidence history

This per-track memory is what lets you re-identify after occlusion and produce coherent trajectories.

### Why trackers fail

- ID fragmentation: same object gets two track IDs because it was occluded too long
- Track drift: Kalman prediction diverges from actual motion during long occlusions
- Duplicate tracks: two tracks claim the same detection because cost matrix thresholds are wrong
- Feature collapse: ReID embeddings aren't discriminative enough for objects with similar appearance

---

## Beat 3: Demo

### Demo 1: IoU cost matrix and Hungarian assignment from scratch

Build a 3-track × 4-detection cost matrix. Compute IoU between every (track, detection) pair. Solve with `scipy.optimize.linear_sum_assignment`. Show which detections matched which tracks, which detections spawned new tracks, and which tracks went unmatched.

Output: printed cost matrix, assignment pairs, unmatched detections, unmatched tracks.

### Demo 2: Kalman filter prediction for a single track

Initialize a constant-velocity Kalman filter (state: [x, y, w, h, vx, vy, vw, vh]). Feed it 10 frames of a moving bounding box. Print predicted vs. actual position for each frame to show prediction error accumulating when the object changes direction.

Output: frame-by-frame table of predicted position, actual position, error magnitude.

### Demo 3: Minimal tracker with video memory

Implement a `Track` class with state machine and feature ring buffer. Implement a `Tracker` class with predict → match → update → spawn → prune loop. Run it on 20 synthetic frames of 3 moving objects (one occluded for 5 frames). Print track IDs, states, and feature buffer lengths each frame.

Output: frame logs showing track state transitions and re-identification after occlusion.

---

## Beat 4: Use It

### GTM Redirect

Multi-object tracking maps to **Zone 2 (Data Foundation) → Intent Signal Processing** and **Zone 4 (Activation) → Event-Driven Triggers** [CITATION NEEDED — concept: MOT applied to GTM intent signal deduplication].

Specific mechanism: when processing event streams (page views, content engagements, email opens), the same problem as MOT appears—multiple signals from the same account need to be associated into a coherent "track" over time. The assignment problem (is this signal from the same entity?) is the same Hungarian/IoU logic applied to entity matching instead of bounding boxes.

Exercise hooks:
- (Easy) Modify the IoU cost matrix to use string similarity instead of spatial IoU, matching company name variants.
- (Medium) Build a signal tracker that groups time-series events into tracks with confirmed/tentative states, requiring 3 matches within a 7-day window before confirming an account track.
- (Hard) Implement a deduplication tracker for CRM contacts that uses feature vectors (job title, company, location embeddings) for re-identification when names are misspelled across sources.

---

## Beat 5: Ship It

Build a command-line MOT system that reads a directory of frames (images), runs detection with a lightweight detector, and outputs a JSON file with track IDs, bounding box histories, and state transitions. The system accepts config for: IoU threshold, max age before track deletion, min hits for confirmation, and feature buffer size.

This is the same architecture as ByteTrack [CITATION NEEDED — concept: ByteTrack low-confidence detection matching] applied to your own detection pipeline.

Exercise hooks:
- (Easy) Ship the tracker with IoU-only matching and Kalman prediction on a provided sequence of 50 annotated frames.
- (Medium) Add appearance-based matching using embeddings from a pre-trained feature extractor, with a fallback cascade (IoU first, then features for unmatched).
- (Hard) Implement the full DeepSORT two-stage matching: high-confidence detections match first, low-confidence detections match to remaining tracks second. Measure ID switch count against ground truth.

---

## Beat 6: Evaluate It

### Conceptual checks

- Given a 5-track × 7-detection cost matrix with specific IoU values, trace through the Hungarian algorithm output by hand and identify which tracks go unmatched.
- A track has been in "lost" state for 12 frames (max_age = 15). It reappears with an IoU of 0.3 and a cosine feature distance of 0.15. Thresholds: IoU minimum 0.3, feature distance maximum 0.2. Does it re-associate? What if the feature buffer was cleared at frame 10?
- Three objects cross paths. Explain the sequence of state transitions that produces ID fragmentation vs. correct re-identification.

### Mechanism tracing

Given tracker output logs from Demo 3 showing an occlusion event:
- Identify the exact frame where the Kalman prediction diverges from reappearance position
- Calculate how many frames of feature history remain when the track re-associates
- Explain why the track confirmed rather than spawned as new

### Failure diagnosis

Given a tracking output with known bugs (provided as JSON):
- Identify tracks with ID fragmentation (same physical object, two IDs)
- Identify duplicate tracks (two IDs claiming the same detections)
- Propose which threshold or parameter change would fix each issue