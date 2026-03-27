# Truck Load Fill Estimation

This project estimates truck load fill level from images and videos using two YOLO models:

- a truck detection model
- a segmentation model that detects the truck `Box` and the `content` inside it

The repository contains two main execution paths:

- `size_estimation_v4.py`: run fill estimation on a single image, or on the first frame of a video
- `auto_select_truck_frames.py`: process a full video, track trucks through sampled frames, choose one best frame per truck, and report the fill result

The project is designed for fixed-camera scenes where trucks enter the frame, pass through the scene, and then leave.

## Repository Contents

- [size_estimation_v4.py](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/size_estimation_v4.py)
- [auto_select_truck_frames.py](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/auto_select_truck_frames.py)
- [requirements.txt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/requirements.txt)
- [truck.pt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/Yolo-wight/truck.pt)
- [best_size_March_25.pt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/Yolo-wight/best_size_March_25.pt)
- [sizev2.pt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/Yolo-wight/sizev2.pt)

## Models Used

### 1. Truck Detection Model

Path:

- [truck.pt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/Yolo-wight/truck.pt)

Purpose:

- detect trucks in the full frame
- provide a truck bounding box
- isolate the truck region before segmentation

This model is used in both scripts.

### 2. Segmentation Model

Default path:

- [best_size_March_25.pt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/Yolo-wight/best_size_March_25.pt)

Purpose:

- segment the truck container area as `Box`
- segment the truck load/material area as `content`

This is the default segmentation model for all current runs.

Older model still present in the repo:

- [sizev2.pt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/Yolo-wight/sizev2.pt)

It is kept in the repository, but it is no longer the default runtime model.

## High-Level Pipeline

The project works in this order:

1. Read an image or a video frame.
2. Detect truck(s) in the frame using `truck.pt`.
3. Crop each detected truck.
4. Run segmentation on the cropped truck using `best_size_March_25.pt`.
5. Find the `Box` mask and `content` mask.
6. Estimate fill level from the relative position of `content` inside the `Box`.
7. For videos, choose one representative frame per truck and report a single result.

## Fill Estimation Logic

The fill percentage logic is implemented in [size_estimation_v4.py](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/size_estimation_v4.py) and mirrored in the automated wrapper.

The algorithm does not calculate volume. It estimates fill level from vertical occupancy inside the segmented box.

Detailed logic:

1. Restrict `content_mask` to the area inside `box_mask`.
2. Apply morphological closing to reduce small holes and noise.
3. For the `box_mask`, find:
   - the first row where the box appears
   - the last row where the box appears
4. For the `content_mask`, find:
   - the first row where content appears
5. Compute:
   - `box_height = box_bottom - box_top`
   - `content_height = box_bottom - content_top`
6. Fill percentage:

```text
fill = (content_height / box_height) * 100
```

Then clamp to `[0, 100]`.

Interpretation:

- if `content` reaches close to the top of the `Box`, fill is high
- if `content` is absent, fill becomes `0`
- if no valid `Box` is found, no valid fill result can be produced

## Script 1: size_estimation_v4.py

Path:

- [size_estimation_v4.py](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/size_estimation_v4.py)

### What it does

This script is the core single-frame estimator.

It:

- loads the detection and segmentation models
- accepts an image path as input
- also accepts a video path and reads the first frame
- detects truck(s)
- segments each truck crop
- computes fill percentage
- draws overlays
- shows the result in a preview window
- saves `result.jpg` unless disabled by environment variables

### Main functions and sections

#### `resize_mask(mask, target_shape)`

Used to resize a segmentation mask from model output resolution to the truck crop resolution.

Uses nearest-neighbor interpolation so binary masks stay clean.

#### `resize_for_display(image, max_width=960, max_height=540)`

Resizes the final displayed result window so large images fit on screen.

This changes only visualization, not inference or saved output resolution.

#### `load_frame(input_path)`

Input loader for:

- images: `cv2.imread`
- videos: `cv2.VideoCapture(...).read()` first frame only

#### `calculate_fill_percentage(box_mask, content_mask)`

Implements the fill formula described above.

#### Model loading

This section sets:

```python
truck_model = YOLO(...truck.pt)
size_model = YOLO(...best_size_March_25.pt)
```

It also detects whether CUDA is available:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### Truck detection loop

The script loops over `results.boxes` from the truck detector.

For each detection:

- check confidence threshold
- keep only detections whose class name is `truck`
- crop the truck region

#### Segmentation loop

For each truck crop:

- run the segmentation model
- inspect segmentation classes
- store mask for `Box`
- store mask for `content`

Class matching is done with:

```python
if class_name.lower() == "box"
elif class_name.lower() == "content"
```

#### Overlay drawing

Overlay colors:

- `Box` = blue `(255, 0, 0)`
- `content` = green `(0, 255, 0)`

Then a transparency blend is applied with:

```python
cv2.addWeighted(overlay, 0.4, truck_crop, 0.6, 0)
```

#### Final reporting

If multiple trucks exist in the frame:

- the script stores each truck fill in `all_fills`
- reports the average as `FINAL FILL`

### Environment controls

The script supports:

- `YOLO_NO_DISPLAY=1`
- `YOLO_NO_SAVE=1`

These are used by the automation wrapper so batch processing does not block on GUI windows.

## Script 2: auto_select_truck_frames.py

Path:

- [auto_select_truck_frames.py](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/auto_select_truck_frames.py)

### What it does

This script processes a full video and tries to produce one fill result per truck.

It is designed for:

- fixed camera
- trucks moving through the scene
- one representative frame needed per truck

### Main stages

#### 1. Load models once

Function:

- `load_models()`

This loads:

- `truck_model`
- `size_model`
- `truck_classes`
- `size_classes`
- device

The current optimized implementation keeps models loaded in memory for the whole run.

#### 2. Sample the video

Function:

- `analyze_video(...)`

Sampling logic:

- read video FPS
- compute frame interval from requested `--fps`
- only process sampled frames

Example:

- video FPS = 25
- requested sampling = 5
- process every 5th frame

#### 3. Detect trucks on sampled frames

Function:

- `detect_trucks(...)`

For each sampled frame:

- run `truck.pt`
- filter by confidence
- filter by truck class
- compute a visibility score

#### 4. Score detections

Function:

- `detection_score(...)`

This score is used to estimate frame quality for selecting candidate frames.

Score components:

- detection confidence
- bbox area relative to frame size
- closeness of bbox center to frame center
- blur/sharpness estimate
- penalty if bbox touches frame borders

This is not the final fill score. It is a visibility/quality score.

#### 5. Track trucks across sampled frames

Function:

- `match_tracks(...)`

The script links truck detections across frames using IoU between bounding boxes.

Each track represents one truck passage through the scene.

Track state stores:

- `track_id`
- latest bbox
- hit count
- missed count
- history of sampled detections

Track finalization happens when:

- the truck is no longer matched for more than `MAX_MISSED_SAMPLES`
- and the track has enough detections (`MIN_TRACK_HITS`)

#### 6. Save sampled candidate frames

Function:

- `save_frame(...)`

Every accepted sampled frame for a track is saved into the output folder.

Naming pattern:

```text
truck_001_frame_00050.jpg
```

This makes later review and debugging easy.

#### 7. Live preview

Function:

- `draw_preview(...)`

While the video is being processed, the script can display:

- truck bounding boxes
- track labels
- segmentation overlay on each current track

Current behavior:

- preview window is scaled smaller for easier viewing
- preview window is resizable
- press `q` to stop early

#### 8. Candidate frame selection for fill evaluation

Function:

- `select_fill_candidates(track)`

To avoid evaluating every sampled frame by default, the script now evaluates only a short list of candidate frames:

- top visibility-score frames
- a few frames near the middle of the truck passage

This is a speed optimization.

#### 9. In-process fill estimation

Functions:

- `estimate_fill_for_frame(...)`
- `evaluate_track_detections(...)`

The script now performs fill evaluation directly in the same Python process, reusing the already loaded models.

This avoids the older expensive behavior where a separate Python process was launched for each frame.

That optimization significantly reduces post-video delay.

#### 10. Final frame selection

Function:

- `select_best_detection(track)`

Final selection is no longer based only on truck visibility.

Current rule:

- if multiple positive non-trivial fill results exist, prefer that stable positive-fill cluster
- choose a frame near the median positive fill
- also prefer a frame near the temporal midpoint of the track
- use detection score as a tie-breaker

Fallbacks:

- if no positive fill frames exist, use the best valid fill frame
- if no valid fill frames exist, use the best visibility-scored frame

This change was added because some videos showed a good truck view but a poor fill frame if only visibility was used.

#### 11. Optional CSV generation

Default behavior:

- no CSVs are written unless explicitly requested

Optional flags:

- `--write-summary-csv`
- `--write-all-frame-csv`

`--write-summary-csv` writes one selected result per truck.

`--write-all-frame-csv` writes fill results for every sampled frame in accepted tracks.

The all-frame CSV is intentionally optional because it is the slowest mode.

## Command-Line Usage

### Activate environment

PowerShell:

```powershell
Set-Location 'C:\Users\moham\OneDrive\Documents\truck_size_2\estimation_size_yolo'
.\.venv\Scripts\Activate.ps1
```

CMD:

```cmd
cd /d C:\Users\moham\OneDrive\Documents\truck_size_2\estimation_size_yolo
.venv\Scripts\activate.bat
```

### Run on one image

```powershell
python .\size_estimation_v4.py 'C:\path\to\image.jpg'
```

### Run on one video

```powershell
python .\auto_select_truck_frames.py 'C:\path\to\video.mp4' --fps 3
```

### Run on many videos in a folder

```powershell
Get-ChildItem 'C:\path\to\videos' -Filter *.mp4 |
ForEach-Object {
    python .\auto_select_truck_frames.py $_.FullName --fps 3
}
```

### Enable summary CSV only

```powershell
python .\auto_select_truck_frames.py 'C:\path\to\video.mp4' --fps 3 --write-summary-csv
```

### Enable full all-frame CSV

```powershell
python .\auto_select_truck_frames.py 'C:\path\to\video.mp4' --fps 3 --write-all-frame-csv
```

### Disable preview window

```powershell
python .\auto_select_truck_frames.py 'C:\path\to\video.mp4' --fps 3 --no-preview
```

## Output Files

Default output folder pattern:

```text
auto_outputs/<video_stem>/
```

Possible outputs:

- selected truck frame images
- `truck_XXX_fill_output.txt`
- `truck_fill_summary.csv` if requested
- `all_frame_fill_levels.csv` if requested

## Important Design Notes

### Why videos are sampled

Running inference on every frame is expensive and unnecessary for fixed-camera scenes. Sampling reduces cost while still preserving enough temporal coverage to find a representative frame.

### Why the best frame is not always the biggest truck frame

The biggest/most centered frame is not always the best fill frame.

Possible problems:

- segmentation may fail on a visually good frame
- box may be detected but content may be missed
- load visibility may be unstable across nearby frames

That is why final selection now considers actual fill-estimation success.

### Why some frames return `0.00%`

This usually means one of these:

- `Box` was detected but `content` was not
- `content` mask was too weak or absent
- the truck box interior was not segmented correctly

### Why some frames fail completely

This usually means:

- no usable truck detection
- no usable `Box` segmentation
- failed frame load

## Known Limitations

- Fill is estimated from 2D vertical occupancy, not real 3D volume.
- The method assumes the segmented `Box` and `content` are meaningful in the current camera angle.
- If the truck bed is occluded, cropped, blurred, or poorly lit, the result may become unstable.
- The video wrapper assumes a fixed-camera, truck-passage workflow.
- The single-frame estimator reports average fill if multiple trucks are present in one frame.

## Recommended Usage Pattern

For image testing:

- use `size_estimation_v4.py`

For production-style video processing:

- use `auto_select_truck_frames.py`
- keep preview on if you want visual monitoring
- use `--fps 3` or `--fps 5` depending on speed/accuracy tradeoff
- request CSVs only when you need deeper analysis

## Low-Level Summary of the Code

At low level, the repository does the following:

- reads pixels with OpenCV
- runs YOLO detection on a frame
- slices the truck crop from the frame array
- runs YOLO segmentation on that crop
- converts masks into binary arrays
- computes fill from row-wise occupancy of binary masks
- uses OpenCV to render overlays and windows
- uses simple track matching based on IoU over sampled frames
- chooses one representative frame using both visual quality and fill-estimation behavior

## Dependencies

Defined in [requirements.txt](/c:/Users/moham/OneDrive/Documents/truck_size_2/estimation_size_yolo/requirements.txt):

- `numpy`
- `opencv-python`
- `torch`
- `ultralytics`

## Current Default Runtime Behavior

- segmentation model defaults to `best_size_March_25.pt`
- video preview is shown by default
- preview window is smaller and resizable
- no CSVs are written unless explicitly requested
- batch video mode prefers faster operation over exhaustive reporting

## Suggested Future Improvements

- add a dedicated logging mode for candidate-frame scores
- add optional saving of the selected final overlay image per truck
- add a pure-library shared inference module so single-image and video code paths share more code directly
- add configurable class names from a config file instead of hardcoded `box` and `content`
- add a confidence-based stability score for segmentation output across adjacent frames

