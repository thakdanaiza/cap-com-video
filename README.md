# cap-com-video v1

Runnable release bundle for video review, CRF quality testing, and image/video sharpness comparison in the `non-flip` / `flip` workflow.

Default release path:

```text
/home/cra-space-center/Desktop/real-6-day-26-30-copy/cap-com-video/
```

## Release Contents

```text
cap-com-video/
  README.md
  README.tex
  README.pdf
  requirements.txt
  environment.yml
  video_review_center.py
  test_video_crop_calibrator_crf.py
  compare_image_sharpness.py
  build_clip_edge_frames.py
  build_long_video_from_clip_ranges.py
  crop_calibration_space_24.json
```

- `video_review_center.py` starts a local browser UI for reviewing processed `flip` videos, optional `non-flip` sources, event/session data, long-video jobs, and edge-frame jobs.
- `test_video_crop_calibrator_crf.py` renders one calibrated input video at multiple CRF values so you can choose the best size/quality balance.
- `compare_image_sharpness.py` compares image or video inputs and writes side-by-side preview images plus a JSON metrics report.
- `build_clip_edge_frames.py` and `build_long_video_from_clip_ranges.py` are runtime helpers imported by `video_review_center.py`.
- `crop_calibration_space_24.json` is a sample calibration profile for CRF testing.

## System Requirements

Install system packages:

```bash
sudo apt update
sudo apt install -y python3 ffmpeg
```

`ffmpeg` and `ffprobe` must be available in `PATH`. They are used for rendering, probing duration/size, building long videos, and extracting frames.

## Conda Setup

Create and activate the conda environment:

```bash
cd /home/cra-space-center/Desktop/real-6-day-26-30-copy/cap-com-video
conda env create -f environment.yml
conda activate cap-com-video
```

If the environment already exists, update it:

```bash
conda env update -n cap-com-video -f environment.yml --prune
conda activate cap-com-video
```

Alternative pip install inside an existing conda environment:

```bash
conda activate vdobrowser
python -m pip install -r requirements.txt
```

Check external tools:

```bash
ffmpeg -version
ffprobe -version
```

## 1. Start Video Review Center

Run the local web server:

```bash
cd /home/cra-space-center/Desktop/real-6-day-26-30-copy/cap-com-video
conda activate cap-com-video
python video_review_center.py \
  --flip /home/cra-space-center/Desktop/real-6-day-26-30-copy/flip \
  --non-flip /home/cra-space-center/Desktop/real-6-day-26-30-copy/non-flip \
  --host 127.0.0.1 \
  --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Useful pages:

```text
http://127.0.0.1:8765/                 video review UI
http://127.0.0.1:8765/events           event/session map
http://127.0.0.1:8765/events-export.html
http://127.0.0.1:8765/long-builder.html
http://127.0.0.1:8765/edge-frame-builder.html
```

Important options:

```bash
python video_review_center.py --help
```

- `--flip`: root folder for processed videos.
- `--non-flip`: optional root folder for original videos.
- `--command-csv`: command export CSV/TSV for event/session data.
- `--remote-video-csv`: remote video list CSV for matching remote clips.
- `--event-notes`: JSON file used to save event group names and notes.
- `--refresh-seconds`: auto-refresh interval for the video index when API requests arrive. Use `0` to disable.

Stop the server with `Ctrl+C`.

## 2. Test CRF Quality

Use this tool to choose a CRF value that keeps enough image detail without producing unnecessarily large files:

```bash
cd /home/cra-space-center/Desktop/real-6-day-26-30-copy/cap-com-video
conda activate cap-com-video
python test_video_crop_calibrator_crf.py /path/to/input.mp4 \
  --config ./crop_calibration_space_24.json \
  --output-dir ./crf_test \
  --crf-values 12 16 18 20 23 \
  --max-seconds 15 \
  --sample-seconds 5 \
  --overwrite
```

CRF guide:

```text
Lower CRF = higher quality, sharper detail, larger files
Higher CRF = stronger compression, smaller files, more softness/artifacts
CRF 12-16 = high quality for detail inspection
CRF 18-20 = good starting range for v1 balancing
CRF 23+   = smaller files, but visual checking is required
```

Outputs:

```text
crf_test/
  input_stem_crf12.mp4
  input_stem_crf12_sample.png
  input_stem_crf16.mp4
  input_stem_crf16_sample.png
  ...
  summary.json
  note.txt
```

- `summary.json` stores output path, duration, size, and bitrate for each CRF.
- `note.txt` records the input, config, preset, max seconds, sample seconds, and CRF guide.
- `*_sample.png` files are quick side-by-side inspection frames for texture, edges, text, fine detail, and noise.

## 3. Compare Sharpness

Compare PNG frames from a CRF test:

```bash
cd /home/cra-space-center/Desktop/real-6-day-26-30-copy/cap-com-video
conda activate cap-com-video
python compare_image_sharpness.py \
  --input ./crf_test/input_stem_crf16_sample.png \
  --input ./crf_test/input_stem_crf18_sample.png \
  --label CRF16 \
  --label CRF18 \
  --output-dir ./sharpness_compare
```

Compare frames directly from videos:

```bash
python compare_image_sharpness.py \
  --input /path/to/raw.mp4 \
  --input /path/to/calibrated.mp4 \
  --label raw \
  --label calibrated \
  --time-seconds 10 \
  --output-dir ./sharpness_compare
```

Compare a specific region of interest:

```bash
python compare_image_sharpness.py \
  --input ./crf_test/input_stem_crf16_sample.png \
  --input ./crf_test/input_stem_crf20_sample.png \
  --label CRF16 \
  --label CRF20 \
  --crop 100,80,400,240 \
  --output-dir ./sharpness_compare_roi
```

Outputs:

```text
sharpness_compare/
  comparison_strip.jpg
  comparison_zoom_strip.jpg
  report.json
```

- `comparison_strip.jpg` is a side-by-side preview with key metrics.
- `comparison_zoom_strip.jpg` is a centered zoom crop for detail/noise inspection.
- `report.json` includes metrics such as `laplacian_var`, `tenengrad`, `brenner`, `mean_gradient`, `noise_std`, `psnr`, and `ssim`.

Metrics are screening tools, not final visual judgment. Noisy footage can score as sharp even when it does not look better.

## v1 Release Checklist

Before marking this bundle as v1-ready:

```bash
cd /home/cra-space-center/Desktop/real-6-day-26-30-copy/cap-com-video
conda activate cap-com-video
python -m py_compile \
  video_review_center.py \
  test_video_crop_calibrator_crf.py \
  compare_image_sharpness.py \
  build_clip_edge_frames.py \
  build_long_video_from_clip_ranges.py

python video_review_center.py --help
python test_video_crop_calibrator_crf.py --help
python compare_image_sharpness.py --help
```

Validation with real media:

```text
1. Video Review Center starts and indexes more than 0 videos.
2. http://127.0.0.1:8765/ opens and can play videos.
3. CRF testing completes on a short sample video and writes summary.json/note.txt.
4. Sharpness comparison completes with at least two PNG or MP4 inputs.
5. comparison_strip.jpg and comparison_zoom_strip.jpg open correctly and labels are readable.
```

If the review UI opens but event/session data is empty, verify `--command-csv` and `--remote-video-csv`.

