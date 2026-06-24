# EvSLAM Dataset

Event-based SLAM Benchmark for High-Speed Maneuvers. This dataset provides multiple sequences captured by a stereo event camera (DVXplorer) equipped with IMU on various mobile robotic platforms (drone, slider, robotic arm, and mecanum-wheeled robot), covering diverse challenging motion patterns and extreme lighting conditions.

Synchronous stereo grayscale and monocular color images with consistent resolution are also provided, supporting event-image fusion research and comparison with frame-based methods.

## Website

[https://nail-hnu.github.io/EvSLAM_Dataset/](https://nail-hnu.github.io/EvSLAM_Dataset/)

## Benchmark

[https://www.codabench.org/competitions/17239/](https://www.codabench.org/competitions/17239/)

## Evaluation Toolkit

The `evaluation/` directory provides a standalone toolkit for evaluating trajectory estimation results against the EvSLAM benchmark. It computes Absolute Trajectory Error (ATE) and the speed-weighted Area Under the Curve (AUC) metric.

### Prerequisites

```bash
pip install numpy matplotlib
```

### Quick Start

```bash
python3 evaluation/evaluate.py \
    --submission ./your_results \
    --gt evaluation/ground_truth \
    --output ./evaluation_output
```

### Command-Line Arguments

| Argument | Description |
|---|---|
| `--submission` | Directory containing your estimated trajectory files (one `.txt` per sequence) |
| `--gt` | Directory containing ground-truth data (`{seq}_gt.txt` and `{seq}_ref_time.txt`) |
| `--output` | Directory for output files (default: `./output`) |

### Submission File Format

Each file must be named `<sequence_name>.txt`. The 9 benchmark sequences are:

`drone_circle_fast` `drone_line_fast` `drone_8_hdr` `drone_s_hdr` `drone_rot_norm` `drone_rot_fast` `slider_hdr` `arm_hdr` `car_circle_hdr`

Each line in the file contains 11 space-separated values:

```
timestamp tx ty tz qx qy qz qw vx vy vz
```

| Field | Description |
|---|---|
| `timestamp` | Pose timestamp (seconds) |
| `tx ty tz` | 3D position in the world coordinate frame (meters) |
| `qx qy qz qw` | Orientation as a unit quaternion (world coordinate frame) |
| `vx vy vz` | Linear velocity in the camera-centric coordinate frame (m/s) |

Timestamps must match the reference timestamps in `{seq}_ref_time.txt` within a tolerance of 0.001 seconds.

### Output Files

| File | Description |
|---|---|
| `scores.json` | Overall and per-sequence ATE and AUC scores |
| `S_xi_result.pdf` | Speed-weighted success-rate curves |
| `{seq}_est_ape.txt` | Extracted pose data from your submission |
| `{seq}_gt_ape.txt` | Matched ground-truth poses |
| `{seq}_auc.txt` | Relative velocity error (RVE) data for AUC computation |

### Using draw_auc.py Standalone

You can also use the AUC plotting module directly:

```python
from draw_auc import plot_multi_S_xi_curves

# datafiles: list of paths to *_auc.txt files (format: ts rve v_norm)
auc_dict = plot_multi_S_xi_curves(
    datafiles=["arm_hdr_auc.txt", "drone_circle_fast_auc.txt", ...],
    out_pdf="S_xi_result.pdf",
    curve_labels=["arm_hdr", "drone_circle_fast", ...],
)
```

Each `*_auc.txt` file must contain 3 space-separated columns per line: `timestamp rve v_gt_norm`, where `rve = ||v_gt - v_est|| / ||v_gt||`.

### Computing AUC for Other Datasets

To evaluate results on your own dataset, you need to provide:

1. **Ground-truth poses** — `{seq}_gt.txt`, each line formatted as:
   ```
   timestamp tx ty tz qx qy qz qw vx vy vz
   ```

2. **Reference timestamps** — `{seq}_ref_time.txt`, one timestamp per line:
   ```
   1.403636580
   1.470303247
   ...
   ```

3. **Estimated trajectories** — `{seq}.txt`, same format as ground truth.

Place all three per sequence in their respective directories, then run:
```bash
python3 evaluation/evaluate.py --submission <est_dir> --gt <gt_dir> --output <out_dir>
```

The script will validate timestamps against the reference, match estimated poses to GT, compute ATE via Umeyama alignment, and produce the speed-weighted AUC curves.
