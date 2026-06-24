#!/usr/bin/env python3
"""
EvSLAM Evaluation Toolkit
==========================
Evaluate trajectory estimation results against the EvSLAM benchmark.

Usage:
    python evaluate.py --submission <results_dir> --gt <ground_truth_dir>

Directories:
    --submission     Directory containing your estimated trajectory .txt files
                     (one per sequence, e.g. drone_circle_fast.txt).
    --gt             Directory containing ground-truth data:
                     - {seq}_gt.txt        (full GT poses: timestamp tx ty tz qx qy qz qw vx vy vz)
                     - {seq}_ref_time.txt  (reference timestamps for validation)
    --output         Directory for output files (default: ./output).
                     Produces: scores.json, S_xi_result.pdf, and per-seq intermediate files.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

# ============================================================
# Expected evaluation sequences
# ============================================================
EXPECTED_SEQUENCES = [
    "drone_circle_fast",
    "drone_line_fast",
    "drone_8_hdr",
    "drone_s_hdr",
    "drone_rot_norm",
    "drone_rot_fast",
    "slider_hdr",
    "arm_hdr",
    "car_circle_hdr",
]

# ============================================================
#  Ingestion: validate & produce est_ape / gt_ape / auc files
# ============================================================

def is_approx_subset(sub_sorted, ref_sorted, tol):
    """Check every element in sub_sorted has a match in ref_sorted within +-tol."""
    i = 0
    n = len(ref_sorted)
    for t in sub_sorted:
        while i < n and ref_sorted[i] < t - tol:
            i += 1
        if i >= n or abs(ref_sorted[i] - t) > tol:
            return False
    return True


def find_submission_files(submission_dir, gt_dir):
    """Scan submission directory, match against available GT data.
    Returns (valid_pairs, extra, gt_missing) where:
      - valid_pairs: list of (seq_name, sub_path, gt_path, ref_time_path) that can be processed
      - extra: submission files without corresponding GT (ignored)
      - gt_missing: expected sequences not present in submission (informational)
    """
    submitted = sorted(
        f.replace('.txt', '') for f in os.listdir(submission_dir)
        if f.endswith('.txt')
    )
    valid, extra, gt_missing = [], [], []

    for name in submitted:
        gt_path = os.path.join(gt_dir, f"{name}_gt.txt")
        ref_path = os.path.join(gt_dir, f"{name}_ref_time.txt")
        if os.path.exists(gt_path) and os.path.exists(ref_path):
            valid.append((name,
                          os.path.join(submission_dir, f"{name}.txt"),
                          gt_path,
                          ref_path))
        else:
            extra.append(name)

    # Check which expected sequences are missing (from submission)
    expected = set(EXPECTED_SEQUENCES)
    actual = set(submitted)
    gt_available = set(
        f.replace('_gt.txt', '') for f in os.listdir(gt_dir)
        if f.endswith('_gt.txt')
    )
    missing_from_sub = expected - actual
    for name in sorted(missing_from_sub):
        if name in gt_available:
            gt_missing.append(name)

    return valid, extra, gt_missing


def process_sequences(submission_dir, gt_dir, output_dir, time_tolerance=0.001):
    """Read submission files, validate timestamps, produce *_est_ape.txt, *_gt_ape.txt, *_auc.txt.
    Returns (passed, failed, skipped) lists for reporting."""
    os.makedirs(output_dir, exist_ok=True)

    valid_pairs, extra, gt_missing = find_submission_files(submission_dir, gt_dir)

    passed, failed, skipped = [], [], []

    # Report extra entries
    if extra:
        print(f"  Skipping {len(extra)} file(s) with no GT: {extra}")
        skipped.extend(extra)
    if gt_missing:
        print(f"  No submission found for {len(gt_missing)} expected sequence(s): {gt_missing}")

    if not valid_pairs:
        print("  No valid submission files found (all missing GT or no .txt files).")
        return passed, failed, skipped + [name for name, _, _, _ in valid_pairs]

    print(f"  Processing {len(valid_pairs)} sequence(s) with GT data...")

    for name, sub_path, gt_path, ref_path in valid_pairs:
        # ----- Read reference timestamps -----
        ref_ts = np.loadtxt(ref_path)
        ref_ts_sorted = np.sort(ref_ts)

        # ----- Read submission -----
        timestamps, poses, velocities = [], [], []
        with open(sub_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 11:
                    continue
                try:
                    timestamps.append(float(cols[0]))
                    poses.append([float(c) for c in cols[:8]])
                    velocities.append([float(c) for c in cols[8:11]])
                except ValueError:
                    continue

        if not timestamps:
            print(f"  [{name}] SKIPPED — empty or invalid file")
            failed.append((name, "empty or invalid"))
            continue

        # ----- Validate timestamps -----
        sub_sorted = sorted(timestamps)
        is_subset = is_approx_subset(sub_sorted, ref_ts_sorted, time_tolerance)
        enough = len(timestamps) >= len(ref_ts) - 5

        if not (is_subset and enough):
            reason = f"timestamp mismatch (subset={is_subset}, n_sub={len(timestamps)}, n_ref={len(ref_ts)})"
            print(f"  [{name}] FAILED — {reason}")
            failed.append((name, reason))
            continue

        # ----- Write est_ape -----
        est_out = os.path.join(output_dir, f"{name}_est_ape.txt")
        with open(est_out, 'w') as f:
            for pose in poses:
                f.write(' '.join(f"{v:.6f}" for v in pose) + '\n')

        # ----- Match GT timestamps (two-pointer), write gt_ape & auc -----
        gt_data = np.loadtxt(gt_path)
        if gt_data.ndim == 1:
            gt_data = gt_data.reshape(1, -1)

        sub_sort_idx = np.argsort(timestamps)
        sub_ts_sorted = np.array(timestamps)[sub_sort_idx]
        sub_vel_sorted = np.array(velocities)[sub_sort_idx]

        gt_ape_lines, auc_lines = [], []
        i, n = 0, len(sub_ts_sorted)

        for gt_row in gt_data:
            gt_ts = gt_row[0]
            gt_vel = gt_row[8:11]
            while i < n and sub_ts_sorted[i] < gt_ts:
                i += 1

            best_idx, best_dist = None, float('inf')
            if i < n:
                dist = abs(sub_ts_sorted[i] - gt_ts)
                if dist <= time_tolerance and dist < best_dist:
                    best_dist, best_idx = dist, i
            if i > 0:
                dist = abs(sub_ts_sorted[i - 1] - gt_ts)
                if dist <= time_tolerance and dist < best_dist:
                    best_dist, best_idx = dist, i - 1

            if best_idx is not None:
                gt_ape_lines.append(' '.join(f"{v:.6f}" for v in gt_row[:8]))
                sub_vel = sub_vel_sorted[best_idx]
                gt_norm = float(np.linalg.norm(gt_vel))
                diff_norm = float(np.linalg.norm(gt_vel - sub_vel))
                rve = diff_norm / gt_norm if gt_norm != 0 else 0.0
                auc_lines.append(f"{gt_ts:.6f} {rve:.8f} {gt_norm:.8f}")

        # ----- Write gt_ape -----
        gt_out = os.path.join(output_dir, f"{name}_gt_ape.txt")
        with open(gt_out, 'w') as f:
            f.write('\n'.join(gt_ape_lines) + '\n')

        # ----- Write auc -----
        auc_out = os.path.join(output_dir, f"{name}_auc.txt")
        with open(auc_out, 'w') as f:
            f.write('\n'.join(auc_lines) + '\n')

        print(f"  [{name}] OK — {len(gt_ape_lines)} poses, {len(auc_lines)} velocity pairs")
        passed.append(name)

    return passed, failed, skipped


# ============================================================
#  Scoring: ATE + AUC
# ============================================================

def umeyama_alignment(est_trans, gt_trans):
    """Umeyama absolute orientation. Returns (R, t)."""
    mu_est = est_trans.mean(axis=0)
    mu_gt = gt_trans.mean(axis=0)
    H = (est_trans - mu_est).T @ (gt_trans - mu_gt)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = mu_gt - R @ mu_est
    return R, t


def compute_ate(input_dir):
    """Compute ATE RMSE for each sequence using Umeyama alignment. Returns (total, per_seq_dict)."""
    est_files = sorted(glob.glob(os.path.join(input_dir, '*_est_ape.txt')))
    per_sequence = {}
    for est_file in est_files:
        seq_name = os.path.basename(est_file).replace('_est_ape.txt', '')
        gt_file = os.path.join(input_dir, f"{seq_name}_gt_ape.txt")
        if not os.path.exists(gt_file):
            continue
        est_data = np.loadtxt(est_file)
        gt_data = np.loadtxt(gt_file)
        if est_data.ndim == 1:
            est_data = est_data.reshape(1, -1)
        if gt_data.ndim == 1:
            gt_data = gt_data.reshape(1, -1)
        est_trans = est_data[:, 1:4]
        gt_trans = gt_data[:, 1:4]
        R, t = umeyama_alignment(est_trans, gt_trans)
        aligned = (R @ est_trans.T).T + t
        errors = np.linalg.norm(aligned - gt_trans, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        per_sequence[seq_name] = rmse
        print(f"  {seq_name}: ATE = {rmse:.4f}")
    if per_sequence:
        total = sum(per_sequence.values())
        print(f"  Total ATE: {total:.4f}")
        return total, per_sequence
    return None, {}


def compute_auc(input_dir, out_pdf=None):
    """Compute AUC for each sequence from *_auc.txt files. Returns (total, per_seq_dict)."""
    from draw_auc import plot_multi_S_xi_curves
    auc_files = sorted(glob.glob(os.path.join(input_dir, '*_auc.txt')))
    auc_dict = plot_multi_S_xi_curves(auc_files, out_pdf=out_pdf, show=False)
    per_sequence = {k.replace('_auc.txt', ''): v for k, v in auc_dict.items()}
    total = sum(per_sequence.values())
    for seq in sorted(per_sequence.keys()):
        print(f"  {seq}: AUC = {per_sequence[seq]:.6f}")
    print(f"  Total AUC: {total:.6f}")
    return total, per_sequence


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EvSLAM Evaluation Toolkit")
    parser.add_argument('--submission', required=True,
                        help='Directory with estimated trajectory .txt files')
    parser.add_argument('--gt', required=True,
                        help='Directory with ground-truth data (_gt.txt and _ref_time.txt)')
    parser.add_argument('--output', default='./output',
                        help='Output directory (default: ./output)')
    args = parser.parse_args()

    print("=" * 60)
    print("EvSLAM Evaluation Toolkit")
    print("=" * 60)

    # Step 1: Process submissions
    print("\n[1/3] Processing submissions...")
    passed, failed, skipped = process_sequences(args.submission, args.gt, args.output)

    if not passed:
        print("\nNo valid sequences to score. Exiting.")
        return

    # Step 2: Compute ATE
    print("\n[2/3] Computing ATE...")
    ate_total, per_seq_ate = compute_ate(args.output)

    # Step 3: Compute AUC
    print("\n[3/3] Computing AUC...")
    pdf_path = os.path.join(args.output, "S_xi_result.pdf")
    auc_total, per_seq_auc = compute_auc(args.output, out_pdf=pdf_path)

    # Write scores.json
    scores = {
        'ate': ate_total,
        'auc': auc_total,
        'per_sequence_ate': per_seq_ate,
        'per_sequence_auc': per_seq_auc,
    }
    score_file = os.path.join(args.output, 'scores.json')
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Passed:  {len(passed)} sequences")
    if failed:
        print(f"  Failed:  {len(failed)} sequences")
        for name, reason in failed:
            print(f"    - {name}: {reason}")
    if skipped:
        print(f"  Skipped: {len(skipped)} sequences (no GT data)")
        print(f"    {', '.join(skipped)}")
    if ate_total is not None:
        print(f"  ATE:     {ate_total:.4f} m ({len(per_seq_ate)} seqs)")
        print(f"  AUC:     {auc_total:.6f} ({len(per_seq_auc)} seqs)")
    print(f"  Scores:  {score_file}")
    if auc_total:
        print(f"  S_xi:    {pdf_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
