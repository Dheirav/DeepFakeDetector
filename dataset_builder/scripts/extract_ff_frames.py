#!/usr/bin/env python3
"""
Extract frames from FaceForensics++ downloaded videos into
data_sources/ai_edited/FaceForensics/ for the dataset pipeline.

Usage (after downloading videos):
    python3 dataset_builder/scripts/extract_ff_frames.py \
        --video-dir /tmp/ff_videos \
        --out-dir data_sources/ai_edited/FaceForensics \
        --frames-per-video 20

The script scans all .mp4 files under --video-dir (recursively),
extracts --frames-per-video evenly-spaced frames from each, and
saves them as JPEGs into --out-dir with a flat naming scheme.
"""
import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm


# Only extract from manipulated sequences, not originals
MANIPULATED_KEYWORDS = [
    'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures',
    'DeepFakeDetection'
]


def is_manipulated(path: Path) -> bool:
    return any(kw in str(path) for kw in MANIPULATED_KEYWORDS)


def extract_frames(video_path: Path, out_dir: Path, frames_per_video: int, counter: list):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    # Pick evenly spaced frame indices
    if total <= frames_per_video:
        indices = list(range(total))
    else:
        step = total / frames_per_video
        indices = [int(i * step) for i in range(frames_per_video)]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w < 256 or h < 256:
            continue
        fname = f"ff_{counter[0]:07d}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        counter[0] += 1
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from FF++ videos")
    parser.add_argument('--video-dir', required=True,
                        help='Root directory containing downloaded FF++ .mp4 files')
    parser.add_argument('--out-dir', default='data_sources/ai_edited/FaceForensics',
                        help='Output directory for extracted frames')
    parser.add_argument('--frames-per-video', type=int, default=20,
                        help='Number of frames to extract per video (default: 20)')
    parser.add_argument('--all-sequences', action='store_true',
                        help='Also extract from original (real) sequences, not just manipulated')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all .mp4 files
    all_videos = sorted(video_dir.rglob('*.mp4'))
    if not args.all_sequences:
        all_videos = [v for v in all_videos if is_manipulated(v)]

    if not all_videos:
        print(f"No .mp4 files found under {video_dir}")
        print("Make sure you ran the download script first.")
        return

    print(f"Found {len(all_videos)} videos")
    print(f"Extracting {args.frames_per_video} frames per video")
    print(f"Saving to: {out_dir}")
    print()

    # Count existing frames to resume without overwriting
    existing = len(list(out_dir.glob('ff_*.jpg')))
    counter = [existing]
    if existing > 0:
        print(f"Resuming: {existing} frames already exist, continuing from ff_{existing:07d}.jpg")

    total_saved = 0
    for video_path in tqdm(all_videos, desc="Extracting frames"):
        n = extract_frames(video_path, out_dir, args.frames_per_video, counter)
        total_saved += n

    print(f"\nDone. Extracted {total_saved} new frames.")
    print(f"Total frames in {out_dir}: {counter[0]}")


if __name__ == '__main__':
    main()
