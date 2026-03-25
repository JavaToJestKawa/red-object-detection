    #!/usr/bin/env python3
import argparse
import math
import sys
import time

import cv2
import numpy as np

RED_LOWER_1 = np.array([161, 155, 84], dtype=np.uint8)
RED_UPPER_1 = np.array([179, 255, 255], dtype=np.uint8)
RED_LOWER_2 = np.array([0, 155, 84], dtype=np.uint8)
RED_UPPER_2 = np.array([5, 255, 255], dtype=np.uint8)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def build_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(frame_hsv, RED_LOWER_1, RED_UPPER_1)
    mask_2 = cv2.inRange(frame_hsv, RED_LOWER_2, RED_UPPER_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=6)

    return red_mask


def detect_object_from_moments(mask: np.ndarray, min_area: float = 300.0):
    moments = cv2.moments(mask, binaryImage=True)

    area_px = moments["m00"]
    if area_px <= 0 or area_px < min_area:
        return None, None, None

    center_x = int(moments["m10"] / area_px)
    center_y = int(moments["m01"] / area_px)
    radius_px = int(math.sqrt(area_px / math.pi))

    return (center_x, center_y), int(area_px), radius_px


def draw_offset_bars(image: np.ndarray, object_x: int, frame_width: int, y: int = 90,
                     bar_thickness: int = 16) -> None:
    frame_center_x = frame_width // 2
    offset = abs(frame_center_x - object_x)

    if object_x > frame_center_x:
        cv2.line(image, (frame_center_x, y), (frame_center_x + offset, y), (0, 0, 255), bar_thickness)
    else:
        cv2.line(image, (frame_center_x - offset, y), (frame_center_x, y), (0, 0, 255), bar_thickness)


def main() -> int:
    parser = argparse.ArgumentParser(description="LAB1: Detekcja czerwonego obiektu")
    parser.add_argument("--video", required=True, help="Ścieżka do pliku wideo")
    parser.add_argument("--min-area", type=float, default=300.0, help="Minimalne pole obiektu [px^2]")
    args = parser.parse_args()

    video_capture = cv2.VideoCapture(args.video)
    if not video_capture.isOpened():
        print(f"ERROR: Cannot open video file: {args.video}", file=sys.stderr)
        return 1

    success, frame = video_capture.read()
    if not success:
        print(f"ERROR: Cannot read first frame from: {args.video}", file=sys.stderr)
        video_capture.release()
        return 1

    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    display_size = (int(frame_width * 3.5 / 5), int(frame_height * 3.5 / 5))

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0
    frame_time = 1.0 / fps

    start_t = time.perf_counter()
    frame_idx = 0

    while success:
        red_mask = build_red_mask(frame)
        object_center, object_area, object_radius = detect_object_from_moments(
            red_mask, min_area=args.min_area
        )

        annotated_frame = frame.copy()
        cv2.line(annotated_frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 255, 100), 1)

        if object_center is not None:
            center_x, center_y = object_center
            cv2.circle(annotated_frame, (center_x, center_y), max(object_radius, 1), (0, 0, 255), 2)
            cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)
            draw_offset_bars(annotated_frame, object_x=center_x, frame_width=frame_width, y=40, bar_thickness=16)

        processed_view = cv2.resize(red_mask, display_size, interpolation=cv2.INTER_NEAREST)
        original_view = cv2.resize(annotated_frame, display_size, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Processed image (HSV mask + morphology)", processed_view)
        cv2.imshow("Original image (tracking + horizontal offset)", original_view)

        frame_idx += 1
        target_t = start_t + frame_idx * frame_time
        remaining = target_t - time.perf_counter()

        key = cv2.waitKey(max(1, int(remaining * 1000))) & 0xFF
        if key in (ord("q"), 27):
            break

        now = time.perf_counter()
        if now > target_t:
            skipped = int((now - target_t) / frame_time)
            if skipped > 0:
                frame_idx += skipped
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        success, frame = video_capture.read()

    video_capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
