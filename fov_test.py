"""
FOV Test Tool - Basketball CV
Displays live camera feed with a click-to-mark overlay.
Use this to assess what court markings are visible from your camera position.

Controls:
  Click        - Mark a point (up to 8 points)
  R            - Reset all marks
  S            - Save a screenshot to fov_snapshot.jpg
  G            - Toggle grid overlay
  Q / ESC      - Quit
"""

import cv2
import sys

SOURCE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WIDTH, HEIGHT = 1280, 720

marks = []  # list of (x, y, label)
show_grid = False
label_counter = [1]

# FPS tracking
import time
fps_times = []

INSTRUCTIONS = [
    "Click to mark court points",
    "R=reset  S=save  G=grid  Q=quit",
]

COURT_LABELS = [
    "Near-L corner",
    "Near-R corner",
    "Far-L corner",
    "Far-R corner",
    "FT line L",
    "FT line R",
    "3pt arc L",
    "3pt arc R",
]


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = len(marks)
        label = COURT_LABELS[idx] if idx < len(COURT_LABELS) else f"Pt {idx+1}"
        marks.append((x, y, label))
        print(f"  Marked [{label}] at pixel ({x}, {y})")


def draw_grid(frame, step=80):
    h, w = frame.shape[:2]
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)
    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)


def draw_marks(frame):
    for i, (x, y, label) in enumerate(marks):
        color = (0, 255, 0) if i < 4 else (255, 165, 0)
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
        # Label with background
        text = f"{i+1}:{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = x + 12, y - 6
        # Keep label on screen
        if tx + tw > frame.shape[1]: tx = x - tw - 12
        if ty - th < 0: ty = y + th + 6
        cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_hud(frame):
    # Instructions
    for i, line in enumerate(INSTRUCTIONS):
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # FPS
    now = time.monotonic()
    fps_times.append(now)
    cutoff = now - 1.0
    while fps_times and fps_times[0] < cutoff:
        fps_times.pop(0)
    fps = len(fps_times)
    fps_color = (0, 255, 0) if fps >= 50 else (0, 165, 255) if fps >= 30 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps}", (WIDTH - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

    # Mark count
    next_label = COURT_LABELS[len(marks)] if len(marks) < len(COURT_LABELS) else "extra point"
    status = f"Marks: {len(marks)}/8  |  Next: {next_label}" if len(marks) < 8 else f"Marks: {len(marks)} (full)"
    cv2.putText(frame, status, (10, HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)


def main():
    global show_grid

    cap = cv2.VideoCapture(SOURCE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):
        cap.grab()

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {SOURCE}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")
    print("Click on court markings in order:")
    for i, lbl in enumerate(COURT_LABELS):
        print(f"  {i+1}. {lbl}")
    print()

    cv2.namedWindow("FOV Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FOV Test", WIDTH, HEIGHT)
    cv2.setMouseCallback("FOV Test", mouse_cb)

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            print("Frame read failed")
            break

        if show_grid:
            draw_grid(frame)

        draw_marks(frame)
        draw_hud(frame)

        cv2.imshow("FOV Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            marks.clear()
            print("Marks reset.")
        elif key == ord('s'):
            fname = "fov_snapshot.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
        elif key == ord('g'):
            show_grid = not show_grid

    cap.release()
    cv2.destroyAllWindows()

    if marks:
        print("\n=== Marked Points ===")
        for i, (x, y, label) in enumerate(marks):
            print(f"  {i+1}. {label:20s}  pixel ({x:4d}, {y:4d})")
        visible = [lbl for _, _, lbl in marks if "corner" in lbl.lower()]
        near = [lbl for lbl in visible if "near" in lbl.lower()]
        print(f"\n  Near corners visible: {len(near)}/2")
        if len(near) == 2:
            print("  => Single camera MAY cover near corners (verify by calibration)")
        elif len(near) == 1:
            print("  => Only one near corner visible - second camera likely needed")
        else:
            print("  => Near corners NOT visible - second camera needed for shot location")


if __name__ == "__main__":
    main()
