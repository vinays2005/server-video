from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import os
import shutil

from skimage.metrics import structural_similarity as ssim

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "backend": "FastAPI"
    }


def compare_frames(frame1, frame2):

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    score, _ = ssim(
        gray1,
        gray2,
        full=True
    )

    return float(score)


def detect_events(curve):

    if len(curve) < 3:
        return []

    values = [x["value"] for x in curve]

    mean = np.mean(values)

    std = np.std(values)

    threshold = mean - 1.5 * std

    events = []

    last_event = -5

    for i in range(1, len(curve)-1):

        cur = curve[i]
        prev = curve[i-1]
        nxt = curve[i+1]

        is_local_min = (
            cur["value"] <= prev["value"]
            and
            cur["value"] <= nxt["value"]
        )

        if (
            is_local_min
            and
            cur["value"] < threshold
            and
            cur["time"] - last_event > 1.5
        ):

            drop = prev["value"] - cur["value"]

            if cur["value"] < 0.30:
                event_type = "Scene Change"

            elif drop > 0.45:
                event_type = "Object Entered"

            elif drop > 0.25:
                event_type = "Motion Spike"

            else:
                event_type = "Object Left"

            events.append({
                "id": f"ev-{i}",
                "time": cur["time"],
                "score": round(cur["value"],4),
                "type": event_type
            })

            last_event = cur["time"]

    return events


@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(...)
):

    filepath = os.path.join(
        UPLOAD_DIR,
        video.filename
    )

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(
            video.file,
            buffer
        )

    cap = cv2.VideoCapture(filepath)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_interval = max(
        1,
        int(fps)
    )

    curve = []

    previous_frame = None

    frame_count = 0

    time_sec = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % frame_interval == 0:

            frame = cv2.resize(
                frame,
                (320,180)
            )

            if previous_frame is None:

                curve.append({
                    "time": time_sec,
                    "value": 1.0
                })

            else:

                similarity = compare_frames(
                    previous_frame,
                    frame
                )

                curve.append({
                    "time": time_sec,
                    "value": round(
                        similarity,
                        5
                    )
                })

            previous_frame = frame

            time_sec += 1

        frame_count += 1

    cap.release()

    events = detect_events(
        curve
    )

    values = [
        p["value"]
        for p in curve
    ]

    return {
        "curve": curve,
        "events": events,
        "stats": {
            "mean": round(float(np.mean(values)),4),
            "std": round(float(np.std(values)),4),
            "min": round(float(min(values)),4),
            "max": round(float(max(values)),4),
            "sampleCount": len(curve),
            "eventCount": len(events)
        }
    }