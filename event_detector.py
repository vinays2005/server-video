import numpy as np

def detect_events(curve):

    if len(curve) < 3:
        return []

    vals = [p["value"] for p in curve]

    mean = np.mean(vals)

    std = np.std(vals)

    threshold = mean - 1.5 * std

    events = []

    last_event_time = -5

    for i in range(1, len(curve)-1):

        pt = curve[i]
        prev = curve[i-1]
        nxt = curve[i+1]

        is_local_min = (
            pt["value"] <= prev["value"]
            and
            pt["value"] <= nxt["value"]
        )

        is_below_threshold = pt["value"] < threshold

        far_enough = (
            pt["time"] - last_event_time > 1.5
        )

        if (
            is_local_min
            and
            is_below_threshold
            and
            far_enough
        ):

            drop = prev["value"] - pt["value"]

            if pt["value"] < -0.3:
                event_type = "Scene Change"

            elif drop > 0.45:
                event_type = "Object Entered"

            elif drop > 0.25:
                event_type = "Motion Spike"

            else:
                event_type = "Object Left"

            events.append({
                "id": f"ev-{i}",
                "time": pt["time"],
                "score": pt["value"],
                "type": event_type
            })

            last_event_time = pt["time"]

    return events