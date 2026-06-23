import express from "express";
import cors from "cors";
import sharp from "sharp";

const app = express();
app.use(cors());
// Allow large payloads — frames come in as base64 batches
app.use(express.json({ limit: "50mb" }));

const PORT = 3001;

// ─── Pixel similarity ─────────────────────────────────────────────────────────
// Receives two grayscale Uint8Array buffers of equal length.
// Returns a value in [-1, 1]: 1 = identical, decreasing with divergence.
function computeSimilarity(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  const meanDiff = sum / (a.length * 255);
  // Amplify to fill the range — typical video changes produce 0.02–0.25 meanDiff
  return Math.max(-1, Math.min(1, 1 - meanDiff * 10));
}

// ─── Event detection ──────────────────────────────────────────────────────────
function detectEvents(curve) {
  if (curve.length < 3) return [];

  const vals = curve.map((p) => p.value);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length;
  const std = Math.sqrt(variance);
  const threshold = mean - 1.5 * std;

  const events = [];
  let lastEventTime = -5;

  for (let i = 1; i < curve.length - 1; i++) {
    const pt = curve[i];
    const prev = curve[i - 1];
    const next = curve[i + 1];

    const isLocalMin = pt.value <= prev.value && pt.value <= next.value;
    const isBelowThreshold = pt.value < threshold;
    const farEnoughFromLast = pt.time - lastEventTime > 1.5;

    if (isLocalMin && isBelowThreshold && farEnoughFromLast) {
      const drop = prev.value - pt.value;
      let type;
      if (pt.value < -0.3) type = "Scene Change";
      else if (drop > 0.45) type = "Object Entered";
      else if (drop > 0.25) type = "Motion Spike";
      else type = "Object Left";

      events.push({
        id: `ev-${i}`,
        time: pt.time,
        score: pt.value,
        type,
      });
      lastEventTime = pt.time;
    }
  }

  return events;
}

// ─── POST /api/analyze ────────────────────────────────────────────────────────
// Body: { frames: [{ time: number, data: string }] }
//   where data is a base64-encoded JPEG of the frame (small, e.g. 128×72)
// Response: { curve: SimilarityPoint[], events: ChangeEvent[], stats: {...} }
app.post("/api/analyze", async (req, res) => {
  const { frames } = req.body;

  if (!Array.isArray(frames) || frames.length < 2) {
    return res.status(400).json({ error: "Need at least 2 frames." });
  }

  console.log(`[analyze] received ${frames.length} frames`);

  const curve = [];
  let prevGray = null;

  for (let i = 0; i < frames.length; i++) {
    const { time, data } = frames[i];

    try {
      // Decode base64 JPEG → raw grayscale pixels via sharp
      const buf = Buffer.from(data, "base64");
      const { data: pixels, info } = await sharp(buf)
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

      if (prevGray) {
        const sim = computeSimilarity(prevGray, pixels);
        curve.push({ time, value: parseFloat(sim.toFixed(5)) });
      } else {
        curve.push({ time, value: 1 });
      }

      prevGray = pixels;
    } catch (err) {
      console.error(`[analyze] frame ${i} at t=${time} failed:`, err.message);
      // Keep the previous similarity value on error
      if (curve.length > 0) {
        curve.push({ time, value: curve[curve.length - 1].value });
      } else {
        curve.push({ time, value: 1 });
      }
    }
  }

  const events = detectEvents(curve);

  // Summary stats
  const vals = curve.map((p) => p.value);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const std = Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / vals.length);
  const min = Math.min(...vals);
  const max = Math.max(...vals);

  console.log(`[analyze] done — ${curve.length} samples, ${events.length} events`);

  res.json({
    curve,
    events,
    stats: {
      mean: parseFloat(mean.toFixed(4)),
      std: parseFloat(std.toFixed(4)),
      min: parseFloat(min.toFixed(4)),
      max: parseFloat(max.toFixed(4)),
      sampleCount: curve.length,
      eventCount: events.length,
    },
  });
});

// ─── GET /api/health ──────────────────────────────────────────────────────────
app.get("/api/health", (_req, res) => {
  res.json({ ok: true, sharp: sharp.versions.sharp, vips: sharp.versions.vips });
});

app.listen(PORT, () => {
  console.log(`FrameAI backend running on http://localhost:${PORT}`);
});
