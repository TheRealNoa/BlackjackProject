import { useEffect, useMemo, useRef, useState } from "react";
import NavBar from "./components/NavBar";
import { predictCard, predictCards } from "./services/api";

const LOCK_THRESHOLD = 0.9;
const TRACK_TTL_MS = 1800;
const DETECT_INTERVAL_MS = 120;

function App() {
  const [mode, setMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLiveScanning, setIsLiveScanning] = useState(false);
  const [scanIntervalMs, setScanIntervalMs] = useState(1200);
  const [multiCardMode, setMultiCardMode] = useState(true);
  const [opencvReady, setOpencvReady] = useState(false);
  const [result, setResult] = useState(null);
  const [liveCards, setLiveCards] = useState([]);
  const [error, setError] = useState("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const scanTimerRef = useRef(null);
  const detectTimerRef = useRef(null);
  const inFlightRef = useRef(false);
  const tracksRef = useRef([]);
  const nextTrackIdRef = useRef(1);

  const hasFile = useMemo(() => Boolean(file), [file]);

  useEffect(() => {
    return () => {
      if (scanTimerRef.current) {
        clearInterval(scanTimerRef.current);
      }
      if (detectTimerRef.current) {
        clearInterval(detectTimerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    const timer = setInterval(() => {
      const cv = window.cv;
      if (cv && typeof cv.Mat === "function") {
        setOpencvReady(true);
        clearInterval(timer);
      }
    }, 400);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (mode !== "live" && isCameraOn) {
      stopCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const onFileChange = (event) => {
    const picked = event.target.files?.[0];
    if (!picked) return;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setFile(picked);
    setResult(null);
    setError("");
    setPreviewUrl(URL.createObjectURL(picked));
  };

  const toBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const full = String(reader.result || "");
        const payload = full.includes(",") ? full.split(",")[1] : full;
        resolve(payload);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const runPrediction = async (imageBase64, source = "upload", writeResult = true) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    if (source === "upload") {
      setIsLoading(true);
    }
    try {
      const data = await predictCard({ imageBase64, topK });
      if (writeResult) {
        setResult(data);
      }
      setError("");
      return data;
    } catch (err) {
      const message = err?.response?.data?.error || err.message || "Prediction failed.";
      setError(message);
      return null;
    } finally {
      if (source === "upload") {
        setIsLoading(false);
      }
      inFlightRef.current = false;
    }
  };

  const runBatchPrediction = async (imagesBase64) => {
    if (inFlightRef.current) return null;
    inFlightRef.current = true;
    try {
      const data = await predictCards({ imagesBase64, topK: 1 });
      setError("");
      return data;
    } catch (err) {
      const message = err?.response?.data?.error || err.message || "Prediction failed.";
      setError(message);
      return null;
    } finally {
      inFlightRef.current = false;
    }
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
    setResult(null);
    setError("");
    const imageBase64 = await toBase64(file);
    await runPrediction(imageBase64, "upload");
  };

  const resetLiveTrackingState = () => {
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    setLiveCards([]);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsCameraOn(true);
      setError("");
      if (mode === "live") {
        startDetectionLoop();
      }
    } catch (err) {
      setError(`Camera access failed: ${err.message}`);
    }
  };

  const stopCamera = () => {
    stopLiveScan();
    stopDetectionLoop();
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    resetLiveTrackingState();
    setIsLiveScanning(false);
    setIsCameraOn(false);
  };

  const drawCurrentFrameToCanvas = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return "";

    const maxWidth = 640;
    const ratio = maxWidth / video.videoWidth;
    const outWidth = video.videoWidth > maxWidth ? maxWidth : video.videoWidth;
    const outHeight = video.videoWidth > maxWidth ? Math.round(video.videoHeight * ratio) : video.videoHeight;

    canvas.width = outWidth;
    canvas.height = outHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, outWidth, outHeight);
    return canvas;
  };

  const detectCardRects = (canvas) => {
    const cv = window.cv;
    if (!opencvReady || !cv || !canvas) return [];

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    const blur = new cv.Mat();
    const edges = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.Canny(blur, edges, 60, 160);
    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const minArea = canvas.width * canvas.height * 0.02;
    const maxArea = canvas.width * canvas.height * 0.9;
    const candidates = [];

    for (let i = 0; i < contours.size(); i += 1) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < minArea || area > maxArea) {
        cnt.delete();
        continue;
      }

      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const rect = cv.boundingRect(approx);
        const ratio = rect.width / rect.height;
        if (ratio > 0.45 && ratio < 0.95) {
          candidates.push({
            x: rect.x,
            y: rect.y,
            w: rect.width,
            h: rect.height,
            area,
          });
        }
      }
      approx.delete();
      cnt.delete();
    }

    src.delete();
    gray.delete();
    blur.delete();
    edges.delete();
    contours.delete();
    hierarchy.delete();

    candidates.sort((a, b) => b.area - a.area);
    return candidates.slice(0, 4).map((c) => ({
      x: c.x,
      y: c.y,
      w: c.w,
      h: c.h,
      nx: c.x / canvas.width,
      ny: c.y / canvas.height,
      nw: c.w / canvas.width,
      nh: c.h / canvas.height,
    }));
  };

  const calcCenter = (box) => ({
    x: box.nx + box.nw / 2,
    y: box.ny + box.nh / 2,
  });

  const centerDistance = (a, b) => {
    const c1 = calcCenter(a);
    const c2 = calcCenter(b);
    return Math.hypot(c1.x - c2.x, c1.y - c2.y);
  };

  const mergeDetectionsIntoTracks = (detections) => {
    const now = Date.now();
    const existing = tracksRef.current.map((t) => ({ ...t }));
    const used = new Set();

    for (const det of detections) {
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let i = 0; i < existing.length; i += 1) {
        if (used.has(i)) continue;
        const d = centerDistance(existing[i], det);
        if (d < 0.2 && d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) {
        existing[bestIdx] = {
          ...existing[bestIdx],
          ...det,
          lastSeen: now,
        };
        used.add(bestIdx);
      } else {
        existing.push({
          id: nextTrackIdRef.current++,
          ...det,
          label: "",
          probability: 0,
          locked: false,
          endpoint: "",
          lastSeen: now,
        });
      }
    }

    const filtered = existing.filter((t) => now - t.lastSeen <= TRACK_TTL_MS);
    tracksRef.current = filtered;
    setLiveCards(filtered);
  };

  const startDetectionLoop = () => {
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
    }
    detectTimerRef.current = setInterval(() => {
      if (!isCameraOn || mode !== "live" || !multiCardMode || !opencvReady) return;
      const canvas = drawCurrentFrameToCanvas();
      if (!canvas) return;
      const rects = detectCardRects(canvas);
      mergeDetectionsIntoTracks(rects);
    }, DETECT_INTERVAL_MS);
  };

  const stopDetectionLoop = () => {
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
      detectTimerRef.current = null;
    }
  };

  const cropRectToBase64 = (canvas, rect) => {
    const crop = document.createElement("canvas");
    crop.width = rect.w;
    crop.height = rect.h;
    const ctx = crop.getContext("2d");
    ctx.drawImage(canvas, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
    const dataUrl = crop.toDataURL("image/jpeg", 0.92);
    return dataUrl.split(",")[1] ?? "";
  };

  const startLiveScan = () => {
    if (!isCameraOn) {
      setError("Start camera first.");
      return;
    }
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
    }
    setIsLiveScanning(true);
    scanTimerRef.current = setInterval(async () => {
      if (mode === "live" && multiCardMode && opencvReady) {
        const now = Date.now();
        const unresolved = tracksRef.current.filter((t) => now - t.lastSeen < 700 && !t.locked);
        if (unresolved.length > 0) {
          const canvas = drawCurrentFrameToCanvas();
          if (!canvas) return;
          const crops = unresolved.map((rect) => cropRectToBase64(canvas, rect));
          const data = await runBatchPrediction(crops);
          if (data?.predictions) {
            const endpoint = data.endpoint ?? "";
            const updated = tracksRef.current.map((track) => {
              const idx = unresolved.findIndex((u) => u.id === track.id);
              if (idx < 0) return track;
              const top = data.predictions[idx]?.top_prediction;
              if (!top) return track;
              const probability = top.probability ?? 0;
              return {
                ...track,
                label: top.label ?? track.label,
                probability,
                locked: track.locked || probability >= LOCK_THRESHOLD,
                endpoint,
              };
            });
            tracksRef.current = updated;
            setLiveCards(updated);
          }
        }
        setResult(null);
        return;
      }

      const canvas = drawCurrentFrameToCanvas();
      if (!canvas) return;
      const imageBase64 = canvas.toDataURL("image/jpeg", 0.9).split(",")[1] ?? "";
      if (!imageBase64) return;
      resetLiveTrackingState();
      await runPrediction(imageBase64, "live");
    }, Math.max(300, Number(scanIntervalMs) || 1200));
  };

  const stopLiveScan = () => {
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
    setIsLiveScanning(false);
  };

  return (
    <div className="page">
      <div className="container">
        <NavBar />

        <section className="panel">
          <h1>Blackjack Card Classifier</h1>
          <p className="muted">
            Switch between upload and live camera mode. Frames are sent to API Gateway and scored by SageMaker.
          </p>
          <div className="modeToggle">
            <button
              type="button"
              className={mode === "upload" ? "modeBtn activeMode" : "modeBtn"}
              onClick={() => setMode("upload")}
            >
              Upload
            </button>
            <button
              type="button"
              className={mode === "live" ? "modeBtn activeMode" : "modeBtn"}
              onClick={() => setMode("live")}
            >
              Live Camera
            </button>
          </div>

          {mode === "upload" && (
            <form onSubmit={onSubmit} className="form">
              <label className="label">Card image</label>
              <input type="file" accept="image/*" onChange={onFileChange} />

              <label className="label">Top K predictions</label>
              <input
                type="number"
                min={1}
                max={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value || 1))}
              />

              <button type="submit" disabled={!hasFile || isLoading}>
                {isLoading ? "Predicting..." : "Predict Card"}
              </button>
            </form>
          )}

          {mode === "live" && (
            <div className="form">
              <label className="label">Top K predictions</label>
              <input
                type="number"
                min={1}
                max={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value || 1))}
              />

              <label className="label">Scan interval (ms)</label>
              <input
                type="number"
                min={300}
                max={5000}
                value={scanIntervalMs}
                onChange={(e) => setScanIntervalMs(Number(e.target.value || 1200))}
              />
              <label className="checkRow">
                <input
                  type="checkbox"
                  checked={multiCardMode}
                  onChange={(e) => {
                    setMultiCardMode(e.target.checked);
                    resetLiveTrackingState();
                  }}
                />
                Detect multiple cards per frame (OpenCV)
              </label>

              <div className="buttonRow">
                {!isCameraOn ? (
                  <button type="button" onClick={startCamera}>
                    Start Camera
                  </button>
                ) : (
                  <button type="button" onClick={stopCamera}>
                    Stop Camera
                  </button>
                )}

                {!isLiveScanning ? (
                  <button type="button" disabled={!isCameraOn} onClick={startLiveScan}>
                    Start Live Scan
                  </button>
                ) : (
                  <button type="button" onClick={stopLiveScan}>
                    Stop Live Scan
                  </button>
                )}
              </div>

              <p className="muted small">
                Camera: {isCameraOn ? "On" : "Off"} | Live scan: {isLiveScanning ? "Running" : "Stopped"}
              </p>
              <p className="muted small">OpenCV detector: {opencvReady ? "Ready" : "Loading..."}</p>
            </div>
          )}
        </section>

        <section className="grid">
          <article className="panel">
            <h2>{mode === "live" ? "Live Preview" : "Image Preview"}</h2>
            {mode === "upload" && !previewUrl && <p className="muted">No image selected.</p>}
            {mode === "upload" && previewUrl && <img className="preview" src={previewUrl} alt="Uploaded card preview" />}
            {mode === "live" && (
              <div className="liveContainer">
                <video ref={videoRef} className="preview livePreview" autoPlay playsInline muted />
                <canvas ref={canvasRef} className="hiddenCanvas" />
                {liveCards.map((card) => (
                  <div
                    key={`box-${card.id}`}
                    className={card.locked ? "cardBox lockedCardBox" : "cardBox"}
                    style={{
                      left: `${card.nx * 100}%`,
                      top: `${card.ny * 100}%`,
                      width: `${card.nw * 100}%`,
                      height: `${card.nh * 100}%`,
                    }}
                  >
                    <span className="cardLabel">
                      {card.label ? `${card.label} ${(card.probability * 100).toFixed(0)}%` : "Scanning..."}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </article>

          <article className="panel">
            <h2>Predictions</h2>
            {error && <p className="error">{error}</p>}
            {!error && !result && liveCards.length === 0 && <p className="muted">No prediction yet.</p>}
            {!error && liveCards.length > 0 && (
              <div>
                <p>
                  <strong>Tracked cards:</strong> {liveCards.length}
                </p>
                <ul>
                  {liveCards.map((card, idx) => (
                    <li key={`${card.id}-${card.label || "scan"}`}>
                      Card {idx + 1}: <strong>{card.label || "Scanning..."}</strong>
                      {card.label ? ` (${(card.probability * 100).toFixed(2)}%)` : ""}
                      {card.locked ? " [locked]" : ""}
                    </li>
                  ))}
                </ul>
                {liveCards[0]?.endpoint && (
                  <p className="muted small">
                    Endpoint: <code>{liveCards[0].endpoint}</code>
                  </p>
                )}
              </div>
            )}
            {result?.predictions?.[0] && (
              <div>
                <p>
                  <strong>Top:</strong> {result.predictions[0].top_prediction.label} (
                  {(result.predictions[0].top_prediction.probability * 100).toFixed(2)}%)
                </p>
                {result.endpoint && (
                  <p className="muted small">
                    Endpoint: <code>{result.endpoint}</code>
                  </p>
                )}
              </div>
            )}
          </article>
        </section>
      </div>
    </div>
  );
}

export default App;
