import { useEffect, useMemo, useRef, useState } from "react";
import NavBar from "./components/NavBar";
import { predictCard, predictCards, predictPipeline } from "./services/api";

const USE_PIPELINE = Boolean((import.meta.env.VITE_PIPELINE_PATH ?? "").trim());

const LOCK_THRESHOLD = 0.9;
const TRACK_TTL_MS = 3500;
const DETECT_INTERVAL_MS = 120;
const CARD_WARP_WIDTH = 200;
const CARD_WARP_HEIGHT = 300;
const CARD_RATIO_MIN = 1.2;
const CARD_RATIO_MAX = 2.0;
const CARD_MIN_EXTENT = 0.65;
const CARD_MIN_AREA_FRAC = 0.003;
const CARD_MAX_AREA_FRAC = 0.55;
const CANNY_LOW = 30;
const CANNY_HIGH = 100;
const NMS_IOU_THRESHOLD = 0.4;
const TRACK_IOU_MATCH = 0.25;
const TRACK_CENTER_MATCH = 0.16;
const TRACK_SMOOTHING_ALPHA = 0.35;
const MIN_STABLE_HITS = 2;
const MAX_MISSED_DETECTIONS = 10;
const ACTIVE_TRACK_WINDOW_MS = 1500;
const STICKY_CONFIRMED_TRACKS = true;
const STICKY_MAX_IDLE_MS = 12000;

// Client-side OpenCV multi-card path (contours + tracking). Off while testing the new model, might change later.
const ENABLE_OPENCV_MULTICARD = false;

function App() {
  const [mode, setMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLiveScanning, setIsLiveScanning] = useState(false);
  const [scanIntervalMs, setScanIntervalMs] = useState(1200);
  const [detectConf, setDetectConf] = useState(0.1);
  const [multiCardMode, setMultiCardMode] = useState(ENABLE_OPENCV_MULTICARD);
  const [opencvReady, setOpencvReady] = useState(false);
  const [result, setResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineFrameSize, setPipelineFrameSize] = useState(null);
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
  const modeRef = useRef(mode);
  const multiCardModeRef = useRef(multiCardMode);
  const opencvReadyRef = useRef(opencvReady);

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
    if (!ENABLE_OPENCV_MULTICARD) {
      setMultiCardMode(false);
    }
  }, []);

  useEffect(() => {
    if (!ENABLE_OPENCV_MULTICARD) {
      setOpencvReady(false);
      return undefined;
    }
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
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    multiCardModeRef.current = multiCardMode;
  }, [multiCardMode]);

  useEffect(() => {
    opencvReadyRef.current = opencvReady;
  }, [opencvReady]);

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
    setPipelineResult(null);
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

  const runPipelinePrediction = async (imageBase64, source = "upload", frameSize = null) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    if (source === "upload") {
      setIsLoading(true);
    }
    try {
      const data = await predictPipeline({ imageBase64, topK, detectConf });
      setPipelineResult(data);
      setPipelineFrameSize(frameSize);
      setResult(null);
      setError("");
      return data;
    } catch (err) {
      const message = err?.response?.data?.error || err.message || "Pipeline prediction failed.";
      setError(message);
      return null;
    } finally {
      if (source === "upload") {
        setIsLoading(false);
      }
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
    setPipelineResult(null);
    setPipelineFrameSize(null);
    setError("");
    const imageBase64 = await toBase64(file);
    if (USE_PIPELINE) {
      const img = new Image();
      const size = await new Promise((resolve) => {
        img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
        img.onerror = () => resolve(null);
        img.src = `data:image/*;base64,${imageBase64}`;
      });
      await runPipelinePrediction(imageBase64, "upload", size);
    } else {
      await runPrediction(imageBase64, "upload");
    }
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
      if (mode === "live" && ENABLE_OPENCV_MULTICARD) {
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
    setResult(null);
    setPipelineResult(null);
    setPipelineFrameSize(null);
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

  const rectIoU = (a, b) => {
    const ax2 = a.x + a.w;
    const ay2 = a.y + a.h;
    const bx2 = b.x + b.w;
    const by2 = b.y + b.h;
    const interW = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
    const interH = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
    const inter = interW * interH;
    if (inter <= 0) return 0;
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
  };

  const rectCenterDistance = (a, b) => {
    const acx = a.x + a.w / 2;
    const acy = a.y + a.h / 2;
    const bcx = b.x + b.w / 2;
    const bcy = b.y + b.h / 2;
    return Math.hypot(acx - bcx, acy - bcy);
  };

  const orderQuadCorners = (pts) => {
    if (!pts || pts.length !== 4) return null;
    const sum = pts.map((p) => p.x + p.y);
    const diff = pts.map((p) => p.y - p.x);
    const tl = pts[sum.indexOf(Math.min(...sum))];
    const br = pts[sum.indexOf(Math.max(...sum))];
    const tr = pts[diff.indexOf(Math.min(...diff))];
    const bl = pts[diff.indexOf(Math.max(...diff))];
    if (!tl || !tr || !bl || !br) return null;
    const topEdge = Math.hypot(tr.x - tl.x, tr.y - tl.y);
    const leftEdge = Math.hypot(bl.x - tl.x, bl.y - tl.y);
    if (topEdge > leftEdge) {
      return [tr, tl, bl, br];
    }
    return [tl, bl, br, tr];
  };

  const rotatedRectCorners = (cv, rotatedRect) => {
    const boxPts = new cv.Mat();
    cv.boxPoints(rotatedRect, boxPts);
    const pts = [];
    for (let i = 0; i < 4; i += 1) {
      pts.push({
        x: boxPts.data32F[i * 2],
        y: boxPts.data32F[i * 2 + 1],
      });
    }
    boxPts.delete();
    return pts;
  };

  const detectCardRects = (canvas) => {
    if (!ENABLE_OPENCV_MULTICARD) return [];
    const cv = window.cv;
    if (!opencvReady || !cv || !canvas) return [];

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    const blur = new cv.Mat();
    const edges = new cv.Mat();
    const adaptive = new cv.Mat();
    const mask = new cv.Mat();
    const closed = new cv.Mat();
    const closeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7));
    const openKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    cv.Canny(blur, edges, CANNY_LOW, CANNY_HIGH);
    cv.adaptiveThreshold(
      blur,
      adaptive,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY_INV,
      25,
      8
    );
    cv.bitwise_or(edges, adaptive, mask);
    cv.morphologyEx(mask, closed, cv.MORPH_CLOSE, closeKernel);
    cv.morphologyEx(closed, closed, cv.MORPH_OPEN, openKernel);
    cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);

    const frameArea = canvas.width * canvas.height;
    const minArea = frameArea * CARD_MIN_AREA_FRAC;
    const maxArea = frameArea * CARD_MAX_AREA_FRAC;
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

      let quadCorners = null;
      let quadArea = 0;
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const quadPts = [];
        for (let j = 0; j < 4; j += 1) {
          const p = approx.intPtr(j, 0);
          quadPts.push({ x: p[0], y: p[1] });
        }
        quadCorners = orderQuadCorners(quadPts);
        quadArea = cv.contourArea(approx);
      }
      approx.delete();

      const rotated = cv.minAreaRect(cnt);
      const rotW = rotated.size.width;
      const rotH = rotated.size.height;
      const shortSide = Math.min(rotW, rotH);
      const longSide = Math.max(rotW, rotH);
      if (shortSide < 18 || longSide < 26) {
        cnt.delete();
        continue;
      }

      const ratio = longSide / shortSide;
      if (ratio < CARD_RATIO_MIN || ratio > CARD_RATIO_MAX) {
        cnt.delete();
        continue;
      }

      const rotatedArea = rotW * rotH;
      const extent = area / Math.max(rotatedArea, 1);
      if (extent < CARD_MIN_EXTENT) {
        cnt.delete();
        continue;
      }

      let corners = quadCorners;
      if (!corners) {
        const rotPts = rotatedRectCorners(cv, rotated);
        corners = orderQuadCorners(rotPts);
      }
      if (!corners) {
        cnt.delete();
        continue;
      }

      const aabb = cv.boundingRect(cnt);
      const effectiveArea = quadCorners ? Math.max(area, quadArea) : area;
      candidates.push({
        x: aabb.x,
        y: aabb.y,
        w: aabb.width,
        h: aabb.height,
        corners,
        area: effectiveArea,
        extent,
        ratio,
        score: effectiveArea * extent,
      });

      cnt.delete();
    }

    src.delete();
    gray.delete();
    blur.delete();
    edges.delete();
    adaptive.delete();
    mask.delete();
    closed.delete();
    closeKernel.delete();
    openKernel.delete();
    contours.delete();
    hierarchy.delete();

    candidates.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const c of candidates) {
      const overlaps = kept.some((k) => rectIoU(k, c) > NMS_IOU_THRESHOLD);
      if (!overlaps) kept.push(c);
      if (kept.length >= 6) break;
    }

    return kept.map((c) => ({
      x: c.x,
      y: c.y,
      w: c.w,
      h: c.h,
      corners: c.corners,
      nx: c.x / canvas.width,
      ny: c.y / canvas.height,
      nw: c.w / canvas.width,
      nh: c.h / canvas.height,
    }));
  };

  const mergeDetectionsIntoTracks = (detections) => {
    const now = Date.now();
    const existing = tracksRef.current.map((t) => ({
      ...t,
      hitCount: t.hitCount ?? 0,
      missCount: t.missCount ?? 0,
      stable: t.stable ?? false,
      sticky: t.sticky ?? false,
    }));
    const used = new Set();

    for (const det of detections) {
      let bestIdx = -1;
      let bestIoU = 0;
      for (let i = 0; i < existing.length; i += 1) {
        if (used.has(i)) continue;
        const prevBox = { x: existing[i].nx, y: existing[i].ny, w: existing[i].nw, h: existing[i].nh };
        const newBox = { x: det.nx, y: det.ny, w: det.nw, h: det.nh };
        const iou = rectIoU(prevBox, newBox);
        const centerDist = rectCenterDistance(prevBox, newBox);
        const isMatch = iou >= TRACK_IOU_MATCH || centerDist <= TRACK_CENTER_MATCH;
        const score = iou + Math.max(0, TRACK_CENTER_MATCH - centerDist);
        if (isMatch && score > bestIoU) {
          bestIoU = score;
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) {
        const nextHits = (existing[bestIdx].hitCount ?? 0) + 1;
        const nowStable = existing[bestIdx].stable || nextHits >= MIN_STABLE_HITS;
        const smoothedNx =
          existing[bestIdx].nx * (1 - TRACK_SMOOTHING_ALPHA) + det.nx * TRACK_SMOOTHING_ALPHA;
        const smoothedNy =
          existing[bestIdx].ny * (1 - TRACK_SMOOTHING_ALPHA) + det.ny * TRACK_SMOOTHING_ALPHA;
        const smoothedNw =
          existing[bestIdx].nw * (1 - TRACK_SMOOTHING_ALPHA) + det.nw * TRACK_SMOOTHING_ALPHA;
        const smoothedNh =
          existing[bestIdx].nh * (1 - TRACK_SMOOTHING_ALPHA) + det.nh * TRACK_SMOOTHING_ALPHA;
        existing[bestIdx] = {
          ...existing[bestIdx],
          ...det,
          nx: smoothedNx,
          ny: smoothedNy,
          nw: smoothedNw,
          nh: smoothedNh,
          lastSeen: now,
          hitCount: nextHits,
          missCount: 0,
          stable: nowStable,
          sticky: existing[bestIdx].sticky || (STICKY_CONFIRMED_TRACKS && nowStable),
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
          hitCount: 1,
          missCount: 0,
          stable: false,
          sticky: false,
        });
      }
    }

    const aged = existing.map((track, idx) => {
      if (used.has(idx) || track.lastSeen === now) return track;
      return { ...track, missCount: (track.missCount ?? 0) + 1 };
    });

    const filtered = aged.filter((t) => {
      if (STICKY_CONFIRMED_TRACKS && t.sticky) {
        return now - t.lastSeen <= STICKY_MAX_IDLE_MS;
      }
      return now - t.lastSeen <= TRACK_TTL_MS && (t.missCount ?? 0) <= MAX_MISSED_DETECTIONS;
    });
    tracksRef.current = filtered;
    setLiveCards(filtered.filter((t) => t.stable || t.locked || t.sticky));
  };

  const startDetectionLoop = () => {
    if (!ENABLE_OPENCV_MULTICARD) return;
    if (detectTimerRef.current) {
      clearInterval(detectTimerRef.current);
    }
    detectTimerRef.current = setInterval(() => {
      if (
        !ENABLE_OPENCV_MULTICARD ||
        !streamRef.current ||
        modeRef.current !== "live" ||
        !multiCardModeRef.current ||
        !opencvReadyRef.current
      )
        return;
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
    const cv = window.cv;
    if (ENABLE_OPENCV_MULTICARD && opencvReady && cv && rect?.corners?.length === 4) {
      const crop = document.createElement("canvas");
      crop.width = CARD_WARP_WIDTH;
      crop.height = CARD_WARP_HEIGHT;
      const src = cv.imread(canvas);
      const dst = new cv.Mat();
      const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
        rect.corners[0].x, rect.corners[0].y,
        rect.corners[1].x, rect.corners[1].y,
        rect.corners[2].x, rect.corners[2].y,
        rect.corners[3].x, rect.corners[3].y,
      ]);
      const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        0, CARD_WARP_HEIGHT,
        CARD_WARP_WIDTH, CARD_WARP_HEIGHT,
        CARD_WARP_WIDTH, 0,
      ]);
      const matrix = cv.getPerspectiveTransform(srcTri, dstTri);
      cv.warpPerspective(
        src,
        dst,
        matrix,
        new cv.Size(CARD_WARP_WIDTH, CARD_WARP_HEIGHT),
        cv.INTER_LINEAR,
        cv.BORDER_REPLICATE,
        new cv.Scalar()
      );
      cv.imshow(crop, dst);
      const dataUrl = crop.toDataURL("image/jpeg", 0.92);
      src.delete();
      dst.delete();
      srcTri.delete();
      dstTri.delete();
      matrix.delete();
      return dataUrl.split(",")[1] ?? "";
    }

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
      // Server pipeline does detect+crop+classify; skip client OpenCV path when it is on.
      if (
        ENABLE_OPENCV_MULTICARD &&
        !USE_PIPELINE &&
        mode === "live" &&
        multiCardMode &&
        opencvReady
      ) {
        const now = Date.now();
        const activeTracks = tracksRef.current.filter(
          (t) =>
            (t.sticky && STICKY_CONFIRMED_TRACKS) ||
            (now - t.lastSeen < ACTIVE_TRACK_WINDOW_MS && (t.stable || t.locked))
        );
        const unresolved = activeTracks.filter((t) => !t.locked);
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
              const shouldStick =
                track.sticky || (STICKY_CONFIRMED_TRACKS && (track.stable || probability >= LOCK_THRESHOLD));
              return {
                ...track,
                label: top.label ?? track.label,
                probability,
                locked: track.locked || probability >= LOCK_THRESHOLD,
                sticky: shouldStick,
                endpoint,
              };
            });
            tracksRef.current = updated;
            setLiveCards(updated.filter((t) => t.stable || t.locked || t.sticky));
          }
        }
        if (activeTracks.length > 0) {
          setResult(null);
          return;
        }
      }

      const canvas = drawCurrentFrameToCanvas();
      if (!canvas) return;
      const imageBase64 = canvas.toDataURL("image/jpeg", 0.9).split(",")[1] ?? "";
      if (!imageBase64) return;
      resetLiveTrackingState();
      if (USE_PIPELINE) {
        await runPipelinePrediction(imageBase64, "live", {
          w: canvas.width,
          h: canvas.height,
        });
      } else {
        await runPrediction(imageBase64, "live");
      }
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
            {USE_PIPELINE && " Multi-card frames use the detect → classify pipeline when VITE_PIPELINE_PATH is set."}
            </p>
            <p className="muted small">
              Build-time routes: classifier{" "}
              <code>{import.meta.env.VITE_PREDICT_PATH ?? "/predict"}</code>
              {" · "}
              pipeline{" "}
              <code>
                {(import.meta.env.VITE_PIPELINE_PATH ?? "").trim() || "(empty — add to .env.production and rebuild)"}
              </code>
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

              {USE_PIPELINE && (
                <>
                  <label className="label">
                    Detector confidence: {detectConf.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0.05}
                    max={0.9}
                    step={0.05}
                    value={detectConf}
                    onChange={(e) => setDetectConf(Number(e.target.value))}
                  />
                </>
              )}

              <label className={`checkRow${!ENABLE_OPENCV_MULTICARD ? " muted" : ""}`}>
                <input
                  type="checkbox"
                  checked={multiCardMode}
                  disabled={!ENABLE_OPENCV_MULTICARD}
                  onChange={(e) => {
                    setMultiCardMode(e.target.checked);
                    resetLiveTrackingState();
                  }}
                />
                Detect multiple cards per frame (OpenCV)
                {!ENABLE_OPENCV_MULTICARD && " — off (use server pipeline / YOLO). Enable in code: ENABLE_OPENCV_MULTICARD."}
              </label>

              {!USE_PIPELINE && (
                <p className="muted small" style={{ marginTop: "0.75rem", maxWidth: "42rem" }}>
                  <strong>Note:</strong> Without <code>VITE_PIPELINE_PATH</code>, live mode sends the{" "}
                  <em>entire</em> camera frame to the classifier. The model was trained on{" "}
                  <em>tight card crops</em>, so distant table shots often look like random suits (e.g. J♠ vs J♣) with
                  low confidence. Fix: add <code>VITE_PIPELINE_PATH=/predict-pipeline</code> in{" "}
                  <code>frontend/.env</code> (YOLO crop → classify), or zoom so the card fills most of the frame, or
                  use <strong>Upload</strong> with a close-up photo.
                </p>
              )}
              {USE_PIPELINE && (
                <p className="muted small" style={{ marginTop: "0.75rem" }}>
                  Pipeline mode: each frame is sent to <code>/predict-pipeline</code> (detect boxes, then classify each
                  crop).
                </p>
              )}

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
              {ENABLE_OPENCV_MULTICARD && (
                <p className="muted small">OpenCV detector: {opencvReady ? "Ready" : "Loading..."}</p>
              )}
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
                {pipelineFrameSize?.w > 0 &&
                  pipelineFrameSize?.h > 0 &&
                  (() => {
                    const fw = pipelineFrameSize.w;
                    const fh = pipelineFrameSize.h;
                    const cards = pipelineResult?.cards ?? [];
                    const dets = pipelineResult?.detections ?? [];
                    const items = cards.length > 0
                      ? cards.map((c, i) => ({
                          xyxy: c.xyxy,
                          label: c.classification?.top_prediction?.label,
                          prob: c.classification?.top_prediction?.probability,
                          detConf: c.detection_confidence,
                          key: `pipe-card-${i}`,
                        }))
                      : dets.map((d, i) => ({
                          xyxy: d.xyxy,
                          label: d.label,
                          prob: null,
                          detConf: d.confidence,
                          key: `pipe-det-${i}`,
                        }));
                    return items
                      .filter((it) => Array.isArray(it.xyxy) && it.xyxy.length === 4)
                      .map((it) => {
                        const [x1, y1, x2, y2] = it.xyxy;
                        const left = Math.max(0, x1) / fw;
                        const top = Math.max(0, y1) / fh;
                        const width = Math.max(1, x2 - x1) / fw;
                        const height = Math.max(1, y2 - y1) / fh;
                        const labelText = it.label
                          ? `${it.label}${it.prob != null ? ` ${(it.prob * 100).toFixed(0)}%` : ""}${
                              it.detConf != null ? ` (det ${(it.detConf * 100).toFixed(0)}%)` : ""
                            }`
                          : it.detConf != null
                            ? `card ${(it.detConf * 100).toFixed(0)}%`
                            : "card";
                        return (
                          <div
                            key={it.key}
                            className="cardBox lockedCardBox"
                            style={{
                              left: `${left * 100}%`,
                              top: `${top * 100}%`,
                              width: `${width * 100}%`,
                              height: `${height * 100}%`,
                            }}
                          >
                            <span className="cardLabel">{labelText}</span>
                          </div>
                        );
                      });
                  })()}
              </div>
            )}
          </article>

          <article className="panel">
            <h2>Predictions</h2>
            {error && <p className="error">{error}</p>}
            {!error &&
              !result?.predictions?.[0] &&
              !pipelineResult &&
              liveCards.length === 0 && <p className="muted">No prediction yet.</p>}
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
            {!error && pipelineResult && (
              <div>
                {pipelineResult.message && <p className="muted">{pipelineResult.message}</p>}
                {!pipelineResult.message &&
                  !(pipelineResult.cards?.length > 0) &&
                  (pipelineResult.detections?.length > 0) && (
                    <p className="muted">
                      {pipelineResult.detections.length} detection(s) returned but no crops were classified.
                    </p>
                  )}
                {!pipelineResult.message &&
                  !(pipelineResult.cards?.length > 0) &&
                  !(pipelineResult.detections?.length > 0) && (
                    <p className="muted">Pipeline returned no detections.</p>
                  )}
                <details style={{ marginTop: "0.75rem" }}>
                  <summary className="muted small" style={{ cursor: "pointer" }}>
                    Debug: raw pipeline response
                  </summary>
                  <pre
                    className="muted small"
                    style={{
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      maxHeight: "260px",
                      overflow: "auto",
                      background: "rgba(255,255,255,0.04)",
                      padding: "0.5rem",
                      borderRadius: "6px",
                    }}
                  >
                    {JSON.stringify(
                      {
                        detector_endpoint: pipelineResult.detector_endpoint,
                        classifier_endpoint: pipelineResult.classifier_endpoint,
                        detections_count: pipelineResult.detections?.length ?? 0,
                        cards_count: pipelineResult.cards?.length ?? 0,
                        detections: (pipelineResult.detections ?? []).slice(0, 8),
                        message: pipelineResult.message,
                        frame_size: pipelineFrameSize,
                        detect_conf_sent: detectConf,
                      },
                      null,
                      2
                    )}
                  </pre>
                </details>
                {pipelineResult.cards?.length > 0 && (
                  <ul>
                    {pipelineResult.cards.map((card, idx) => {
                      const top = card.classification?.top_prediction;
                      return (
                        <li key={`pipe-${idx}-${top?.label ?? "?"}`}>
                          Card {idx + 1}
                          {top?.label != null ? (
                            <>
                              : <strong>{top.label}</strong> (
                              {((top.probability ?? 0) * 100).toFixed(2)}%)
                            </>
                          ) : (
                            <span className="muted"> (no classification)</span>
                          )}
                          <span className="muted small">
                            {" "}
                            det {((card.detection_confidence ?? 0) * 100).toFixed(0)}%
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                )}
                {(pipelineResult.detector_endpoint || pipelineResult.classifier_endpoint) && (
                  <p className="muted small">
                    {pipelineResult.detector_endpoint && (
                      <>
                        Detector: <code>{pipelineResult.detector_endpoint}</code>
                        <br />
                      </>
                    )}
                    {pipelineResult.classifier_endpoint && (
                      <>
                        Classifier: <code>{pipelineResult.classifier_endpoint}</code>
                      </>
                    )}
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
