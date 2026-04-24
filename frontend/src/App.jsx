import { useEffect, useMemo, useRef, useState } from "react";
import NavBar from "./components/NavBar";
import { predictCard, predictCards, predictPipeline } from "./services/api";

const USE_PIPELINE = Boolean((import.meta.env.VITE_PIPELINE_PATH ?? "").trim());

const TRACK_TTL_MS = 3500;
const DETECT_INTERVAL_MS = 250;
const CAMERA_REQUEST_WIDTH = 960;
const CAMERA_REQUEST_HEIGHT = 540;
const CAMERA_REQUEST_FPS = 30;
const PROCESS_MAX_WIDTH = 480;
const CARD_WARP_WIDTH = 200;
const CARD_WARP_HEIGHT = 300;
const TRACK_IOU_MATCH = 0.25;
const TRACK_CENTER_MATCH = 0.16;
const ACTIVE_TRACK_WINDOW_MS = 1500;
const STICKY_CONFIRMED_TRACKS = true;
const STICKY_MAX_IDLE_MS = 12000;

/** Normalized fixed crop region (first locked card) + optional quad for warp. */
const emptySlotPreview = () => ({ dealer: null, player: null });

const DEFAULT_TUNING = {
  cardRatioMin: 1.2,
  cardRatioMax: 2.0,
  cardMinExtent: 0.65,
  cardMinAreaFrac: 0.003,
  cardMaxAreaFrac: 0.55,
  cannyLow: 30,
  cannyHigh: 100,
  nmsIou: 0.4,
  trackSmoothing: 0.35,
  minStableHits: 2,
  maxMissed: 10,
  lockThreshold: 0.9,
};

// Client-side OpenCV multi-card path (contours + tracking). Re-enabled after abandoning the YOLO pipeline.
const ENABLE_OPENCV_MULTICARD = true;

const RANK_VALUES = {
  ace: 11,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  jack: 10,
  queen: 10,
  king: 10,
};

const parseCardLabel = (label) => {
  if (!label) return null;
  const m = String(label).toLowerCase().match(/^([a-z]+)\s+of\s+([a-z]+)s?$/);
  if (!m) return null;
  const rank = m[1];
  if (!(rank in RANK_VALUES)) return null;
  return { rank, suit: m[2] };
};

const handTotal = (cards) => {
  let total = 0;
  let aces = 0;
  for (const c of cards) {
    const parsed = parseCardLabel(c.label);
    if (!parsed) continue;
    total += RANK_VALUES[parsed.rank];
    if (parsed.rank === "ace") aces += 1;
  }
  while (total > 21 && aces > 0) {
    total -= 10;
    aces -= 1;
  }
  return total;
};

const handSummary = (cards) => {
  const classified = cards.filter((c) => parseCardLabel(c.label));
  const total = handTotal(classified);
  if (classified.length === 0) {
    return { total: 0, status: "empty" };
  }
  if (total > 21) return { total, status: "bust" };
  if (total === 21 && classified.length === 2) return { total, status: "blackjack" };
  if (total === 21) return { total, status: "twenty-one" };
  return { total, status: "ok" };
};

const HI_LO_VALUES = {
  two: 1, three: 1, four: 1, five: 1, six: 1,
  seven: 0, eight: 0, nine: 0,
  ten: -1, jack: -1, queen: -1, king: -1, ace: -1,
};

const hiLoDelta = (label) => {
  const p = parseCardLabel(label);
  if (!p) return 0;
  return HI_LO_VALUES[p.rank] ?? 0;
};

const currentRoundLeader = (dealer, player) => {
  const d = handSummary(dealer);
  const p = handSummary(player);
  if (p.status === "bust" && d.status === "bust") return "dealer";
  if (p.status === "bust") return "dealer";
  if (d.status === "bust") return "player";
  if (p.status === "empty" && d.status === "empty") return null;
  if (p.total > d.total) return "player";
  if (d.total > p.total) return "dealer";
  if (p.total === 0) return null;
  return "push";
};

const suggestAction = ({ dealer, player, trueCount }) => {
  if (player.length === 0) return "Waiting for player cards";
  const playerSummary = handSummary(player);
  if (playerSummary.status === "bust") return "Busted — round over";
  if (playerSummary.status === "blackjack") return "Blackjack — stand";
  const dealerUp = dealer.find((c) => parseCardLabel(c.label));
  const dealerRank = dealerUp ? parseCardLabel(dealerUp.label)?.rank : null;
  const dealerVal = dealerRank ? RANK_VALUES[dealerRank] : null;
  const total = playerSummary.total;

  // Simple basic strategy, with a few count-based deviations.
  if (total <= 8) return "Hit";
  if (total === 9) {
    if (dealerVal && dealerVal >= 3 && dealerVal <= 6) return "Double (or hit)";
    return "Hit";
  }
  if (total === 10 || total === 11) {
    if (!dealerVal) return "Hit";
    if (total === 11 && dealerVal >= 2 && dealerVal <= 10) return "Double (or hit)";
    if (total === 10 && dealerVal >= 2 && dealerVal <= 9) return "Double (or hit)";
    return "Hit";
  }
  if (total === 12) {
    if (!dealerVal) return "Hit";
    if (dealerVal >= 4 && dealerVal <= 6) return "Stand";
    if (trueCount >= 3 && dealerVal === 2) return "Stand (+3 deviation)";
    if (trueCount >= 2 && dealerVal === 3) return "Stand (+2 deviation)";
    return "Hit";
  }
  if (total >= 13 && total <= 16) {
    if (!dealerVal) return "Hit";
    if (dealerVal >= 2 && dealerVal <= 6) return "Stand";
    if (total === 16 && dealerVal === 10 && trueCount >= 0) return "Stand (Illustrious 18)";
    if (total === 15 && dealerVal === 10 && trueCount >= 4) return "Stand (deviation)";
    return "Hit";
  }
  return "Stand";
};

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
  const [cropPadding, setCropPadding] = useState(4);
  const [multiCardMode, setMultiCardMode] = useState(ENABLE_OPENCV_MULTICARD);
  const [opencvReady, setOpencvReady] = useState(false);
  const [result, setResult] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineFrameSize, setPipelineFrameSize] = useState(null);
  const [liveCards, setLiveCards] = useState([]);
  const [error, setError] = useState("");
  /** Fixed dealer/player regions after first card each; OpenCV stops once both exist. */
  const [dealerSlot, setDealerSlot] = useState(null);
  const [playerSlot, setPlayerSlot] = useState(null);
  const [slotArmed, setSlotArmed] = useState(null);
  const [slotPreview, setSlotPreview] = useState(emptySlotPreview);

  const [numDecks, setNumDecks] = useState(6);
  const [burnedCards, setBurnedCards] = useState(1);
  const [runningCount, setRunningCount] = useState(0);
  const [cardsSeen, setCardsSeen] = useState(0);
  const [roundsWon, setRoundsWon] = useState({ player: 0, dealer: 0, push: 0 });
  const [committedDealer, setCommittedDealer] = useState([]);
  const [committedPlayer, setCommittedPlayer] = useState([]);
  const [tuning, setTuning] = useState(DEFAULT_TUNING);
  const [showTuning, setShowTuning] = useState(false);

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
  const dealerSlotRef = useRef(null);
  const playerSlotRef = useRef(null);
  const slotArmedRef = useRef(null);
  const nextHandCardIdRef = useRef(1);
  const discoveryAnchoredRef = useRef({ dealer: false, player: false });
  const tuningRef = useRef(DEFAULT_TUNING);

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
    tuningRef.current = tuning;
  }, [tuning]);

  useEffect(() => {
    dealerSlotRef.current = dealerSlot;
  }, [dealerSlot]);

  useEffect(() => {
    playerSlotRef.current = playerSlot;
  }, [playerSlot]);

  useEffect(() => {
    slotArmedRef.current = slotArmed;
  }, [slotArmed]);

  useEffect(() => {
    if (mode !== "live" && isCameraOn) {
      stopCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  /** First card per side: OpenCV track locks → save fixed slot + commit (one card at a time per half). */
  useEffect(() => {
    if (!ENABLE_OPENCV_MULTICARD || USE_PIPELINE || !multiCardMode) return;
    if (dealerSlot && playerSlot) return;
    if (liveCards.length === 0) return;

    const locked = liveCards.filter((c) => c.label && c.locked);
    const card = !dealerSlot
      ? locked.find((c) => (c.ny ?? 0) + (c.nh ?? 0) / 2 < 0.5)
      : !playerSlot
        ? locked.find((c) => (c.ny ?? 0) + (c.nh ?? 0) / 2 >= 0.5)
        : null;
    if (!card) return;

    const canvas = drawCurrentFrameToCanvas();
    if (!canvas) return;

    const anchorKey = !dealerSlot ? "dealer" : "player";
    if (discoveryAnchoredRef.current[anchorKey]) return;
    discoveryAnchoredRef.current[anchorKey] = true;
    const cw = canvas.width;
    const ch = canvas.height;
    const cornersNorm =
      card.corners && card.corners.length === 4
        ? card.corners.map((p) => ({ nx: p.x / cw, ny: p.y / ch }))
        : null;
    const slot = {
      nx: card.nx,
      ny: card.ny,
      nw: card.nw,
      nh: card.nh,
      cornersNorm,
    };
    const entry = {
      id: nextHandCardIdRef.current++,
      label: card.label,
      probability: card.probability ?? 0,
    };
    const delta = hiLoDelta(card.label);

    if (!dealerSlot) {
      dealerSlotRef.current = slot;
      setDealerSlot(slot);
      setCommittedDealer((prev) => [...prev, entry]);
    } else {
      playerSlotRef.current = slot;
      setPlayerSlot(slot);
      setCommittedPlayer((prev) => [...prev, entry]);
    }
    setRunningCount((v) => v + delta);
    setCardsSeen((v) => v + 1);

    tracksRef.current = [];
    setLiveCards([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- drawCurrentFrameToCanvas is stable for this effect
  }, [liveCards, dealerSlot, playerSlot, multiCardMode]);

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
      const data = await predictPipeline({ imageBase64, topK, detectConf, cropPadding });
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
    // Keep nextTrackIdRef monotonic across camera restarts (tracks only reset here).
    setLiveCards([]);
  };

  const resetAll = () => {
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    nextHandCardIdRef.current = 1;
    discoveryAnchoredRef.current = { dealer: false, player: false };
    dealerSlotRef.current = null;
    playerSlotRef.current = null;
    slotArmedRef.current = null;
    setLiveCards([]);
    setDealerSlot(null);
    setPlayerSlot(null);
    setSlotArmed(null);
    setSlotPreview(emptySlotPreview());
    setCommittedDealer([]);
    setCommittedPlayer([]);
    setRunningCount(0);
    setCardsSeen(0);
    setRoundsWon({ player: 0, dealer: 0, push: 0 });
    setResult(null);
    setError("");
  };

  const recordRoundWinner = (who) => {
    if (who !== "player" && who !== "dealer" && who !== "push") return;
    setRoundsWon((prev) => ({ ...prev, [who]: prev[who] + 1 }));
    tracksRef.current = [];
    discoveryAnchoredRef.current = { dealer: false, player: false };
    dealerSlotRef.current = null;
    playerSlotRef.current = null;
    slotArmedRef.current = null;
    setLiveCards([]);
    setDealerSlot(null);
    setPlayerSlot(null);
    setSlotArmed(null);
    setSlotPreview(emptySlotPreview());
    setCommittedDealer([]);
    setCommittedPlayer([]);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: CAMERA_REQUEST_WIDTH },
          height: { ideal: CAMERA_REQUEST_HEIGHT },
          frameRate: { ideal: CAMERA_REQUEST_FPS, max: CAMERA_REQUEST_FPS },
        },
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
    discoveryAnchoredRef.current = { dealer: false, player: false };
    dealerSlotRef.current = null;
    playerSlotRef.current = null;
    slotArmedRef.current = null;
    setDealerSlot(null);
    setPlayerSlot(null);
    setSlotArmed(null);
    setSlotPreview(emptySlotPreview());
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

    const maxWidth = PROCESS_MAX_WIDTH;
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

    cv.Canny(blur, edges, tuningRef.current.cannyLow, tuningRef.current.cannyHigh);
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
    const minArea = frameArea * tuningRef.current.cardMinAreaFrac;
    const maxArea = frameArea * tuningRef.current.cardMaxAreaFrac;
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
      if (ratio < tuningRef.current.cardRatioMin || ratio > tuningRef.current.cardRatioMax) {
        cnt.delete();
        continue;
      }

      const rotatedArea = rotW * rotH;
      const extent = area / Math.max(rotatedArea, 1);
      if (extent < tuningRef.current.cardMinExtent) {
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
      const overlaps = kept.some((k) => rectIoU(k, c) > tuningRef.current.nmsIou);
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
        const nowStable = existing[bestIdx].stable || nextHits >= tuningRef.current.minStableHits;
        const smoothedNx =
          existing[bestIdx].nx * (1 - tuningRef.current.trackSmoothing) + det.nx * tuningRef.current.trackSmoothing;
        const smoothedNy =
          existing[bestIdx].ny * (1 - tuningRef.current.trackSmoothing) + det.ny * tuningRef.current.trackSmoothing;
        const smoothedNw =
          existing[bestIdx].nw * (1 - tuningRef.current.trackSmoothing) + det.nw * tuningRef.current.trackSmoothing;
        const smoothedNh =
          existing[bestIdx].nh * (1 - tuningRef.current.trackSmoothing) + det.nh * tuningRef.current.trackSmoothing;
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
      if (t.locked) {
        return (
          now - t.lastSeen <= TRACK_TTL_MS * 2 &&
          (t.missCount ?? 0) <= Math.max(tuningRef.current.maxMissed, 30)
        );
      }
      if (STICKY_CONFIRMED_TRACKS && t.sticky) {
        return now - t.lastSeen <= STICKY_MAX_IDLE_MS;
      }
      return now - t.lastSeen <= TRACK_TTL_MS && (t.missCount ?? 0) <= tuningRef.current.maxMissed;
    });
    tracksRef.current = filtered;
    const visible = filtered.filter((t) => t.stable || t.locked || t.sticky);
    setLiveCards((prev) => {
      if (prev.length !== visible.length) return visible;
      for (let i = 0; i < visible.length; i++) {
        const a = prev[i];
        const b = visible[i];
        if (
          a.id !== b.id ||
          a.label !== b.label ||
          a.locked !== b.locked ||
          Math.abs((a.probability ?? 0) - (b.probability ?? 0)) > 0.002 ||
          Math.abs((a.nx ?? 0) - b.nx) > 0.01 ||
          Math.abs((a.ny ?? 0) - b.ny) > 0.01 ||
          Math.abs((a.nw ?? 0) - b.nw) > 0.01 ||
          Math.abs((a.nh ?? 0) - b.nh) > 0.01
        ) {
          return visible;
        }
      }
      return prev;
    });
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
      if (dealerSlotRef.current && playerSlotRef.current) return;

      const canvas = drawCurrentFrameToCanvas();
      if (!canvas) return;
      let rects = detectCardRects(canvas);
      const centerYN = (r) => r.ny + r.nh / 2;
      const inHalf = (upper) =>
        rects
          .filter((r) => (upper ? centerYN(r) < 0.5 : centerYN(r) >= 0.5))
          .sort((a, b) => b.nw * b.nh - a.nw * a.nh);
      if (!dealerSlotRef.current) {
        const list = inHalf(true);
        rects = list[0] ? [list[0]] : [];
      } else if (!playerSlotRef.current) {
        const list = inHalf(false);
        rects = list[0] ? [list[0]] : [];
      } else {
        rects = [];
      }
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

  const cropSlotToBase64 = (canvas, slot) => {
    if (!canvas || !slot) return "";
    const cw = canvas.width;
    const ch = canvas.height;
    if (slot.cornersNorm?.length === 4) {
      const corners = slot.cornersNorm.map((p) => ({ x: p.nx * cw, y: p.ny * ch }));
      return cropRectToBase64(canvas, {
        corners,
        nx: slot.nx,
        ny: slot.ny,
        nw: slot.nw,
        nh: slot.nh,
        x: Math.round(slot.nx * cw),
        y: Math.round(slot.ny * ch),
        w: Math.max(1, Math.round(slot.nw * cw)),
        h: Math.max(1, Math.round(slot.nh * ch)),
      });
    }
    const x = Math.round(slot.nx * cw);
    const y = Math.round(slot.ny * ch);
    const w = Math.max(1, Math.round(slot.nw * cw));
    const h = Math.max(1, Math.round(slot.nh * ch));
    return cropRectToBase64(canvas, { x, y, w, h, nx: slot.nx, ny: slot.ny, nw: slot.nw, nh: slot.nh });
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
      // Fixed slots: no full-frame search — classify only the armed slot after both anchors exist.
      if (
        ENABLE_OPENCV_MULTICARD &&
        !USE_PIPELINE &&
        mode === "live" &&
        multiCardMode &&
        opencvReady
      ) {
        const canvas = drawCurrentFrameToCanvas();
        if (!canvas) return;

        if (dealerSlotRef.current && playerSlotRef.current) {
          const armed = slotArmedRef.current;
          if (armed && !inFlightRef.current) {
            const slot = armed === "dealer" ? dealerSlotRef.current : playerSlotRef.current;
            const imageBase64 = cropSlotToBase64(canvas, slot);
            if (imageBase64) {
              const data = await runPrediction(imageBase64, "live", false);
              const top = data?.predictions?.[0]?.top_prediction;
              const prob = top?.probability ?? 0;
              const label = top?.label ?? "";
              if (armed === "dealer") {
                setSlotPreview((p) => ({ ...p, dealer: label ? { label, probability: prob } : null }));
              } else {
                setSlotPreview((p) => ({ ...p, player: label ? { label, probability: prob } : null }));
              }
              if (label && prob >= tuningRef.current.lockThreshold) {
                const entry = {
                  id: nextHandCardIdRef.current++,
                  label,
                  probability: prob,
                };
                const delta = hiLoDelta(label);
                if (armed === "dealer") {
                  setCommittedDealer((prev) => [...prev, entry]);
                } else {
                  setCommittedPlayer((prev) => [...prev, entry]);
                }
                setRunningCount((v) => v + delta);
                setCardsSeen((v) => v + 1);
                slotArmedRef.current = null;
                setSlotArmed(null);
                setSlotPreview(emptySlotPreview());
              }
            }
          }
          setResult(null);
          return;
        }

        const now = Date.now();
        const activeTracks = tracksRef.current.filter(
          (t) =>
            (t.sticky && STICKY_CONFIRMED_TRACKS) ||
            (now - t.lastSeen < ACTIVE_TRACK_WINDOW_MS && (t.stable || t.locked))
        );
        const unresolved = activeTracks.filter((t) => !t.locked);
        if (unresolved.length > 0) {
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
                track.sticky || (STICKY_CONFIRMED_TRACKS && (track.stable || probability >= tuningRef.current.lockThreshold));
              return {
                ...track,
                label: top.label ?? track.label,
                probability,
                locked: track.locked || probability >= tuningRef.current.lockThreshold,
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
        setResult(null);
        return;
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
    slotArmedRef.current = null;
    setSlotArmed(null);
    setSlotPreview(emptySlotPreview());
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
            {USE_PIPELINE && " Pipeline mode uses detect → classify when VITE_PIPELINE_PATH is set."}
            {!USE_PIPELINE &&
              ENABLE_OPENCV_MULTICARD &&
              " Live OpenCV mode uses one fixed slot per side after the first card is found in each half of the frame."}
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

                  <label className="label">
                    Crop padding around detection: {cropPadding.toFixed(1)}x
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={10}
                    step={0.5}
                    value={cropPadding}
                    onChange={(e) => setCropPadding(Number(e.target.value))}
                  />
                  <p className="muted small">
                    Detector boxes here are the card&apos;s corner index. Pad each detection by this factor of its size before
                    classifying (e.g. 4x = the crop becomes ~9× the detected box). Increase if the classifier sees only corners.
                  </p>
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
                    discoveryAnchoredRef.current = { dealer: false, player: false };
                    dealerSlotRef.current = null;
                    playerSlotRef.current = null;
                    slotArmedRef.current = null;
                    setDealerSlot(null);
                    setPlayerSlot(null);
                    setSlotArmed(null);
                    setSlotPreview(emptySlotPreview());
                  }}
                />
                Dealer / player slots (OpenCV finds the first card each; same spot for later cards)
                {!ENABLE_OPENCV_MULTICARD && " — off (use server pipeline / YOLO). Enable in code: ENABLE_OPENCV_MULTICARD."}
              </label>

              <label className="label">Decks in shoe</label>
              <select
                value={numDecks}
                onChange={(e) => setNumDecks(Number(e.target.value))}
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={4}>4</option>
                <option value={6}>6</option>
                <option value={8}>8</option>
              </select>

              <label className="label">
                Cards burned / thrown out at shuffle: {burnedCards}
              </label>
              <input
                type="number"
                min={0}
                max={52}
                step={1}
                value={burnedCards}
                onChange={(e) => setBurnedCards(Math.max(0, Number(e.target.value) || 0))}
              />
              <p className="muted small">
                In most casinos the dealer burns 1 card after a shuffle, and sometimes more are discarded. Those cards
                are not visible but are gone from the shoe.
              </p>

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

              {ENABLE_OPENCV_MULTICARD && (
                <details
                  open={showTuning}
                  onToggle={(e) => setShowTuning(e.currentTarget.open)}
                  style={{ marginTop: "0.75rem" }}
                >
                  <summary style={{ cursor: "pointer" }}>
                    <strong>Detector tuning (advanced)</strong>
                  </summary>
                  <p className="muted small">
                    Change one value at a time. Effects are live; no restart needed.
                  </p>

                  <label className="label">
                    Lock threshold (classifier confidence): {tuning.lockThreshold.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0.5}
                    max={0.99}
                    step={0.01}
                    value={tuning.lockThreshold}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, lockThreshold: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Min card area (% of frame): {(tuning.cardMinAreaFrac * 100).toFixed(2)}%
                  </label>
                  <input
                    type="range"
                    min={0.0005}
                    max={0.05}
                    step={0.0005}
                    value={tuning.cardMinAreaFrac}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cardMinAreaFrac: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Max card area (% of frame): {(tuning.cardMaxAreaFrac * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={tuning.cardMaxAreaFrac}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cardMaxAreaFrac: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Aspect ratio min (height/width): {tuning.cardRatioMin.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={1.0}
                    max={1.5}
                    step={0.05}
                    value={tuning.cardRatioMin}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cardRatioMin: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Aspect ratio max (height/width): {tuning.cardRatioMax.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={1.5}
                    max={3.0}
                    step={0.05}
                    value={tuning.cardRatioMax}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cardRatioMax: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Min extent (contour fill of its bbox): {tuning.cardMinExtent.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0.3}
                    max={0.95}
                    step={0.01}
                    value={tuning.cardMinExtent}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cardMinExtent: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Canny low threshold: {tuning.cannyLow}
                  </label>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    step={1}
                    value={tuning.cannyLow}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cannyLow: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Canny high threshold: {tuning.cannyHigh}
                  </label>
                  <input
                    type="range"
                    min={30}
                    max={250}
                    step={1}
                    value={tuning.cannyHigh}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, cannyHigh: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    NMS IoU threshold: {tuning.nmsIou.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0.1}
                    max={0.8}
                    step={0.05}
                    value={tuning.nmsIou}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, nmsIou: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Track smoothing alpha: {tuning.trackSmoothing.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    value={tuning.trackSmoothing}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, trackSmoothing: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Min stable hits before showing: {tuning.minStableHits}
                  </label>
                  <input
                    type="range"
                    min={1}
                    max={8}
                    step={1}
                    value={tuning.minStableHits}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, minStableHits: Number(e.target.value) }))
                    }
                  />

                  <label className="label">
                    Max missed detections before drop: {tuning.maxMissed}
                  </label>
                  <input
                    type="range"
                    min={2}
                    max={40}
                    step={1}
                    value={tuning.maxMissed}
                    onChange={(e) =>
                      setTuning((t) => ({ ...t, maxMissed: Number(e.target.value) }))
                    }
                  />

                  <div className="buttonRow">
                    <button type="button" onClick={() => setTuning(DEFAULT_TUNING)}>
                      Reset tuning to defaults
                    </button>
                  </div>
                </details>
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
                <div className="tableDivider" />
                <span className="tableZoneLabel dealer">Dealer</span>
                <span className="tableZoneLabel player">Player</span>
                {dealerSlot && (
                  <div
                    className="cardBox slotBox"
                    style={{
                      left: `${dealerSlot.nx * 100}%`,
                      top: `${dealerSlot.ny * 100}%`,
                      width: `${dealerSlot.nw * 100}%`,
                      height: `${dealerSlot.nh * 100}%`,
                    }}
                  >
                    <span className="cardLabel">
                      Dealer slot
                      {slotPreview.dealer?.label
                        ? ` · ${slotPreview.dealer.label} ${(slotPreview.dealer.probability * 100).toFixed(0)}%`
                        : ""}
                    </span>
                  </div>
                )}
                {playerSlot && (
                  <div
                    className="cardBox slotBox"
                    style={{
                      left: `${playerSlot.nx * 100}%`,
                      top: `${playerSlot.ny * 100}%`,
                      width: `${playerSlot.nw * 100}%`,
                      height: `${playerSlot.nh * 100}%`,
                    }}
                  >
                    <span className="cardLabel">
                      Player slot
                      {slotPreview.player?.label
                        ? ` · ${slotPreview.player.label} ${(slotPreview.player.probability * 100).toFixed(0)}%`
                        : ""}
                    </span>
                  </div>
                )}
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
            <h2>Blackjack table</h2>
            {error && <p className="error">{error}</p>}

            {(() => {
              const dealerCards = committedDealer;
              const playerCards = committedPlayer;
              const scanningDealer = [];
              const scanningPlayer = [];
              for (const card of liveCards) {
                const centerY = (card.ny ?? 0) + (card.nh ?? 0) / 2;
                (centerY < 0.5 ? scanningDealer : scanningPlayer).push(card);
              }
              const slotMode =
                mode === "live" &&
                ENABLE_OPENCV_MULTICARD &&
                multiCardMode &&
                !USE_PIPELINE;
              const leader = currentRoundLeader(dealerCards, playerCards);
              const decksRemaining = Math.max(
                0.25,
                numDecks - (cardsSeen + burnedCards) / 52
              );
              const trueCount = runningCount / decksRemaining;
              const suggestion = suggestAction({
                dealer: dealerCards,
                player: playerCards,
                trueCount,
              });

              const renderHand = (title, hand, scanning) => {
                const summary = handSummary(hand);
                return (
                  <div className="handBlock" key={title}>
                    <h3>
                      {title}{" "}
                      {summary.status !== "empty" && (
                        <span className="handTotal">— {summary.total}</span>
                      )}
                      {summary.status === "bust" && <span className="muted small"> bust</span>}
                      {summary.status === "blackjack" && (
                        <span className="muted small"> blackjack!</span>
                      )}
                      {summary.status === "twenty-one" && (
                        <span className="muted small"> 21</span>
                      )}
                    </h3>
                    {hand.length === 0 && scanning.length === 0 ? (
                      <p className="muted small">No cards yet.</p>
                    ) : (
                      <ul>
                        {hand.map((card, idx) => (
                          <li key={`c-${card.id}`}>
                            Card {idx + 1}: <strong>{card.label}</strong>{" "}
                            ({(card.probability * 100).toFixed(0)}%)
                          </li>
                        ))}
                        {scanning.map((card) => (
                          <li key={`s-${card.id}`} className="muted small">
                            Scanning: {card.label ? `${card.label} (${(card.probability * 100).toFixed(0)}%)` : "…"}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              };

              const slotPrompt = (() => {
                if (!slotMode || !isLiveScanning) return null;
                if (!dealerSlot) {
                  return "Hold one dealer card in the upper half until it locks (green/yellow box). That fixes the dealer slot.";
                }
                if (!playerSlot) {
                  return "Dealer slot is set. Hold one player card in the lower half until it locks.";
                }
                return "Slots are fixed. Put the next card in the correct slot, then tap Scan for that side. Only that crop is sent to the classifier (no OpenCV search).";
              })();

              return (
                <div>
                  {slotPrompt && (
                    <div className="handBlock">
                      <h3>Live scan steps</h3>
                      <p className="muted small">{slotPrompt}</p>
                      {dealerSlot && playerSlot && (
                        <div className="buttonRow" style={{ marginTop: "0.5rem" }}>
                          <button
                            type="button"
                            disabled={Boolean(slotArmed)}
                            onClick={() => {
                              slotArmedRef.current = "dealer";
                              setSlotArmed("dealer");
                              setSlotPreview((p) => ({ ...p, dealer: null }));
                            }}
                          >
                            Scan dealer slot
                          </button>
                          <button
                            type="button"
                            disabled={Boolean(slotArmed)}
                            onClick={() => {
                              slotArmedRef.current = "player";
                              setSlotArmed("player");
                              setSlotPreview((p) => ({ ...p, player: null }));
                            }}
                          >
                            Scan player slot
                          </button>
                          {slotArmed && (
                            <button
                              type="button"
                              onClick={() => {
                                slotArmedRef.current = null;
                                setSlotArmed(null);
                                setSlotPreview(emptySlotPreview());
                              }}
                            >
                              Cancel scan
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                  {renderHand("Dealer's cards", dealerCards, scanningDealer)}
                  {renderHand("Player's cards", playerCards, scanningPlayer)}

                  <div className="handBlock">
                    <h3>Who's ahead</h3>
                    <p>
                      {leader === "player" && (
                        <>
                          <strong>Player</strong> leads this round.
                        </>
                      )}
                      {leader === "dealer" && (
                        <>
                          <strong>Dealer</strong> leads this round.
                        </>
                      )}
                      {leader === "push" && <>Currently a push.</>}
                      {leader === null && <span className="muted">Waiting for cards.</span>}
                    </p>
                  </div>

                  <div className="handBlock">
                    <h3>Suggested action</h3>
                    <p>
                      <strong>{suggestion}</strong>
                    </p>
                    <p className="muted small">
                      Basic strategy with a few count-based deviations (Hi-Lo). Not financial advice.
                    </p>
                  </div>

                  <div className="handBlock">
                    <h3>Card counting (Hi-Lo)</h3>
                    <ul>
                      <li>Running count: <strong>{runningCount}</strong></li>
                      <li>True count: <strong>{trueCount.toFixed(2)}</strong></li>
                      <li>Cards seen: {cardsSeen}</li>
                      <li>Decks remaining: ~{decksRemaining.toFixed(2)} / {numDecks}</li>
                      <li>Burned / discarded: {burnedCards}</li>
                    </ul>
                  </div>

                  <div className="handBlock">
                    <h3>Rounds</h3>
                    <p>
                      Player: <strong>{roundsWon.player}</strong> · Dealer:{" "}
                      <strong>{roundsWon.dealer}</strong> · Pushes:{" "}
                      <strong>{roundsWon.push}</strong>
                    </p>
                    <div className="buttonRow">
                      <button type="button" onClick={() => recordRoundWinner("player")}>
                        Player won
                      </button>
                      <button type="button" onClick={() => recordRoundWinner("dealer")}>
                        Dealer won
                      </button>
                      <button type="button" onClick={() => recordRoundWinner("push")}>
                        Push
                      </button>
                    </div>
                    <p className="muted small">
                      Recording a round clears the table and dealer/player slots so you can anchor fresh spots. The
                      running count is unchanged (cards stay out of the shoe).
                    </p>
                  </div>

                  <div className="handBlock">
                    <div className="buttonRow">
                      <button type="button" onClick={resetAll}>
                        Reset everything
                      </button>
                    </div>
                    <p className="muted small">
                      Resets the running count, cards seen, round scores, and current table. Use this when a new shoe
                      is shuffled in.
                    </p>
                  </div>

                  {mode === "live" && result?.endpoint && (
                    <p className="muted small">
                      Endpoint: <code>{result.endpoint}</code>
                    </p>
                  )}
                </div>
              );
            })()}
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
