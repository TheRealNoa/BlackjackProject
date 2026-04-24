import { useEffect, useRef, useState } from "react";
import NavBar from "./components/NavBar";
import RequireAuth from "./components/RequireAuth";
import { useAuth } from "./context/AuthContext";
import {
  fetchLeaderboard,
  predictCard,
  predictCards,
  predictPipeline,
  submitRoundOutcome,
} from "./services/api";

const USE_PIPELINE = Boolean((import.meta.env.VITE_PIPELINE_PATH ?? "").trim());
const LIVE_TOP_K = 3;
const LIVE_SCAN_INTERVAL_MS = 600;

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
  lockThreshold: 0.52,
  /** Consecutive slot scans with same new label ≥ lockThreshold before auto-commit. */
  slotAgreeFrames: 2,
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

/** True if at least one ace can count as 11 without busting (standard "soft" hand). */
const handIsSoft = (cards) => {
  const classified = cards.filter((c) => parseCardLabel(c.label));
  if (classified.length === 0) return false;
  let minTotal = 0;
  let aceCount = 0;
  for (const c of classified) {
    const parsed = parseCardLabel(c.label);
    if (!parsed) continue;
    const v = RANK_VALUES[parsed.rank];
    minTotal += parsed.rank === "ace" ? 1 : v;
    if (parsed.rank === "ace") aceCount += 1;
  }
  if (aceCount === 0) return false;
  return minTotal + 10 <= 21;
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

/** Dealer stands on 17+ except soft 17 (ace + 6 style), which continues until hard 17+ or bust. */
const dealerStands = (dealerCards) => {
  const s = handSummary(dealerCards);
  if (s.status === "bust") return true;
  if (s.total > 17) return true;
  if (s.total === 17 && handIsSoft(dealerCards)) return false;
  return s.total >= 17;
};

const resolveRoundOutcome = (dealerCards, playerCards) => {
  const d = handSummary(dealerCards);
  const p = handSummary(playerCards);
  if (p.status === "bust") return "dealer";
  if (d.status === "bust") return "player";
  if (p.total > d.total) return "player";
  if (d.total > p.total) return "dealer";
  return "push";
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

/** Returns { label, probability } when ready to auto-commit, else null. Mutates `ag` per side. */
const trySlotClassifierAgreement = (ag, lastLabel, side, label, prob, threshold, needFrames) => {
  const row = ag[side];
  const last = lastLabel[side];
  if (!label || !parseCardLabel(label)) {
    row.streak = 0;
    row.candidate = null;
    return null;
  }
  if (prob < threshold) {
    row.streak = 0;
    row.candidate = null;
    return null;
  }
  if (label === last) {
    row.streak = 0;
    row.candidate = null;
    return null;
  }
  if (label !== row.candidate) {
    row.candidate = label;
    row.streak = 1;
  } else {
    row.streak += 1;
  }
  if (row.streak >= needFrames) {
    row.streak = 0;
    row.candidate = null;
    return { label, probability: prob };
  }
  return null;
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
  const { session } = useAuth();
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLiveScanning, setIsLiveScanning] = useState(false);
  const [detectConf] = useState(0.1);
  const [cropPadding] = useState(4);
  const [opencvReady, setOpencvReady] = useState(false);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineFrameSize, setPipelineFrameSize] = useState(null);
  const [liveCards, setLiveCards] = useState([]);
  const [error, setError] = useState("");
  /** Fixed dealer/player regions after first card each; OpenCV stops once both exist. */
  const [dealerSlot, setDealerSlot] = useState(null);
  const [playerSlot, setPlayerSlot] = useState(null);
  const [slotPreview, setSlotPreview] = useState(emptySlotPreview);

  const [numDecks, setNumDecks] = useState(1);
  const [burnedCards, setBurnedCards] = useState(1);
  const [runningCount, setRunningCount] = useState(0);
  const [cardsSeen, setCardsSeen] = useState(0);
  const [roundsWon, setRoundsWon] = useState({ player: 0, dealer: 0, push: 0 });
  const [committedDealer, setCommittedDealer] = useState([]);
  const [committedPlayer, setCommittedPlayer] = useState([]);
  const [awaitTableClear, setAwaitTableClear] = useState(false);
  const [lastRoundOutcome, setLastRoundOutcome] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);
  const [leaderboardError, setLeaderboardError] = useState("");
  const [tuning] = useState(DEFAULT_TUNING);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const scanTimerRef = useRef(null);
  const detectTimerRef = useRef(null);
  const inFlightRef = useRef(false);
  const tracksRef = useRef([]);
  const nextTrackIdRef = useRef(1);
  const opencvReadyRef = useRef(opencvReady);
  const dealerSlotRef = useRef(null);
  const playerSlotRef = useRef(null);
  const lastCommittedSlotLabelRef = useRef({ dealer: null, player: null });
  const slotAgreeRef = useRef({
    dealer: { streak: 0, candidate: null },
    player: { streak: 0, candidate: null },
  });
  const nextHandCardIdRef = useRef(1);
  const discoveryAnchoredRef = useRef({ dealer: false, player: false });
  const tuningRef = useRef(DEFAULT_TUNING);

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
    };
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
    lastCommittedSlotLabelRef.current.dealer = committedDealer.at(-1)?.label ?? null;
    lastCommittedSlotLabelRef.current.player = committedPlayer.at(-1)?.label ?? null;
  }, [committedDealer, committedPlayer]);

  const committedDealerRef = useRef([]);
  const committedPlayerRef = useRef([]);
  const lastAutoScoredHandSigRef = useRef("");
  useEffect(() => {
    committedDealerRef.current = committedDealer;
  }, [committedDealer]);
  useEffect(() => {
    committedPlayerRef.current = committedPlayer;
  }, [committedPlayer]);

  const clearTableForNextRound = () => {
    tracksRef.current = [];
    discoveryAnchoredRef.current = { dealer: false, player: false };
    slotAgreeRef.current = {
      dealer: { streak: 0, candidate: null },
      player: { streak: 0, candidate: null },
    };
    dealerSlotRef.current = null;
    playerSlotRef.current = null;
    setLiveCards([]);
    setDealerSlot(null);
    setPlayerSlot(null);
    setSlotPreview(emptySlotPreview());
    setCommittedDealer([]);
    setCommittedPlayer([]);
  };

  const endRoundWithWinner = (who) => {
    if (who !== "player" && who !== "dealer" && who !== "push") return;
    setRoundsWon((prev) => ({ ...prev, [who]: prev[who] + 1 }));
    setLastRoundOutcome(who);
    clearTableForNextRound();
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
    setIsLiveScanning(false);
    setError("");
    setAwaitTableClear(true);

    if (session?.idToken) {
      submitRoundOutcome({
        outcome: who,
        username: session.username || session.email || "",
      })
        .then(() => fetchLeaderboard({ limit: 10, minGames: 1 }))
        .then((data) => {
          setLeaderboard(data?.items ?? []);
          setLeaderboardError("");
        })
        .catch((err) => {
          const message = err?.response?.data?.error || err?.message || "Could not update leaderboard.";
          setLeaderboardError(message);
        });
    }
  };

  useEffect(() => {
    let cancelled = false;
    setLeaderboardLoading(true);
    fetchLeaderboard({ limit: 10, minGames: 1 })
      .then((data) => {
        if (cancelled) return;
        setLeaderboard(data?.items ?? []);
        setLeaderboardError("");
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err?.response?.data?.error || err?.message || "Could not load leaderboard.";
        setLeaderboardError(message);
      })
      .finally(() => {
        if (cancelled) return;
        setLeaderboardLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /** Auto-resolve when player busts, dealer busts, or dealer has ≥2 cards and stands (hard 17+; hits soft 17). */
  useEffect(() => {
    if (!ENABLE_OPENCV_MULTICARD || USE_PIPELINE) return;

    const dealer = committedDealer;
    const player = committedPlayer;
    if (dealer.length === 0 && player.length === 0) {
      lastAutoScoredHandSigRef.current = "";
      return;
    }

    const handSig = `${dealer.map((c) => c.id).join(",")}|${player.map((c) => c.id).join(",")}`;
    const d = handSummary(dealer);
    const p = handSummary(player);
    const playerClassified = player.filter((c) => parseCardLabel(c.label));

    let outcome = null;
    if (p.status === "bust") outcome = "dealer";
    else if (d.status === "bust") outcome = "player";
    else if (dealer.length >= 2 && dealerStands(dealer) && playerClassified.length >= 1) {
      if (p.status !== "empty" && d.total === p.total) outcome = "push";
      else if (
        playerClassified.length >= 2 &&
        d.total > p.total
      ) {
        outcome = "dealer";
      } else {
        outcome = resolveRoundOutcome(dealer, player);
      }
    }
    if (!outcome) return;
    if (lastAutoScoredHandSigRef.current === handSig) return;
    lastAutoScoredHandSigRef.current = handSig;
    endRoundWithWinner(outcome);
  }, [committedDealer, committedPlayer]);

  /** First card per side: OpenCV track locks → save fixed slot + commit (one card at a time per half). */
  useEffect(() => {
    if (!ENABLE_OPENCV_MULTICARD || USE_PIPELINE) return;
    if (awaitTableClear) return;
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
      lastCommittedSlotLabelRef.current.dealer = card.label;
    } else {
      playerSlotRef.current = slot;
      setPlayerSlot(slot);
      setCommittedPlayer((prev) => [...prev, entry]);
      lastCommittedSlotLabelRef.current.player = card.label;
    }
    setRunningCount((v) => v + delta);
    setCardsSeen((v) => v + 1);

    tracksRef.current = [];
    setLiveCards([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- drawCurrentFrameToCanvas is stable for this effect
  }, [liveCards, dealerSlot, playerSlot, awaitTableClear]);

  const runPrediction = async (imageBase64) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const data = await predictCard({ imageBase64, topK: LIVE_TOP_K });
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

  const runPipelinePrediction = async (imageBase64, frameSize = null) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const data = await predictPipeline({
        imageBase64,
        topK: LIVE_TOP_K,
        detectConf,
        cropPadding,
      });
      setPipelineResult(data);
      setPipelineFrameSize(frameSize);
      setError("");
      return data;
    } catch (err) {
      const message = err?.response?.data?.error || err.message || "Pipeline prediction failed.";
      setError(message);
      return null;
    } finally {
      inFlightRef.current = false;
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
    slotAgreeRef.current = {
      dealer: { streak: 0, candidate: null },
      player: { streak: 0, candidate: null },
    };
    dealerSlotRef.current = null;
    playerSlotRef.current = null;
    setLiveCards([]);
    setDealerSlot(null);
    setPlayerSlot(null);
    setSlotPreview(emptySlotPreview());
    setCommittedDealer([]);
    setCommittedPlayer([]);
    setRunningCount(0);
    setCardsSeen(0);
    setRoundsWon({ player: 0, dealer: 0, push: 0 });
    setAwaitTableClear(false);
    setLastRoundOutcome(null);
    setError("");
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
      if (ENABLE_OPENCV_MULTICARD) {
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
    slotAgreeRef.current = {
      dealer: { streak: 0, candidate: null },
      player: { streak: 0, candidate: null },
    };
    dealerSlotRef.current = null;
    playerSlotRef.current = null;
    setDealerSlot(null);
    setPlayerSlot(null);
    setSlotPreview(emptySlotPreview());
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
      if (!ENABLE_OPENCV_MULTICARD || !streamRef.current || !opencvReadyRef.current) return;
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

  const attachLiveScanInterval = () => {
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
    }
    setIsLiveScanning(true);
    scanTimerRef.current = setInterval(async () => {
      // Fixed slots: classify both slot crops; auto-commit when classifier agrees across several scans.
      if (ENABLE_OPENCV_MULTICARD && !USE_PIPELINE && opencvReady) {
        const canvas = drawCurrentFrameToCanvas();
        if (!canvas) return;

        if (dealerSlotRef.current && playerSlotRef.current) {
          const th = tuningRef.current.lockThreshold;
          const need = Math.max(1, Math.round(tuningRef.current.slotAgreeFrames ?? 2));

          if (!inFlightRef.current) {
            const d64 = cropSlotToBase64(canvas, dealerSlotRef.current);
            const p64 = cropSlotToBase64(canvas, playerSlotRef.current);
            if (d64 && p64) {
              const data = await runBatchPrediction([d64, p64]);
              const preds = data?.predictions ?? [];
              const dTop = preds[0]?.top_prediction;
              const pTop = preds[1]?.top_prediction;
              const dLabel = dTop?.label ?? "";
              const dProb = dTop?.probability ?? 0;
              const pLabel = pTop?.label ?? "";
              const pProb = pTop?.probability ?? 0;

              setSlotPreview({
                dealer: dLabel ? { label: dLabel, probability: dProb } : null,
                player: pLabel ? { label: pLabel, probability: pProb } : null,
              });

              const ag = slotAgreeRef.current;
              const last = lastCommittedSlotLabelRef.current;
              const dCommit = trySlotClassifierAgreement(ag, last, "dealer", dLabel, dProb, th, need);
              const pCommit = trySlotClassifierAgreement(ag, last, "player", pLabel, pProb, th, need);

              const dealerSnap = committedDealerRef.current;
              const dSnapSum = handSummary(dealerSnap);
              const playerMayCommitCard = dealerSnap.length < 2;
              const dealerMayCommitCard = dSnapSum.status !== "bust" && dSnapSum.total < 17;

              if (dCommit && dealerMayCommitCard) {
                const entry = {
                  id: nextHandCardIdRef.current++,
                  label: dCommit.label,
                  probability: dCommit.probability,
                };
                lastCommittedSlotLabelRef.current.dealer = dCommit.label;
                setCommittedDealer((prev) => [...prev, entry]);
                setRunningCount((v) => v + hiLoDelta(dCommit.label));
                setCardsSeen((v) => v + 1);
              }
              if (pCommit && playerMayCommitCard) {
                const entry = {
                  id: nextHandCardIdRef.current++,
                  label: pCommit.label,
                  probability: pCommit.probability,
                };
                lastCommittedSlotLabelRef.current.player = pCommit.label;
                setCommittedPlayer((prev) => [...prev, entry]);
                setRunningCount((v) => v + hiLoDelta(pCommit.label));
                setCardsSeen((v) => v + 1);
              }
            }
          }
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
          return;
        }
        return;
      }

      const canvas = drawCurrentFrameToCanvas();
      if (!canvas) return;
      const imageBase64 = canvas.toDataURL("image/jpeg", 0.9).split(",")[1] ?? "";
      if (!imageBase64) return;
      resetLiveTrackingState();
      if (USE_PIPELINE) {
        await runPipelinePrediction(imageBase64, {
          w: canvas.width,
          h: canvas.height,
        });
      } else {
        await runPrediction(imageBase64);
      }
    }, LIVE_SCAN_INTERVAL_MS);
  };

  const startLiveScan = () => {
    if (!isCameraOn) {
      setError("Start camera first.");
      return;
    }
    if (awaitTableClear) {
      setError("Remove the old cards from the frame, then tap Continue below.");
      return;
    }
    setError("");
    attachLiveScanInterval();
  };

  const continueAfterTableClear = () => {
    setAwaitTableClear(false);
    setLastRoundOutcome(null);
    setError("");
    if (!isCameraOn) return;
    attachLiveScanInterval();
  };

  const stopLiveScan = () => {
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
    setSlotPreview(emptySlotPreview());
    slotAgreeRef.current = {
      dealer: { streak: 0, candidate: null },
      player: { streak: 0, candidate: null },
    };
    setIsLiveScanning(false);
  };

  const dealerCards = committedDealer;
  const playerCards = committedPlayer;
  const scanningDealer = [];
  const scanningPlayer = [];
  for (const card of liveCards) {
    const centerY = (card.ny ?? 0) + (card.nh ?? 0) / 2;
    (centerY < 0.5 ? scanningDealer : scanningPlayer).push(card);
  }
  const slotMode = ENABLE_OPENCV_MULTICARD && !USE_PIPELINE;
  const leader = currentRoundLeader(dealerCards, playerCards);
  const decksRemaining = Math.max(0.25, numDecks - (cardsSeen + burnedCards) / 52);
  const trueCount = runningCount / decksRemaining;
  const suggestion = suggestAction({
    dealer: dealerCards,
    player: playerCards,
    trueCount,
  });
  const suggestionTone = (() => {
    const s = suggestion.toLowerCase();
    if (s.includes("push")) return "suggestionPush";
    if (s.includes("stand") || s.includes("hold")) return "suggestionHold";
    return "";
  })();
  const slotPrompt = (() => {
    if (!slotMode || !isLiveScanning) return null;
    if (!dealerSlot) {
      return "Hold one dealer card in the upper half until it locks (green/yellow box). That fixes the dealer slot.";
    }
    if (!playerSlot) {
      return "Dealer slot is set. Hold one player card in the lower half until it locks.";
    }
    return "Slots are set. The scanner adds cards when it agrees on a read. After the dealer has two cards you cannot add player cards; the dealer then draws until hard 17 or higher (will take another card on soft 17).";
  })();
  const roundOverlayText = (() => {
    if (!awaitTableClear || !lastRoundOutcome) return "";
    if (lastRoundOutcome === "player") return "Player wins";
    if (lastRoundOutcome === "dealer") return "Dealer wins";
    return "Push";
  })();
  const roundOverlayClass = `roundOverlay ${
    lastRoundOutcome === "player" ? "player" : lastRoundOutcome === "dealer" ? "dealer" : "push"
  }`;

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

  return (
    <RequireAuth>
      <div className="page">
        <div className="container">
          <NavBar />

        <section className="panel">
          <h1>Blackjack Card Classifier</h1>
          <p className="muted">
            Practice blackjack with your camera: lock dealer and player slots on the felt, let the model read cards as
            you play, and get basic-strategy hints with a Hi-Lo running count.
          </p>

          <details className="settingsDetails">
            <summary className="settingsSummary">Settings</summary>
            <div className="form settingsFormInner">
              <label className="label">Decks in shoe</label>
              <select value={numDecks} onChange={(e) => setNumDecks(Number(e.target.value))}>
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={4}>4</option>
                <option value={6}>6</option>
                <option value={8}>8</option>
              </select>

              <label className="label">Cards burned / thrown out at shuffle: {burnedCards}</label>
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
            </div>
          </details>

          <div className="form">
            <div className="buttonRow" style={{ marginTop: "0.75rem" }}>
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
                <button type="button" disabled={!isCameraOn || awaitTableClear} onClick={startLiveScan}>
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

            {awaitTableClear && (
              <div className="tableClearBanner">
                <p className="tableClearText">
                  Round finished — take the cards off the table so the camera won&apos;t pick them up again.
                </p>
                <button type="button" className="tableClearBtn" onClick={continueAfterTableClear}>
                  Continue
                </button>
              </div>
            )}
          </div>
        </section>

        <section className="cameraLayout">
          <article className="panel sidePanel">
            <h2>Round info</h2>
            {error && <p className="error">{error}</p>}
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
              <p className={`suggestionBadge ${suggestionTone}`}>
                <strong>{suggestion}</strong>
              </p>
              <p className="muted small">
                Basic strategy with a few count-based deviations (Hi-Lo). Not financial advice.
              </p>
            </div>
            <div className="handBlock">
              <h3>Rounds won</h3>
              <p>
                Player: <strong>{roundsWon.player}</strong> · Dealer:{" "}
                <strong>{roundsWon.dealer}</strong> · Pushes:{" "}
                <strong>{roundsWon.push}</strong>
              </p>
              <p className="muted small">
                      A round ends when someone busts, or when the dealer has at least two cards and stands on hard 17+ (dealer hits soft 17).
                Then hands reset for the next round (slots cleared so you can re-anchor). The running count is
                unchanged.
              </p>
            </div>
            <div className="handBlock">
              <h3>Leaderboard (Top 10)</h3>
              {leaderboardLoading && <p className="muted small">Loading leaderboard…</p>}
              {!leaderboardLoading && leaderboardError && (
                <p className="muted small">Leaderboard unavailable: {leaderboardError}</p>
              )}
              {!leaderboardLoading && !leaderboardError && leaderboard.length === 0 && (
                <p className="muted small">No entries yet.</p>
              )}
              {!leaderboardLoading && !leaderboardError && leaderboard.length > 0 && (
                <ol className="leaderboardList">
                  {leaderboard.map((row) => (
                    <li key={row.user_id}>
                      <span>{row.username || "Unknown"}</span>
                      <span className="muted small">
                        W {row.wins} · L {row.losses} · P {row.pushes} · WR {row.win_rate_pct}%
                      </span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </article>

          <article className="panel cameraMainPanel">
            <h2>Table view</h2>
            <div className="liveContainer">
                <video ref={videoRef} className="preview livePreview" autoPlay playsInline muted />
                <canvas ref={canvasRef} className="hiddenCanvas" />
                {roundOverlayText && <div className={roundOverlayClass}>{roundOverlayText}</div>}
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
          </article>

          <article className="panel sidePanel">
            <h2>Hands & count</h2>
            {slotPrompt && (
              <div className="handBlock">
                <h3>Live scan steps</h3>
                <p className="muted small">{slotPrompt}</p>
                {dealerSlot && playerSlot && (
                  <p className="muted small" style={{ marginTop: "0.35rem" }}>
                    The table view shows each slot&apos;s current readout each scan.
                  </p>
                )}
              </div>
            )}
            {renderHand("Dealer's cards", dealerCards, scanningDealer)}
            {renderHand("Player's cards", playerCards, scanningPlayer)}
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
          </article>
        </section>
      </div>
    </div>
    </RequireAuth>
  );
}

export default App;
