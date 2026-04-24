import axios from "axios";
import { getAuthIdToken } from "./authToken";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3000";
const predictPath = import.meta.env.VITE_PREDICT_PATH ?? "/predict";
const pipelinePath = (import.meta.env.VITE_PIPELINE_PATH ?? "").trim();
const leaderboardPath = (import.meta.env.VITE_LEADERBOARD_PATH ?? "/leaderboard").trim();
const leaderboardResultPath = (import.meta.env.VITE_LEADERBOARD_RESULT_PATH ?? "/leaderboard/result").trim();

const api = axios.create({
  baseURL: apiBaseUrl,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.request.use((config) => {
  const token = getAuthIdToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export async function predictCard({ imageBase64, topK = 3 }) {
  const payload = {
    image_base64: imageBase64,
    top_k: topK,
  };
  const res = await api.post(predictPath, payload);
  return res.data;
}

export async function predictCards({ imagesBase64, topK = 1 }) {
  const payload = {
    images_base64: imagesBase64,
    top_k: topK,
  };
  const res = await api.post(predictPath, payload);
  return res.data;
}

export async function predictPipeline({
  imageBase64,
  topK = 3,
  detectConf = 0.1,
  detectIou = 0.45,
  maxDetections = 12,
  cropPadding = 0.02,
}) {
  if (!pipelinePath) {
    throw new Error("VITE_PIPELINE_PATH is not set (e.g. /predict-pipeline)");
  }
  const payload = {
    image_base64: imageBase64,
    top_k: topK,
    detect_conf: detectConf,
    detect_iou: detectIou,
    max_detections: maxDetections,
    crop_padding: cropPadding,
  };
  const res = await api.post(pipelinePath, payload);
  return res.data;
}

export async function fetchLeaderboard({ limit = 10, minGames = 1 } = {}) {
  const res = await api.get(leaderboardPath, {
    params: { limit, min_games: minGames },
  });
  return res.data;
}

export async function submitRoundOutcome({ outcome, username }) {
  const res = await api.post(leaderboardResultPath, {
    outcome,
    username,
  });
  return res.data;
}
