import axios from "axios";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3000";
const predictPath = import.meta.env.VITE_PREDICT_PATH ?? "/predict";
const pipelinePath = (import.meta.env.VITE_PIPELINE_PATH ?? "").trim();

const api = axios.create({
  baseURL: apiBaseUrl,
  headers: {
    "Content-Type": "application/json",
  },
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
  detectConf = 0.25,
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
