import axios from "axios";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3000";
const predictPath = import.meta.env.VITE_PREDICT_PATH ?? "/predict";

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
