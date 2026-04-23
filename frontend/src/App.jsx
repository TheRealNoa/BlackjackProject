import { useEffect, useMemo, useRef, useState } from "react";
import NavBar from "./components/NavBar";
import { predictCard } from "./services/api";

function App() {
  const [mode, setMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLiveScanning, setIsLiveScanning] = useState(false);
  const [scanIntervalMs, setScanIntervalMs] = useState(1200);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const scanTimerRef = useRef(null);
  const inFlightRef = useRef(false);

  const hasFile = useMemo(() => Boolean(file), [file]);

  useEffect(() => {
    return () => {
      if (scanTimerRef.current) {
        clearInterval(scanTimerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

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

  const runPrediction = async (imageBase64, source = "upload") => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    if (source === "upload") {
      setIsLoading(true);
    }
    try {
      const data = await predictCard({ imageBase64, topK });
      setResult(data);
      setError("");
    } catch (err) {
      const message = err?.response?.data?.error || err.message || "Prediction failed.";
      setError(message);
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
    setError("");
    const imageBase64 = await toBase64(file);
    await runPrediction(imageBase64, "upload");
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
    } catch (err) {
      setError(`Camera access failed: ${err.message}`);
    }
  };

  const stopCamera = () => {
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
    setIsLiveScanning(false);
    setIsCameraOn(false);
  };

  const captureFrameBase64 = () => {
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
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
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
      const imageBase64 = captureFrameBase64();
      if (!imageBase64) return;
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
            </div>
          )}
        </section>

        <section className="grid">
          <article className="panel">
            <h2>{mode === "live" ? "Live Preview" : "Image Preview"}</h2>
            {mode === "upload" && !previewUrl && <p className="muted">No image selected.</p>}
            {mode === "upload" && previewUrl && <img className="preview" src={previewUrl} alt="Uploaded card preview" />}
            {mode === "live" && (
              <div>
                <video ref={videoRef} className="preview livePreview" autoPlay playsInline muted />
                <canvas ref={canvasRef} className="hiddenCanvas" />
              </div>
            )}
          </article>

          <article className="panel">
            <h2>Predictions</h2>
            {error && <p className="error">{error}</p>}
            {!error && !result && <p className="muted">No prediction yet.</p>}
            {result?.predictions?.[0] && (
              <div>
                <p>
                  <strong>Top:</strong> {result.predictions[0].top_prediction.label} (
                  {(result.predictions[0].top_prediction.probability * 100).toFixed(2)}%)
                </p>
                <ul>
                  {result.predictions[0].top_k.map((item) => (
                    <li key={`${item.class_index}-${item.label}`}>
                      {item.label}: {(item.probability * 100).toFixed(2)}%
                    </li>
                  ))}
                </ul>
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
