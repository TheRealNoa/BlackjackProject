import { useMemo, useState } from "react";
import NavBar from "./components/NavBar";
import { predictCard } from "./services/api";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(3);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const hasFile = useMemo(() => Boolean(file), [file]);

  const onFileChange = (event) => {
    const picked = event.target.files?.[0];
    if (!picked) return;
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

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
    setIsLoading(true);
    setResult(null);
    setError("");
    try {
      const imageBase64 = await toBase64(file);
      const data = await predictCard({ imageBase64, topK });
      setResult(data);
    } catch (err) {
      const message = err?.response?.data?.error || err.message || "Prediction failed.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <NavBar />

        <section className="panel">
          <h1>Blackjack Card Classifier</h1>
          <p className="muted">
            Upload a card image, send it to the API Gateway endpoint, and view top predictions from SageMaker.
          </p>

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
        </section>

        <section className="grid">
          <article className="panel">
            <h2>Image Preview</h2>
            {!previewUrl && <p className="muted">No image selected.</p>}
            {previewUrl && <img className="preview" src={previewUrl} alt="Uploaded card preview" />}
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
