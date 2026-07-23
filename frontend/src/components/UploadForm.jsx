import { useState, useRef } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

/* ── Icon map for recommendation levels ── */
const LEVEL_ICONS = {
  critical: "\u26D4",
  high: "\u26A0\uFE0F",
  moderate: "\uD83D\uDD0D",
  low: "\u2705",
  warning: "\uD83D\uDCE4",
  info: "\u2139\uFE0F",
};

/* ── Styles ── */
const s = {
  card: {
    background: "white",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow-lg)",
    padding: "28px",
    marginBottom: "24px",
  },
  sectionTitle: {
    fontSize: "15px",
    fontWeight: 700,
    color: "var(--gray-700)",
    marginBottom: "12px",
    paddingBottom: "8px",
    borderBottom: "2px solid var(--gray-200)",
  },
  fileInput: {
    width: "100%",
    padding: "12px",
    border: "2px dashed var(--gray-300)",
    borderRadius: "var(--radius)",
    background: "var(--gray-50)",
    cursor: "pointer",
    fontSize: "14px",
  },
  submitBtn: {
    width: "100%",
    padding: "14px",
    background: "var(--primary)",
    color: "white",
    border: "none",
    borderRadius: "var(--radius)",
    fontSize: "16px",
    fontWeight: 600,
    marginTop: "16px",
    transition: "background 0.2s",
  },
  submitBtnDisabled: {
    background: "var(--gray-300)",
    cursor: "not-allowed",
  },
  preview: {
    width: "100%",
    maxHeight: "300px",
    objectFit: "contain",
    borderRadius: "var(--radius)",
    marginTop: "12px",
    border: "1px solid var(--gray-200)",
  },
  metricRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "6px 0",
    fontSize: "14px",
  },
  metricLabel: { color: "var(--gray-500)" },
  metricValue: { fontWeight: 600 },
  bar: {
    height: "8px",
    borderRadius: "4px",
    background: "var(--gray-200)",
    marginTop: "4px",
    overflow: "hidden",
  },
  barFill: (pct, color) => ({
    height: "100%",
    width: `${Math.min(pct * 100, 100)}%`,
    background: color,
    borderRadius: "4px",
    transition: "width 0.4s ease",
  }),
  recBox: (color) => ({
    padding: "16px",
    borderRadius: "var(--radius)",
    borderLeft: `5px solid ${color}`,
    background: `${color}11`,
    marginBottom: "12px",
  }),
  recTitle: (color) => ({
    fontSize: "16px",
    fontWeight: 700,
    color: color,
    marginBottom: "6px",
  }),
  recDesc: {
    fontSize: "14px",
    color: "var(--gray-700)",
    lineHeight: 1.5,
  },
  riskChip: {
    display: "inline-block",
    padding: "3px 10px",
    borderRadius: "12px",
    fontSize: "12px",
    fontWeight: 600,
    background: "#fef2f2",
    color: "#991b1b",
    marginRight: "6px",
    marginBottom: "6px",
  },
  skinNote: {
    fontSize: "13px",
    color: "var(--gray-500)",
    padding: "10px",
    background: "var(--gray-50)",
    borderRadius: "var(--radius)",
    marginTop: "10px",
  },
  errorBox: {
    padding: "12px",
    borderRadius: "var(--radius)",
    background: "#fef2f2",
    color: "#991b1b",
    fontSize: "14px",
    marginTop: "12px",
  },
  spinner: {
    display: "inline-block",
    width: "18px",
    height: "18px",
    border: "3px solid white",
    borderTopColor: "transparent",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
    marginRight: "8px",
    verticalAlign: "middle",
  },
  qualityBanner: (ok) => ({
    padding: "12px 16px",
    borderRadius: "var(--radius)",
    background: ok ? "#f0fdf4" : "#fef2f2",
    border: `1px solid ${ok ? "#bbf7d0" : "#fecaca"}`,
    fontSize: "13px",
    color: ok ? "#166534" : "#991b1b",
    marginBottom: "12px",
  }),
  segToggle: {
    display: "flex",
    gap: "8px",
    marginBottom: "12px",
  },
  segBtn: (active) => ({
    padding: "6px 14px",
    fontSize: "13px",
    fontWeight: active ? 600 : 400,
    background: active ? "var(--primary)" : "var(--gray-100)",
    color: active ? "white" : "var(--gray-700)",
    border: "none",
    borderRadius: "var(--radius)",
    cursor: "pointer",
  }),
  imgContainer: {
    position: "relative",
    width: "100%",
    textAlign: "center",
  },
  segImage: {
    width: "100%",
    maxHeight: "280px",
    objectFit: "contain",
    borderRadius: "var(--radius)",
    border: "1px solid var(--gray-200)",
  },
};

function UploadForm({ onBookConsultation }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Evolution state
  const [prevFile, setPrevFile] = useState(null);
  const [prevPreviewUrl, setPrevPreviewUrl] = useState(null);
  const [showEvolution, setShowEvolution] = useState(false);
  const inputRef = useRef();
  const prevInputRef = useRef();

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handlePrevFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setPrevFile(f);
    setPrevPreviewUrl(URL.createObjectURL(f));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return setError("Please select an image first.");

    const formData = new FormData();
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      let res;
      if (showEvolution && prevFile) {
        // Longitudinal evolution: two-image comparison
        formData.append("file_current", file);
        formData.append("file_previous", prevFile);
        res = await axios.post(`${API_URL}/predict-evolution`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      } else {
        // Standard single-image prediction
        formData.append("file", file);
        res = await axios.post(`${API_URL}/predict`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      }
      setResult(res.data);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        "Failed to connect to the backend. Make sure the server is running.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setPrevFile(null);
    setPrevPreviewUrl(null);
    setResult(null);
    setError(null);
    setShowEvolution(false);
    if (inputRef.current) inputRef.current.value = "";
    if (prevInputRef.current) prevInputRef.current.value = "";
  };

  return (
    <>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      {/* Upload Card */}
      <div style={s.card}>
        <h2 style={{ ...s.sectionTitle, borderBottom: "none", fontSize: "18px" }}>
          Upload Skin Lesion Image
        </h2>
        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/bmp,image/webp"
            onChange={handleFileChange}
            style={s.fileInput}
          />
          {previewUrl && <img src={previewUrl} alt="Preview" style={s.preview} />}

          {/* Evolution toggle */}
          <div style={{ marginTop: "14px" }}>
            <label style={{ fontSize: "14px", color: "var(--gray-700)", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={showEvolution}
                onChange={(e) => setShowEvolution(e.target.checked)}
                style={{ marginRight: "8px" }}
              />
              Compare with previous image (Longitudinal Evolution Tracking)
            </label>
          </div>

          {/* Previous image upload for evolution */}
          {showEvolution && (
            <div style={{ marginTop: "12px", padding: "14px", background: "var(--gray-50)", borderRadius: "var(--radius)" }}>
              <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--gray-700)", marginBottom: "8px" }}>
                Upload Previous Image (earlier photo of the same lesion)
              </div>
              <input
                ref={prevInputRef}
                type="file"
                accept="image/jpeg,image/png,image/bmp,image/webp"
                onChange={handlePrevFileChange}
                style={{ ...s.fileInput, border: "2px dashed var(--gray-300)" }}
              />
              {prevPreviewUrl && (
                <img src={prevPreviewUrl} alt="Previous" style={{ ...s.preview, maxHeight: "200px" }} />
              )}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !file || (showEvolution && !prevFile)}
            style={{
              ...s.submitBtn,
              ...(loading || !file || (showEvolution && !prevFile) ? s.submitBtnDisabled : {}),
            }}
          >
            {loading ? (
              <>
                <span style={s.spinner} />
                {showEvolution ? "Analyzing Evolution..." : "Analyzing..."}
              </>
            ) : showEvolution ? (
              "Analyze with Evolution Comparison"
            ) : (
              "Analyze Lesion"
            )}
          </button>
          {(result || error) && (
            <button
              type="button"
              onClick={handleReset}
              style={{ ...s.submitBtn, background: "var(--gray-500)", marginTop: "8px" }}
            >
              Reset
            </button>
          )}
        </form>
        {error && <div style={s.errorBox}>{error}</div>}
      </div>

      {/* Results */}
      {result && <ResultsPanel result={result} onBookConsultation={onBookConsultation} />}
    </>
  );
}

/* ── Results Panel ── */
function ResultsPanel({ result, onBookConsultation }) {
  const rec = result.recommendation?.primary_recommendation;
  const abcde = result.abcde;
  const seg = result.segmentation;
  const unc = result.uncertainty_details;
  const fairness = result.fairness;
  const quality = result.image_quality;
  const evolution = result.evolution;
  const xai = result.explainability;
  const [segView, setSegView] = useState("overlay");
  const [xaiView, setXaiView] = useState("gradcam_overlay");

  const handleBookClick = () => {
    if (onBookConsultation) {
      onBookConsultation({
        severity_tier: result.recommendation?.severity_tier,
        disease_code: result.recommendation?.disease_code,
        disease_label: result.recommendation?.disease_label,
        confidence: result.confidence,
      });
    }
  };

  return (
    <>
      {/* Image Quality Assessment */}
      {quality && (
        <div style={s.qualityBanner(quality.acceptable)}>
          {quality.acceptable ? (
            <span>Image Quality: Good — analysis is reliable.</span>
          ) : (
            <span>
              Image Quality Issues: {quality.issues.join(", ")}.
              Results may be less reliable.
            </span>
          )}
        </div>
      )}

      {/* Recommendation Banner */}
      {rec && (
        <div style={s.card}>
          <div style={s.sectionTitle}>Recommendation</div>
          <div style={s.recBox(rec.color || "#333")}>
            <div style={s.recTitle(rec.color || "#333")}>
              {LEVEL_ICONS[rec.icon] || ""} {rec.title}
            </div>
            <div style={s.recDesc}>{rec.description}</div>
            {rec.details && (
              <div style={{ ...s.recDesc, marginTop: "8px", fontStyle: "italic" }}>
                {rec.details}
              </div>
            )}
          </div>

          {result.recommendation.risk_factors?.length > 0 && (
            <div style={{ marginTop: "8px" }}>
              <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--gray-700)" }}>
                Risk Factors:{" "}
              </span>
              {result.recommendation.risk_factors.map((rf, i) => (
                <span key={i} style={s.riskChip}>{rf}</span>
              ))}
            </div>
          )}

          {result.recommendation.secondary_recommendations?.map((sr, i) => (
            <div key={i} style={{ ...s.skinNote, marginTop: "8px" }}>
              {sr.message}
            </div>
          ))}

          {result.recommendation.skin_tone_note && (
            <div style={s.skinNote}>{result.recommendation.skin_tone_note}</div>
          )}

          {onBookConsultation && (
            <button
              onClick={handleBookClick}
              style={{
                width: "100%",
                padding: "14px",
                marginTop: "14px",
                background: rec.level === "low" ? "var(--gray-500)" : "var(--primary)",
                color: "white",
                border: "none",
                borderRadius: "var(--radius)",
                fontSize: "15px",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              {rec.level === "low" ? "Consult a Doctor Anyway" : "Book Doctor Consultation"}
            </button>
          )}
        </div>
      )}

      {/* All detailed sections — only if lesion detected */}
      {result.lesion_detected !== false && (
        <>
          {/* Segmentation Visualization */}
          {seg && (
            <div style={s.card}>
              <div style={s.sectionTitle}>Segmentation (U-Net)</div>
              <div style={s.segToggle}>
                <button style={s.segBtn(segView === "overlay")} onClick={() => setSegView("overlay")}>
                  Overlay
                </button>
                <button style={s.segBtn(segView === "mask")} onClick={() => setSegView("mask")}>
                  Mask Only
                </button>
              </div>
              <div style={s.imgContainer}>
                {segView === "overlay" && seg.overlay_base64 && (
                  <img
                    src={`data:image/png;base64,${seg.overlay_base64}`}
                    alt="Segmentation overlay"
                    style={s.segImage}
                  />
                )}
                {segView === "mask" && seg.mask_base64 && (
                  <img
                    src={`data:image/png;base64,${seg.mask_base64}`}
                    alt="Segmentation mask"
                    style={{ ...s.segImage, background: "#000" }}
                  />
                )}
              </div>
              <div style={{ ...s.metricRow, marginTop: "12px" }}>
                <span style={s.metricLabel}>Lesion Coverage</span>
                <span style={s.metricValue}>{(seg.lesion_coverage * 100).toFixed(1)}%</span>
              </div>
              <div style={s.bar}>
                <div style={s.barFill(seg.lesion_coverage, "#10b981")} />
              </div>
            </div>
          )}

          {/* XAI: GradCAM / GradCAM++ / SHAP */}
          {xai && (
            <div style={s.card}>
              <div style={s.sectionTitle}>Visual Explainability (GradCAM / GradCAM++ / SHAP)</div>
              <div style={{ fontSize: "13px", color: "var(--gray-500)", marginBottom: "12px" }}>
                Heatmaps show which regions of the image drove the model's prediction.
                Red = high activation, Blue = low or negative contribution (SHAP).
              </div>

              {/* Tab buttons */}
              <div style={{ ...s.segToggle, flexWrap: "wrap", gap: "6px", marginBottom: "14px" }}>
                {xai.gradcam && !xai.gradcam.error && (
                  <>
                    <button style={s.segBtn(xaiView === "gradcam_overlay")} onClick={() => setXaiView("gradcam_overlay")}>
                      GradCAM Overlay
                    </button>
                    <button style={s.segBtn(xaiView === "gradcam_heat")} onClick={() => setXaiView("gradcam_heat")}>
                      GradCAM Heatmap
                    </button>
                  </>
                )}
                {xai.gradcam_plus && !xai.gradcam_plus.error && (
                  <>
                    <button style={s.segBtn(xaiView === "gcpp_overlay")} onClick={() => setXaiView("gcpp_overlay")}>
                      GradCAM++ Overlay
                    </button>
                    <button style={s.segBtn(xaiView === "gcpp_heat")} onClick={() => setXaiView("gcpp_heat")}>
                      GradCAM++ Heatmap
                    </button>
                  </>
                )}
                {xai.shap && !xai.shap.error && (
                  <button style={s.segBtn(xaiView === "shap")} onClick={() => setXaiView("shap")}>
                    SHAP
                  </button>
                )}
              </div>

              {/* Image display */}
              <div style={s.imgContainer}>
                {xaiView === "gradcam_overlay" && xai.gradcam?.overlay_base64 && (
                  <img src={`data:image/png;base64,${xai.gradcam.overlay_base64}`} alt="GradCAM overlay" style={s.segImage} />
                )}
                {xaiView === "gradcam_heat" && xai.gradcam?.heatmap_base64 && (
                  <img src={`data:image/png;base64,${xai.gradcam.heatmap_base64}`} alt="GradCAM heatmap" style={s.segImage} />
                )}
                {xaiView === "gcpp_overlay" && xai.gradcam_plus?.overlay_base64 && (
                  <img src={`data:image/png;base64,${xai.gradcam_plus.overlay_base64}`} alt="GradCAM++ overlay" style={s.segImage} />
                )}
                {xaiView === "gcpp_heat" && xai.gradcam_plus?.heatmap_base64 && (
                  <img src={`data:image/png;base64,${xai.gradcam_plus.heatmap_base64}`} alt="GradCAM++ heatmap" style={s.segImage} />
                )}
                {xaiView === "shap" && xai.shap?.overlay_base64 && (
                  <img src={`data:image/png;base64,${xai.shap.overlay_base64}`} alt="SHAP explanation" style={s.segImage} />
                )}
              </div>

              {/* Legend */}
              <div style={{ display: "flex", gap: "16px", marginTop: "10px", fontSize: "12px", color: "var(--gray-500)" }}>
                {(xaiView === "gradcam_overlay" || xaiView === "gradcam_heat" || xaiView === "gcpp_overlay" || xaiView === "gcpp_heat") && (
                  <>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      <span style={{ display: "inline-block", width: 12, height: 12, background: "#ff0000", borderRadius: 2 }} /> High activation
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      <span style={{ display: "inline-block", width: 12, height: 12, background: "#0000ff", borderRadius: 2 }} /> Low activation
                    </span>
                  </>
                )}
                {xaiView === "shap" && (
                  <>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      <span style={{ display: "inline-block", width: 12, height: 12, background: "#ff0000", borderRadius: 2 }} /> Supports prediction
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                      <span style={{ display: "inline-block", width: 12, height: 12, background: "#0000ff", borderRadius: 2 }} /> Against prediction
                    </span>
                  </>
                )}
              </div>

              {/* Error fallback */}
              {xai.gradcam?.error && (
                <div style={{ ...s.errorBox, marginTop: "8px" }}>GradCAM error: {xai.gradcam.error}</div>
              )}
              {xai.gradcam_plus?.error && (
                <div style={{ ...s.errorBox, marginTop: "8px" }}>GradCAM++ error: {xai.gradcam_plus.error}</div>
              )}
              {xai.shap?.error && (
                <div style={{ ...s.errorBox, marginTop: "8px" }}>SHAP error: {xai.shap.error}</div>
              )}
            </div>
          )}

          {/* Classification Result */}
          <div style={s.card}>
            <div style={s.sectionTitle}>Classification (EfficientNet)</div>
            <div style={s.metricRow}>
              <span style={s.metricLabel}>Predicted Condition</span>
              <span style={{ ...s.metricValue, fontSize: "16px" }}>
                {result.recommendation?.disease_label || result.disease}
              </span>
            </div>
            <div style={s.metricRow}>
              <span style={s.metricLabel}>Disease Code</span>
              <span style={s.metricValue}>{result.disease}</span>
            </div>
            <div style={s.metricRow}>
              <span style={s.metricLabel}>Confidence</span>
              <span style={s.metricValue}>{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <div style={s.bar}>
              <div style={s.barFill(result.confidence, "var(--primary)")} />
            </div>
            <div style={{ ...s.metricRow, marginTop: "8px" }}>
              <span style={s.metricLabel}>Uncertainty</span>
              <span style={s.metricValue}>{(result.uncertainty * 100).toFixed(1)}%</span>
            </div>
            <div style={s.bar}>
              <div style={s.barFill(result.uncertainty * 5, "var(--warning)")} />
            </div>

            {result.class_probabilities && (
              <div style={{ marginTop: "16px" }}>
                <div style={{ fontSize: "13px", fontWeight: 600, marginBottom: "8px", color: "var(--gray-500)" }}>
                  All Class Probabilities
                </div>
                {Object.entries(result.class_probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([cls, prob]) => (
                    <div key={cls} style={{ marginBottom: "4px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                        <span>{cls}</span>
                        <span style={{ fontWeight: 600 }}>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div style={s.bar}>
                        <div style={s.barFill(prob, cls === result.disease ? "var(--primary)" : "var(--gray-300)")} />
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>

          {/* MC Dropout Uncertainty Details */}
          {unc && (
            <div style={s.card}>
              <div style={s.sectionTitle}>Uncertainty Analysis (MC Dropout)</div>
              <div style={s.metricRow}>
                <span style={s.metricLabel}>Monte Carlo Passes</span>
                <span style={s.metricValue}>{unc.mc_passes}</span>
              </div>
              <div style={s.metricRow}>
                <span style={s.metricLabel}>Predictive Entropy</span>
                <span style={s.metricValue}>
                  {unc.predictive_entropy.toFixed(3)} / {unc.max_entropy.toFixed(3)}
                </span>
              </div>
              <div style={s.bar}>
                <div style={s.barFill(unc.normalized_entropy, unc.normalized_entropy > 0.5 ? "#ef4444" : "#10b981")} />
              </div>
              <div style={{ ...s.metricRow, marginTop: "8px" }}>
                <span style={s.metricLabel}>MC Pass Agreement</span>
                <span style={{
                  ...s.metricValue,
                  color: unc.mc_agreement > 0.8 ? "var(--success)" : unc.mc_agreement > 0.5 ? "var(--warning)" : "var(--danger)",
                }}>
                  {(unc.mc_agreement * 100).toFixed(0)}%
                </span>
              </div>
              <div style={s.bar}>
                <div style={s.barFill(unc.mc_agreement, unc.mc_agreement > 0.7 ? "#10b981" : "#f59e0b")} />
              </div>
              <div style={{
                ...s.qualityBanner(unc.mc_agreement > 0.7 && unc.normalized_entropy < 0.5),
                marginTop: "12px",
              }}>
                {unc.is_uncertain
                  ? "Model shows high uncertainty. Multiple forward passes disagree on the prediction. Professional evaluation recommended."
                  : "Model is confident. Predictions are consistent across Monte Carlo passes."}
              </div>

              {/* Per-class std */}
              {unc.std_per_class && (
                <div style={{ marginTop: "12px" }}>
                  <div style={{ fontSize: "13px", fontWeight: 600, marginBottom: "8px", color: "var(--gray-500)" }}>
                    Uncertainty per Class (Std Dev)
                  </div>
                  {Object.entries(unc.std_per_class)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, std]) => (
                      <div key={cls} style={{ marginBottom: "4px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                          <span>{cls}</span>
                          <span style={{ fontWeight: 600 }}>{(std * 100).toFixed(2)}%</span>
                        </div>
                        <div style={s.bar}>
                          <div style={s.barFill(std * 10, "#8b5cf6")} />
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}

          {/* ABCDE Analysis */}
          {abcde && (
            <div style={s.card}>
              <div style={s.sectionTitle}>ABCDE Morphological Analysis (Explainability)</div>
              <ABCDERow label="A — Asymmetry" value={abcde.asymmetry} max={1} color="#ef4444" />
              <ABCDERow label="B — Border Irregularity" value={abcde.border} max={1} color="#f97316" />
              <ABCDERow label="C — Color Variation" value={abcde.color} max={1} color="#8b5cf6" />
              <ABCDERow label="D — Diameter" value={abcde.diameter_mm} max={10} unit=" mm" color="#0ea5e9" />
              <ABCDERow label="E — Evolution Risk" value={abcde.evolution_risk} max={1} color="#ec4899" />

              {abcde.overall_score != null && (
                <div style={{ ...s.metricRow, marginTop: "12px", paddingTop: "12px", borderTop: "1px solid var(--gray-200)" }}>
                  <span style={{ ...s.metricLabel, fontWeight: 600 }}>Overall ABCDE Score</span>
                  <span style={s.metricValue}>{(abcde.overall_score * 100).toFixed(0)}%</span>
                </div>
              )}

              {abcde.interpretation?.length > 0 && (
                <ul style={{ marginTop: "12px", paddingLeft: "20px", fontSize: "13px", color: "var(--gray-700)" }}>
                  {abcde.interpretation.map((line, i) => (
                    <li key={i} style={{ marginBottom: "4px" }}>{line}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* Evolution / Longitudinal Analysis */}
          {evolution && (
            <div style={s.card}>
              <div style={s.sectionTitle}>Longitudinal Evolution Analysis</div>
              <div style={{
                padding: "12px 16px",
                borderRadius: "var(--radius)",
                background: evolution.alert || evolution["Evolution Alert"] === "YES" ? "#fef2f2" : "#f0fdf4",
                border: `1px solid ${evolution.alert || evolution["Evolution Alert"] === "YES" ? "#fecaca" : "#bbf7d0"}`,
                marginBottom: "14px",
                fontSize: "14px",
                fontWeight: 600,
                color: evolution.alert || evolution["Evolution Alert"] === "YES" ? "#991b1b" : "#166534",
              }}>
                {evolution.alert || evolution["Evolution Alert"] === "YES"
                  ? "Evolution Alert: Significant changes detected between images."
                  : "No significant evolution detected. Lesion appears stable."}
              </div>

              <div style={s.metricRow}>
                <span style={s.metricLabel}>Area Change</span>
                <span style={{
                  ...s.metricValue,
                  color: Math.abs(evolution.area_change_pct || evolution["Area Change (%)"] || 0) > 20 ? "var(--danger)" : "var(--gray-700)",
                }}>
                  {(evolution.area_change_pct ?? evolution["Area Change (%)"] ?? 0).toFixed(1)}%
                </span>
              </div>
              <div style={s.metricRow}>
                <span style={s.metricLabel}>Diameter Change</span>
                <span style={{
                  ...s.metricValue,
                  color: Math.abs(evolution.diameter_change_mm || evolution["Diameter Change (mm)"] || 0) > 2 ? "var(--danger)" : "var(--gray-700)",
                }}>
                  {(evolution.diameter_change_mm ?? evolution["Diameter Change (mm)"] ?? 0).toFixed(2)} mm
                </span>
              </div>
              <div style={s.metricRow}>
                <span style={s.metricLabel}>Color Change</span>
                <span style={{
                  ...s.metricValue,
                  color: Math.abs(evolution.color_change || evolution["Color Change"] || 0) >= 0.3 ? "var(--danger)" : "var(--gray-700)",
                }}>
                  {(evolution.color_change ?? evolution["Color Change"] ?? 0).toFixed(3)}
                </span>
              </div>
            </div>
          )}

          {/* Fairness & Transparency */}
          <div style={s.card}>
            <div style={s.sectionTitle}>Fairness & Transparency</div>
            {result.skin_tone && (
              <div style={s.metricRow}>
                <span style={s.metricLabel}>Detected Skin Tone</span>
                <span style={s.metricValue}>
                  {result.skin_tone.charAt(0).toUpperCase() + result.skin_tone.slice(1)}
                </span>
              </div>
            )}

            {fairness && (
              <>
                <div style={s.metricRow}>
                  <span style={s.metricLabel}>Model Accuracy (for this skin tone)</span>
                  <span style={s.metricValue}>{(fairness.accuracy * 100).toFixed(0)}%</span>
                </div>
                <div style={s.bar}>
                  <div style={s.barFill(fairness.accuracy, fairness.accuracy > 0.8 ? "#10b981" : "#f59e0b")} />
                </div>
                <div style={{ ...s.metricRow, marginTop: "8px" }}>
                  <span style={s.metricLabel}>False Negative Rate (FNR)</span>
                  <span style={{
                    ...s.metricValue,
                    color: fairness.fnr > 0.15 ? "var(--danger)" : "var(--gray-700)",
                  }}>
                    {(fairness.fnr * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={s.bar}>
                  <div style={s.barFill(fairness.fnr * 3, fairness.fnr > 0.15 ? "#ef4444" : "#10b981")} />
                </div>
                <div style={{ ...s.metricRow, marginTop: "8px" }}>
                  <span style={s.metricLabel}>Training Data Representation</span>
                  <span style={{
                    ...s.metricValue,
                    color: fairness.sample_representation === "high" ? "var(--success)" :
                           fairness.sample_representation === "moderate" ? "var(--warning)" : "var(--danger)",
                  }}>
                    {fairness.sample_representation.charAt(0).toUpperCase() + fairness.sample_representation.slice(1)}
                  </span>
                </div>
                <div style={{ ...s.metricRow, marginTop: "4px" }}>
                  <span style={s.metricLabel}>Result Reliability</span>
                  <span style={s.metricValue}>
                    {fairness.reliability.charAt(0).toUpperCase() + fairness.reliability.slice(1)}
                  </span>
                </div>
                <div style={s.skinNote}>{fairness.note}</div>
              </>
            )}

            <div style={s.metricRow}>
              <span style={s.metricLabel}>Confidence Assessment</span>
              <span style={s.metricValue}>{result.recommendation?.confidence_assessment || "N/A"}</span>
            </div>
            <div style={s.metricRow}>
              <span style={s.metricLabel}>Severity Tier</span>
              <span style={{
                ...s.metricValue,
                color: result.recommendation?.severity_tier === "high" ? "var(--danger)" : "var(--success)",
              }}>
                {result.recommendation?.severity_tier?.toUpperCase() || "N/A"}
              </span>
            </div>
          </div>
        </>
      )}
    </>
  );
}

/* ── ABCDE Row Component ── */
function ABCDERow({ label, value, max, unit = "", color }) {
  const pct = Math.min(value / max, 1);
  return (
    <div style={{ marginBottom: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "14px" }}>
        <span style={{ color: "var(--gray-700)" }}>{label}</span>
        <span style={{ fontWeight: 600 }}>
          {typeof value === "number" ? value.toFixed(2) : value}{unit}
        </span>
      </div>
      <div style={s.bar}>
        <div style={s.barFill(pct, color)} />
      </div>
    </div>
  );
}

export default UploadForm;
