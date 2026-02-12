import { useState, useEffect } from "react";
import "./App.css";
import UploadForm from "./components/UploadForm";
import ConsultationPage from "./components/ConsultationPage";

function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [consultationContext, setConsultationContext] = useState(null);
  const [patientName, setPatientName] = useState(() => localStorage.getItem("patientName") || "");
  const [showNameModal, setShowNameModal] = useState(false);
  const [nameInput, setNameInput] = useState("");

  useEffect(() => {
    if (!patientName) setShowNameModal(true);
  }, [patientName]);

  const handleSaveName = () => {
    const name = nameInput.trim();
    if (!name) return;
    localStorage.setItem("patientName", name);
    setPatientName(name);
    setShowNameModal(false);
  };

  const handleBookConsultation = (context) => {
    setConsultationContext(context);
    setActiveTab("consult");
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Name modal */}
      {showNameModal && (
        <div className="modal-overlay">
          <div className="modal-card">
            <h3 style={{ marginBottom: "8px" }}>Welcome</h3>
            <p style={{ fontSize: "14px", color: "var(--gray-500)", marginBottom: "16px" }}>
              Please enter your name to continue.
            </p>
            <input
              className="modal-input"
              placeholder="Your full name"
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveName()}
              autoFocus
            />
            <button className="modal-btn" onClick={handleSaveName}>
              Continue
            </button>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="app-title">
          Skin Disease Prediction System
          <small>AI-Assisted Screening with Clinical Explainability</small>
        </div>
        {patientName && (
          <div className="app-user">
            {patientName}
          </div>
        )}
      </header>

      {/* Tab bar */}
      <nav className="tab-bar">
        <button
          className={`tab-btn ${activeTab === "predict" ? "active" : ""}`}
          onClick={() => setActiveTab("predict")}
        >
          Skin Analysis
        </button>
        <button
          className={`tab-btn ${activeTab === "consult" ? "active" : ""}`}
          onClick={() => setActiveTab("consult")}
        >
          Doctor Consultation
        </button>
      </nav>

      {/* Main content */}
      <main className="app-main">
        {activeTab === "predict" && (
          <UploadForm onBookConsultation={handleBookConsultation} />
        )}
        {activeTab === "consult" && (
          <ConsultationPage
            predictionContext={consultationContext}
            patientName={patientName}
          />
        )}
      </main>

      <footer className="app-footer">
        This tool is for screening purposes only and does not replace professional medical advice.
      </footer>
    </div>
  );
}

export default App;
