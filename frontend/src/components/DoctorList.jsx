import { useState, useEffect } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

const CONDITION_LABELS = {
  MEL: "Melanoma",
  NV: "Melanocytic Nevus",
  BCC: "Basal Cell Carcinoma",
  AKIEC: "Actinic Keratosis",
  BKL: "Benign Keratosis",
  DF: "Dermatofibroma",
  VASC: "Vascular Lesion",
};

const s = {
  filters: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
    marginBottom: "20px",
  },
  select: {
    padding: "8px 12px",
    border: "1px solid var(--gray-300)",
    borderRadius: "var(--radius)",
    fontSize: "14px",
    background: "white",
    minWidth: "160px",
  },
  card: {
    background: "white",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow)",
    padding: "20px",
    marginBottom: "14px",
    border: "1px solid var(--gray-200)",
  },
  topRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: "8px",
  },
  name: { fontSize: "17px", fontWeight: 700, color: "var(--gray-900)" },
  specialty: { fontSize: "13px", color: "var(--gray-500)", marginTop: "2px" },
  badge: (online) => ({
    display: "inline-block",
    padding: "3px 10px",
    borderRadius: "12px",
    fontSize: "11px",
    fontWeight: 600,
    background: online ? "#dcfce7" : "#fef9c3",
    color: online ? "#166534" : "#854d0e",
    marginRight: "6px",
  }),
  detail: { fontSize: "13px", color: "var(--gray-700)", margin: "4px 0" },
  rating: { color: "#f59e0b", fontWeight: 600, fontSize: "14px" },
  fee: { fontSize: "13px", color: "var(--gray-500)" },
  btnRow: { display: "flex", gap: "8px", marginTop: "12px" },
  btn: (primary) => ({
    flex: 1,
    padding: "10px",
    border: primary ? "none" : "1px solid var(--primary)",
    borderRadius: "var(--radius)",
    background: primary ? "var(--primary)" : "white",
    color: primary ? "white" : "var(--primary)",
    fontWeight: 600,
    fontSize: "14px",
    cursor: "pointer",
  }),
  empty: {
    textAlign: "center",
    padding: "40px",
    color: "var(--gray-500)",
    fontSize: "15px",
  },
};

function DoctorList({ predictionContext, onSelectDoctor }) {
  const [doctors, setDoctors] = useState([]);
  const [cities, setCities] = useState([]);
  const [filterCity, setFilterCity] = useState("");
  const [filterType, setFilterType] = useState("");
  const [loading, setLoading] = useState(true);

  const conditionCode = predictionContext?.disease_code || "";

  useEffect(() => {
    axios.get(`${API_URL}/doctors/cities`).then((res) => setCities(res.data)).catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    const params = {};
    if (conditionCode) params.condition = conditionCode;
    if (filterCity) params.city = filterCity;
    if (filterType) params.consultation_type = filterType;

    axios
      .get(`${API_URL}/doctors`, { params })
      .then((res) => setDoctors(res.data))
      .catch(() => setDoctors([]))
      .finally(() => setLoading(false));
  }, [conditionCode, filterCity, filterType]);

  return (
    <div>
      <h3 style={{ fontSize: "17px", fontWeight: 700, marginBottom: "4px" }}>
        Recommended Doctors
      </h3>
      {conditionCode && (
        <p style={{ fontSize: "13px", color: "var(--gray-500)", marginBottom: "16px" }}>
          Showing specialists for: <strong>{CONDITION_LABELS[conditionCode] || conditionCode}</strong>
        </p>
      )}

      {/* Filters */}
      <div style={s.filters}>
        <select style={s.select} value={filterCity} onChange={(e) => setFilterCity(e.target.value)}>
          <option value="">All Cities</option>
          {cities.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select style={s.select} value={filterType} onChange={(e) => setFilterType(e.target.value)}>
          <option value="">Online & Offline</option>
          <option value="online">Online Only</option>
          <option value="offline">Offline Only</option>
        </select>
      </div>

      {/* Doctor list */}
      {loading ? (
        <div style={s.empty}>Loading doctors...</div>
      ) : doctors.length === 0 ? (
        <div style={s.empty}>No doctors found matching your filters. Try broadening your search.</div>
      ) : (
        doctors.map((doc) => (
          <div key={doc.id} style={s.card}>
            <div style={s.topRow}>
              <div>
                <div style={s.name}>{doc.name}</div>
                <div style={s.specialty}>{doc.specialty}</div>
              </div>
              <div style={s.rating}>
                {"*".repeat(Math.round(doc.rating))} {doc.rating}
              </div>
            </div>

            <div style={s.detail}>{doc.hospital}, {doc.city}</div>
            <div style={s.detail}>{doc.address}</div>
            <div style={s.detail}>{doc.experience_years} years experience</div>

            <div style={{ marginTop: "8px" }}>
              {doc.available_online && <span style={s.badge(true)}>Online Available</span>}
              {doc.available_offline && <span style={s.badge(false)}>In-Clinic</span>}
            </div>

            <div style={s.btnRow}>
              {doc.available_online && (
                <button
                  style={s.btn(true)}
                  onClick={() => onSelectDoctor(doc, "online")}
                >
                  Book Online ({doc.consultation_fee_online > 0 ? `Rs.${doc.consultation_fee_online}` : "Free"})
                </button>
              )}
              {doc.available_offline && (
                <button
                  style={s.btn(!doc.available_online)}
                  onClick={() => onSelectDoctor(doc, "offline")}
                >
                  Book Appointment ({doc.consultation_fee_offline > 0 ? `Rs.${doc.consultation_fee_offline}` : "Free"})
                </button>
              )}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

export default DoctorList;
