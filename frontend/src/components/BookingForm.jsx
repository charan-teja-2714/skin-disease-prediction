import { useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

const CONDITION_LABELS = {
  MEL: "Melanoma", NV: "Melanocytic Nevus", BCC: "Basal Cell Carcinoma",
  AKIEC: "Actinic Keratosis", BKL: "Benign Keratosis", DF: "Dermatofibroma",
  VASC: "Vascular Lesion",
};

const s = {
  card: {
    background: "white",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow-lg)",
    padding: "24px",
  },
  title: { fontSize: "17px", fontWeight: 700, marginBottom: "16px" },
  doctorInfo: {
    padding: "12px",
    background: "var(--gray-50)",
    borderRadius: "var(--radius)",
    marginBottom: "16px",
    fontSize: "14px",
  },
  label: {
    display: "block",
    fontSize: "13px",
    fontWeight: 600,
    color: "var(--gray-700)",
    marginBottom: "4px",
    marginTop: "12px",
  },
  input: {
    width: "100%",
    padding: "10px 12px",
    border: "1px solid var(--gray-300)",
    borderRadius: "var(--radius)",
    fontSize: "14px",
    boxSizing: "border-box",
  },
  textarea: {
    width: "100%",
    padding: "10px 12px",
    border: "1px solid var(--gray-300)",
    borderRadius: "var(--radius)",
    fontSize: "14px",
    minHeight: "70px",
    resize: "vertical",
    boxSizing: "border-box",
  },
  btnRow: { display: "flex", gap: "10px", marginTop: "20px" },
  btn: (primary) => ({
    flex: 1,
    padding: "12px",
    border: primary ? "none" : "1px solid var(--gray-300)",
    borderRadius: "var(--radius)",
    background: primary ? "var(--primary)" : "white",
    color: primary ? "white" : "var(--gray-700)",
    fontWeight: 600,
    fontSize: "15px",
    cursor: "pointer",
  }),
  error: {
    padding: "10px",
    background: "#fef2f2",
    color: "#991b1b",
    borderRadius: "var(--radius)",
    fontSize: "13px",
    marginTop: "10px",
  },
};

function BookingForm({ doctor, consultationType, predictionContext, patientName, onBookingCreated, onBack }) {
  const slots = consultationType === "online" ? doctor.online_slots : doctor.offline_slots;
  const fee = consultationType === "online" ? doctor.consultation_fee_online : doctor.consultation_fee_offline;

  const diseaseCode = predictionContext?.disease_code || "";
  const diseaseLabel = CONDITION_LABELS[diseaseCode] || diseaseCode;
  const confidence = predictionContext?.confidence || 0;

  const defaultReason = diseaseCode
    ? `Follow-up for predicted ${diseaseLabel} (confidence: ${(confidence * 100).toFixed(1)}%)`
    : "";

  const [name, setName] = useState(patientName || "");
  const [date, setDate] = useState("");
  const [timeSlot, setTimeSlot] = useState(slots[0] || "");
  const [reason, setReason] = useState(defaultReason);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  // Min date = today
  const today = new Date().toISOString().split("T")[0];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) return setError("Please enter your name.");
    if (!date) return setError("Please select a date.");
    if (!timeSlot) return setError("Please select a time slot.");

    setSubmitting(true);
    setError(null);

    try {
      const res = await axios.post(`${API_URL}/bookings`, {
        patient_name: name.trim(),
        doctor_id: doctor.id,
        consultation_type: consultationType,
        date,
        time_slot: timeSlot,
        reason,
        disease_code: diseaseCode,
        severity_tier: predictionContext?.severity_tier || "",
      });
      onBookingCreated(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to create booking.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={s.card}>
      <h3 style={s.title}>
        Book {consultationType === "online" ? "Online Consultation" : "Clinic Appointment"}
      </h3>

      <div style={s.doctorInfo}>
        <strong>{doctor.name}</strong> — {doctor.specialty}
        <br />
        {doctor.hospital}, {doctor.city}
        <br />
        <span style={{ color: "var(--primary)", fontWeight: 600 }}>
          Fee: Rs.{fee}
        </span>
      </div>

      <form onSubmit={handleSubmit}>
        <label style={s.label}>Patient Name</label>
        <input style={s.input} value={name} onChange={(e) => setName(e.target.value)} />

        <label style={s.label}>Preferred Date</label>
        <input style={s.input} type="date" min={today} value={date} onChange={(e) => setDate(e.target.value)} />

        <label style={s.label}>Time Slot</label>
        <select style={s.input} value={timeSlot} onChange={(e) => setTimeSlot(e.target.value)}>
          {slots.map((sl) => (
            <option key={sl} value={sl}>{sl}</option>
          ))}
        </select>

        <label style={s.label}>Reason for Consultation</label>
        <textarea style={s.textarea} value={reason} onChange={(e) => setReason(e.target.value)} />

        {error && <div style={s.error}>{error}</div>}

        <div style={s.btnRow}>
          <button type="button" style={s.btn(false)} onClick={onBack}>
            Back
          </button>
          <button type="submit" style={s.btn(true)} disabled={submitting}>
            {submitting ? "Booking..." : "Confirm Booking"}
          </button>
        </div>
      </form>
    </div>
  );
}

export default BookingForm;
