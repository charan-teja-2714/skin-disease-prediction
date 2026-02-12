import { useState, useEffect } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

const s = {
  card: {
    background: "white",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow)",
    padding: "16px 20px",
    marginBottom: "12px",
    border: "1px solid var(--gray-200)",
  },
  topRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "6px",
  },
  doctorName: { fontWeight: 700, fontSize: "15px" },
  statusBadge: (status) => ({
    padding: "3px 10px",
    borderRadius: "12px",
    fontSize: "11px",
    fontWeight: 600,
    background: status === "confirmed" ? "#dcfce7" : "#fef9c3",
    color: status === "confirmed" ? "#166534" : "#854d0e",
  }),
  typeBadge: (type) => ({
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: "10px",
    fontSize: "11px",
    fontWeight: 600,
    background: type === "online" ? "#dbeafe" : "#fef3c7",
    color: type === "online" ? "#1e40af" : "#92400e",
    marginRight: "6px",
  }),
  detail: { fontSize: "13px", color: "var(--gray-500)", margin: "3px 0" },
  chatBtn: {
    marginTop: "10px",
    padding: "8px 16px",
    background: "var(--primary)",
    color: "white",
    border: "none",
    borderRadius: "var(--radius)",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
  },
  empty: {
    textAlign: "center",
    padding: "40px",
    color: "var(--gray-500)",
    fontSize: "14px",
  },
};

function BookingList({ patientName, onOpenChat }) {
  const [bookings, setBookings] = useState([]);
  const [doctors, setDoctors] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!patientName) return;
    setLoading(true);
    Promise.all([
      axios.get(`${API_URL}/bookings`, { params: { patient_name: patientName } }),
      axios.get(`${API_URL}/doctors`),
    ])
      .then(([bRes, dRes]) => {
        setBookings(bRes.data.reverse());
        const docMap = {};
        dRes.data.forEach((d) => { docMap[d.id] = d; });
        setDoctors(docMap);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [patientName]);

  if (loading) return <div style={s.empty}>Loading bookings...</div>;
  if (bookings.length === 0) return <div style={s.empty}>No bookings yet. Book a consultation from the Doctors tab.</div>;

  return (
    <div>
      <h3 style={{ fontSize: "17px", fontWeight: 700, marginBottom: "14px" }}>My Bookings</h3>
      {bookings.map((b) => {
        const doc = doctors[b.doctor_id];
        return (
          <div key={b.id} style={s.card}>
            <div style={s.topRow}>
              <span style={s.doctorName}>{doc ? doc.name : b.doctor_id}</span>
              <span style={s.statusBadge(b.status)}>{b.status}</span>
            </div>
            <div>
              <span style={s.typeBadge(b.consultation_type)}>
                {b.consultation_type === "online" ? "Online" : "In-Clinic"}
              </span>
              {doc && <span style={{ fontSize: "13px", color: "var(--gray-500)" }}>{doc.hospital}, {doc.city}</span>}
            </div>
            <div style={s.detail}>Date: {b.date} | Slot: {b.time_slot}</div>
            {b.reason && <div style={s.detail}>Reason: {b.reason}</div>}
            {b.consultation_type === "online" && (
              <button
                style={s.chatBtn}
                onClick={() => onOpenChat(b.id, doc ? doc.name : "Doctor")}
              >
                Open Chat
              </button>
            )}
            {b.consultation_type === "offline" && doc && (
              <div style={{ ...s.detail, marginTop: "8px" }}>
                Contact: {doc.contact} | {doc.address}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default BookingList;
