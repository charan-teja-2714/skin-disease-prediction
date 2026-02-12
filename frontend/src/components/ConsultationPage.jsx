import { useState } from "react";
import DoctorList from "./DoctorList";
import BookingForm from "./BookingForm";
import BookingList from "./BookingList";
import ChatWindow from "./ChatWindow";

const s = {
  subTabs: {
    display: "flex",
    gap: "0",
    marginBottom: "20px",
    borderBottom: "2px solid var(--gray-200)",
  },
  subTab: (active) => ({
    padding: "10px 20px",
    fontSize: "14px",
    fontWeight: active ? 700 : 400,
    color: active ? "var(--primary)" : "var(--gray-500)",
    background: "none",
    border: "none",
    borderBottom: active ? "2px solid var(--primary)" : "2px solid transparent",
    marginBottom: "-2px",
    cursor: "pointer",
  }),
  confirmed: {
    background: "white",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow-lg)",
    padding: "28px",
    textAlign: "center",
  },
  checkmark: {
    width: "56px",
    height: "56px",
    borderRadius: "50%",
    background: "#dcfce7",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "28px",
    margin: "0 auto 16px",
  },
  confirmedTitle: { fontSize: "18px", fontWeight: 700, color: "var(--success)", marginBottom: "8px" },
  confirmedDetail: { fontSize: "14px", color: "var(--gray-700)", marginBottom: "4px" },
  btn: {
    padding: "12px 24px",
    background: "var(--primary)",
    color: "white",
    border: "none",
    borderRadius: "var(--radius)",
    fontWeight: 600,
    fontSize: "14px",
    cursor: "pointer",
    marginTop: "16px",
    marginRight: "8px",
  },
};

function ConsultationPage({ predictionContext, patientName }) {
  const [subView, setSubView] = useState("doctors"); // "doctors" | "booking" | "confirmed" | "chat" | "my-bookings"
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [consultationType, setConsultationType] = useState("online");
  const [lastBooking, setLastBooking] = useState(null);
  const [chatBookingId, setChatBookingId] = useState(null);
  const [chatDoctorName, setChatDoctorName] = useState("");

  const handleSelectDoctor = (doctor, type) => {
    setSelectedDoctor(doctor);
    setConsultationType(type);
    setSubView("booking");
  };

  const handleBookingCreated = (booking) => {
    setLastBooking(booking);
    setSubView("confirmed");
  };

  const handleOpenChat = (bookingId, doctorName) => {
    setChatBookingId(bookingId);
    setChatDoctorName(doctorName);
    setSubView("chat");
  };

  return (
    <div>
      {/* Sub-tab navigation */}
      {subView !== "chat" && (
        <div style={s.subTabs}>
          <button
            style={s.subTab(subView === "doctors" || subView === "booking" || subView === "confirmed")}
            onClick={() => setSubView("doctors")}
          >
            Find Doctors
          </button>
          <button
            style={s.subTab(subView === "my-bookings")}
            onClick={() => setSubView("my-bookings")}
          >
            My Bookings
          </button>
        </div>
      )}

      {/* Doctor list */}
      {subView === "doctors" && (
        <DoctorList
          predictionContext={predictionContext}
          onSelectDoctor={handleSelectDoctor}
        />
      )}

      {/* Booking form */}
      {subView === "booking" && selectedDoctor && (
        <BookingForm
          doctor={selectedDoctor}
          consultationType={consultationType}
          predictionContext={predictionContext}
          patientName={patientName}
          onBookingCreated={handleBookingCreated}
          onBack={() => setSubView("doctors")}
        />
      )}

      {/* Booking confirmed */}
      {subView === "confirmed" && lastBooking && (
        <div style={s.confirmed}>
          <div style={s.checkmark}>&#10003;</div>
          <div style={s.confirmedTitle}>Booking Confirmed!</div>
          <div style={s.confirmedDetail}>
            <strong>{selectedDoctor?.name}</strong> — {selectedDoctor?.hospital}
          </div>
          <div style={s.confirmedDetail}>
            {lastBooking.consultation_type === "online" ? "Online Consultation" : "Clinic Visit"} on {lastBooking.date} at {lastBooking.time_slot}
          </div>
          <div style={s.confirmedDetail}>Booking ID: {lastBooking.id}</div>

          <div style={{ marginTop: "20px" }}>
            {lastBooking.consultation_type === "online" && (
              <button
                style={s.btn}
                onClick={() => handleOpenChat(lastBooking.id, selectedDoctor?.name || "Doctor")}
              >
                Start Chat
              </button>
            )}
            <button
              style={{ ...s.btn, background: "var(--gray-500)" }}
              onClick={() => setSubView("my-bookings")}
            >
              View My Bookings
            </button>
          </div>
        </div>
      )}

      {/* My bookings */}
      {subView === "my-bookings" && (
        <BookingList
          patientName={patientName}
          onOpenChat={handleOpenChat}
        />
      )}

      {/* Chat */}
      {subView === "chat" && chatBookingId && (
        <ChatWindow
          bookingId={chatBookingId}
          doctorName={chatDoctorName}
          onBack={() => setSubView("my-bookings")}
        />
      )}
    </div>
  );
}

export default ConsultationPage;
