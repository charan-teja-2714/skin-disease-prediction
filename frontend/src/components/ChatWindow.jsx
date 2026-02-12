import { useState, useEffect, useRef } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "520px",
    background: "white",
    borderRadius: "var(--radius)",
    boxShadow: "var(--shadow-lg)",
    overflow: "hidden",
  },
  header: {
    padding: "14px 20px",
    background: "var(--primary)",
    color: "white",
    fontWeight: 600,
    fontSize: "15px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  backBtn: {
    background: "rgba(255,255,255,0.2)",
    border: "none",
    color: "white",
    padding: "6px 14px",
    borderRadius: "var(--radius)",
    cursor: "pointer",
    fontSize: "13px",
  },
  messagesArea: {
    flex: 1,
    overflowY: "auto",
    padding: "16px",
    background: "var(--gray-50)",
  },
  bubble: (isPatient) => ({
    maxWidth: "75%",
    padding: "10px 14px",
    borderRadius: "14px",
    marginBottom: "10px",
    fontSize: "14px",
    lineHeight: 1.5,
    alignSelf: isPatient ? "flex-end" : "flex-start",
    background: isPatient ? "var(--primary)" : "white",
    color: isPatient ? "white" : "var(--gray-900)",
    boxShadow: isPatient ? "none" : "var(--shadow)",
    borderBottomRightRadius: isPatient ? "4px" : "14px",
    borderBottomLeftRadius: isPatient ? "14px" : "4px",
  }),
  timestamp: (isPatient) => ({
    fontSize: "11px",
    color: isPatient ? "rgba(255,255,255,0.7)" : "var(--gray-500)",
    marginTop: "4px",
  }),
  inputArea: {
    display: "flex",
    padding: "12px",
    borderTop: "1px solid var(--gray-200)",
    background: "white",
    gap: "8px",
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    border: "1px solid var(--gray-300)",
    borderRadius: "20px",
    fontSize: "14px",
    outline: "none",
  },
  sendBtn: {
    padding: "10px 20px",
    background: "var(--primary)",
    color: "white",
    border: "none",
    borderRadius: "20px",
    fontWeight: 600,
    fontSize: "14px",
    cursor: "pointer",
  },
  typing: {
    padding: "8px 14px",
    fontSize: "13px",
    color: "var(--gray-500)",
    fontStyle: "italic",
  },
};

function ChatWindow({ bookingId, doctorName, onBack }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [typing, setTyping] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    axios
      .get(`${API_URL}/bookings/${bookingId}/chat`)
      .then((res) => setMessages(res.data))
      .catch(() => {});
  }, [bookingId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  const handleSend = async () => {
    if (!input.trim() || sending) return;
    const text = input.trim();
    setInput("");
    setSending(true);

    // Optimistic patient bubble
    const tempPatient = {
      id: "temp_" + Date.now(),
      sender: "patient",
      message: text,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempPatient]);

    try {
      const res = await axios.post(`${API_URL}/bookings/${bookingId}/chat`, {
        sender: "patient",
        message: text,
      });

      // Replace temp with real message
      setMessages((prev) => {
        const filtered = prev.filter((m) => m.id !== tempPatient.id);
        return [...filtered, res.data.patient_message];
      });

      // Simulate doctor typing delay
      setTyping(true);
      setTimeout(() => {
        setMessages((prev) => [...prev, res.data.doctor_reply]);
        setTyping(false);
      }, 1200);
    } catch {
      // Keep the temp message but mark failed
      setMessages((prev) =>
        prev.map((m) =>
          m.id === tempPatient.id ? { ...m, message: m.message + " (failed to send)" } : m
        )
      );
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (ts) => {
    try {
      return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return "";
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span>Chat with {doctorName}</span>
        <button style={styles.backBtn} onClick={onBack}>
          Back
        </button>
      </div>

      <div style={{ ...styles.messagesArea, display: "flex", flexDirection: "column" }}>
        {messages.length === 0 && !typing && (
          <div style={{ textAlign: "center", color: "var(--gray-500)", marginTop: "40px", fontSize: "14px" }}>
            Start the conversation by sending a message below.
          </div>
        )}
        {messages.map((msg) => {
          const isPatient = msg.sender === "patient";
          return (
            <div key={msg.id} style={styles.bubble(isPatient)}>
              <div>{msg.message}</div>
              <div style={styles.timestamp(isPatient)}>{formatTime(msg.timestamp)}</div>
            </div>
          );
        })}
        {typing && <div style={styles.typing}>Doctor is typing...</div>}
        <div ref={bottomRef} />
      </div>

      <div style={styles.inputArea}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={sending}
        />
        <button style={styles.sendBtn} onClick={handleSend} disabled={sending || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;
