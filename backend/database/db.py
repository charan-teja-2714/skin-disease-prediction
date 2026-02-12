"""
JSON file-based data access layer for doctors, bookings, and chats.
"""

import json
import os
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))


def _read(filename):
    path = os.path.join(DB_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(filename, data):
    path = os.path.join(DB_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


# ---- Doctors ----

def get_doctors(city=None, condition=None, consultation_type=None):
    doctors = _read("doctors.json")
    if city:
        doctors = [d for d in doctors if d["city"].lower() == city.lower()]
    if condition:
        doctors = [d for d in doctors if condition.upper() in d["conditions"]]
    if consultation_type == "online":
        doctors = [d for d in doctors if d["available_online"]]
    elif consultation_type == "offline":
        doctors = [d for d in doctors if d["available_offline"]]
    return doctors


def get_doctor(doctor_id):
    return next((d for d in _read("doctors.json") if d["id"] == doctor_id), None)


def get_cities():
    doctors = _read("doctors.json")
    return sorted(set(d["city"] for d in doctors))


# ---- Bookings ----

def get_bookings(patient_name=None):
    bookings = _read("bookings.json")
    if patient_name:
        bookings = [
            b for b in bookings
            if b["patient_name"].lower() == patient_name.lower()
        ]
    return bookings


def get_booking(booking_id):
    return next((b for b in _read("bookings.json") if b["id"] == booking_id), None)


def create_booking(data):
    bookings = _read("bookings.json")
    booking = {
        "id": f"book_{int(datetime.now().timestamp() * 1000)}",
        "patient_name": data["patient_name"],
        "doctor_id": data["doctor_id"],
        "consultation_type": data["consultation_type"],
        "date": data["date"],
        "time_slot": data["time_slot"],
        "reason": data.get("reason", ""),
        "disease_code": data.get("disease_code", ""),
        "severity_tier": data.get("severity_tier", ""),
        "status": "confirmed",
        "created_at": datetime.now().isoformat(),
    }
    bookings.append(booking)
    _write("bookings.json", bookings)
    return booking


# ---- Chat ----

def get_chat_messages(booking_id):
    chats = _read("chats.json")
    return [c for c in chats if c["booking_id"] == booking_id]


def add_chat_message(booking_id, sender, message):
    chats = _read("chats.json")
    msg = {
        "id": f"msg_{int(datetime.now().timestamp() * 1000)}",
        "booking_id": booking_id,
        "sender": sender,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    chats.append(msg)
    _write("chats.json", chats)
    return msg
