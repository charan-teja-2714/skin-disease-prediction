"""
Consultation API: doctor directory, booking, and chat endpoints.
"""

import os
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from backend.database.db import (
    get_doctors,
    get_doctor,
    get_cities,
    get_bookings,
    get_booking,
    create_booking,
    get_chat_messages,
    add_chat_message,
)
from backend.utils.chat_simulator import generate_doctor_reply

router = APIRouter(tags=["consultation"])


# ---- Request models ----

class BookingRequest(BaseModel):
    patient_name: str
    doctor_id: str
    consultation_type: str  # "online" or "offline"
    date: str
    time_slot: str
    reason: Optional[str] = ""
    disease_code: Optional[str] = ""
    severity_tier: Optional[str] = ""


class ChatMessageRequest(BaseModel):
    sender: str  # "patient"
    message: str


# ---- Doctor endpoints ----

@router.get("/doctors")
def list_doctors(
    city: Optional[str] = None,
    condition: Optional[str] = None,
    consultation_type: Optional[str] = None,
):
    """List doctors, optionally filtered by city, condition, or consultation type."""
    return get_doctors(city=city, condition=condition, consultation_type=consultation_type)


@router.get("/doctors/cities")
def list_cities():
    """List all available cities for filtering."""
    return get_cities()


@router.get("/doctors/{doctor_id}")
def doctor_detail(doctor_id: str):
    doctor = get_doctor(doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doctor


# ---- Booking endpoints ----

@router.post("/bookings")
def create_new_booking(req: BookingRequest):
    doctor = get_doctor(req.doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if req.consultation_type == "online" and not doctor["available_online"]:
        raise HTTPException(status_code=400, detail="This doctor is not available for online consultation")
    if req.consultation_type == "offline" and not doctor["available_offline"]:
        raise HTTPException(status_code=400, detail="This doctor is not available for offline consultation")

    booking = create_booking(req.model_dump())
    return booking


@router.get("/bookings")
def list_bookings(patient_name: Optional[str] = None):
    return get_bookings(patient_name=patient_name)


@router.get("/bookings/{booking_id}")
def booking_detail(booking_id: str):
    booking = get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    # Attach doctor info
    doctor = get_doctor(booking["doctor_id"])
    booking["doctor"] = doctor
    return booking


# ---- Chat endpoints ----

@router.get("/bookings/{booking_id}/chat")
def chat_history(booking_id: str):
    booking = get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return get_chat_messages(booking_id)


@router.post("/bookings/{booking_id}/chat")
def send_chat_message(booking_id: str, req: ChatMessageRequest):
    booking = get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if booking["consultation_type"] != "online":
        raise HTTPException(status_code=400, detail="Chat is only available for online consultations")

    # Save patient message
    patient_msg = add_chat_message(booking_id, req.sender, req.message)

    # Generate and save doctor auto-reply
    existing = get_chat_messages(booking_id)
    is_first = len([m for m in existing if m["sender"] == "doctor"]) == 0

    doctor_text = generate_doctor_reply(
        message=req.message,
        disease_code=booking.get("disease_code", ""),
        severity_tier=booking.get("severity_tier", "low"),
        is_first=is_first,
    )
    doctor_msg = add_chat_message(booking_id, "doctor", doctor_text)

    return {"patient_message": patient_msg, "doctor_reply": doctor_msg}
