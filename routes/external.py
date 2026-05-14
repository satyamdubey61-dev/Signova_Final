from flask import Blueprint, request, jsonify
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from utils.logger import logger

external_bp = Blueprint('external', __name__)

CONTACT_TO_EMAIL = os.environ.get("CONTACT_TO_EMAIL", "info.evolvora@gmail.com")
CONTACT_FROM_EMAIL = os.environ.get("CONTACT_FROM_EMAIL", "info.evolvora@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

@external_bp.route("/contact", methods=["POST"])
def contact():
    if not request.is_json:
        return jsonify({"success": False, "message": "Request must be JSON."}), 400

    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    sender_email = (data.get("email") or "").strip()
    message_body = (data.get("message") or "").strip()

    if not name or not sender_email or not message_body:
        return jsonify({"success": False, "message": "Name, email and message are required."}), 400

    if not GMAIL_APP_PASSWORD:
        return jsonify({
            "success": False,
            "message": "Contact form is not configured. Administrator: set GMAIL_APP_PASSWORD.",
        }), 503

    subject = f"SignifyConnect – Get In Touch from {name}"
    body = f"You received a message from the SignifyConnect contact form.\n\nName: {name}\nEmail: {sender_email}\n\nMessage:\n{message_body}\n"
    
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("SignifyConnect Contact", CONTACT_FROM_EMAIL))
    msg["To"] = CONTACT_TO_EMAIL
    msg["Reply-To"] = sender_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(CONTACT_FROM_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(CONTACT_FROM_EMAIL, CONTACT_TO_EMAIL, msg.as_string())
            logger.info(f"Contact email sent successfully from {sender_email}")
    except Exception as e:
        logger.error(f"Failed to send contact email: {e}")
        return jsonify({"success": False, "message": f"Failed to send email: {str(e)}"}), 500

    return jsonify({"success": True, "message": "Message sent successfully. We'll get back to you soon."})
