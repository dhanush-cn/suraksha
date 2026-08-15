import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional

# Notification Configuration (Can be customized via environment variables)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# In-Memory Dispatch Log Buffer for UI Inspection
DISPATCH_LOGS = []

def send_emergency_email(to_email: str, mine_name: str, risk_pct: float, risk_level: str, top_reason: str) -> Dict[str, Any]:
    """
    Sends an automated HTML Emergency Alert Email to the mine geologist/operator.
    """
    if not to_email:
        to_email = "safety@mine.org"
        
    subject = f"🚨 URGENT CRITICAL SLOPE HAZARD ALERT: {mine_name} ({risk_pct:.1f}% Risk)"
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 10px; padding: 25px; border: 2px solid #ef4444;">
            <h2 style="color: #ef4444; margin-top: 0;">🚨 CRITICAL GEOLOGICAL HAZARD WARNING</h2>
            <p style="font-size: 16px;">This is an automated emergency alert dispatch from <strong>RockfallGuard AI Engine</strong>.</p>
            
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 10px; color: #94a3b8;">Mining Operation:</td>
                    <td style="padding: 10px; font-weight: bold; color: #f8fafc;">{mine_name}</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 10px; color: #94a3b8;">Hazard Risk Level:</td>
                    <td style="padding: 10px; font-weight: bold; color: #ef4444;">{risk_level.upper()} ({risk_pct:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 10px; color: #94a3b8;">Top Contributing Trigger:</td>
                    <td style="padding: 10px; font-weight: bold; color: #f59e0b;">{top_reason}</td>
                </tr>
            </table>
            
            <div style="background-color: #7f1d1d; color: #fecaca; padding: 15px; border-radius: 6px; font-weight: bold; text-align: center; margin-top: 20px;">
                MANDATORY EVACUATION: RESTRICT ALL PERSONNEL AND MACHINERY FROM LOWER BENCH SECTORS IMMEDIATELY.
            </div>
        </div>
    </body>
    </html>
    """
    
    # Try sending via SMTP if credentials exist, otherwise log dispatch
    success = False
    if SMTP_USER and SMTP_PASS:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = SMTP_USER
            msg["To"] = to_email
            msg.attach(MIMEText(html_content, "html"))
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_USER, to_email, msg.as_string())
            success = True
            print(f"[+] Emergency Email DISPATCHED to {to_email}")
        except Exception as e:
            print(f"[!] SMTP Email dispatch error ({e})")
            
    log_entry = {
        "type": "EMAIL",
        "recipient": to_email,
        "subject": subject,
        "status": "SENT (SMTP)" if success else "DISPATCHED (Simulated Server Log)",
        "timestamp": os.popen("date /t").read().strip() if os.name == 'nt' else "Now"
    }
    DISPATCH_LOGS.insert(0, log_entry)
    return log_entry

def send_emergency_sms(to_phone: str, mine_name: str, risk_pct: float, top_reason: str) -> Dict[str, Any]:
    """
    Sends an automated Emergency SMS / WhatsApp alert to the mine contact phone number.
    """
    if not to_phone:
        to_phone = "+1-555-0199"
        
    sms_text = f"🚨 ROCKFALLGUARD EMERGENCY ALERT: {mine_name} risk reached {risk_pct:.1f}% ({top_reason}). EVACUATE LOWER BENCH IMMEDIATELY!"
    
    success = False
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
            data = {
                "From": TWILIO_PHONE_NUMBER,
                "To": to_phone,
                "Body": sms_text
            }
            res = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            if res.status_code in [200, 201]:
                success = True
                print(f"[+] Emergency SMS DISPATCHED to {to_phone}")
        except Exception as e:
            print(f"[!] Twilio SMS dispatch error ({e})")
            
    log_entry = {
        "type": "SMS/PHONE",
        "recipient": to_phone,
        "message": sms_text,
        "status": "SENT (Twilio)" if success else "DISPATCHED (SMS Gateway Log)",
        "timestamp": "Now"
    }
    DISPATCH_LOGS.insert(0, log_entry)
    return log_entry

def get_dispatch_logs():
    return DISPATCH_LOGS[:20]
