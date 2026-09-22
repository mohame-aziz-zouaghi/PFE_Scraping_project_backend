"""
Standalone SMTP test. Run: python -m scripts.test_smtp
Does NOT touch the app — just reads .env and sends one email.
"""
import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

# Load .env from backend/
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os

HOST = os.getenv("SMTP_HOST")
PORT = int(os.getenv("SMTP_PORT", "587"))
USER = os.getenv("SMTP_USER")
PASSWORD = os.getenv("SMTP_PASSWORD")
USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
FROM = os.getenv("EMAIL_FROM")
FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Test")
TO = input("Send test email to: ").strip()


def main() -> int:
    print(f"\n── Config ─────────────────────────────")
    print(f"  host      : {HOST}:{PORT}")
    print(f"  user      : {USER}")
    print(f"  password  : {'*' * len(PASSWORD) if PASSWORD else '(empty)'} "
          f"({len(PASSWORD or '')} chars)")
    print(f"  use_tls   : {USE_TLS}")
    print(f"  use_ssl   : {USE_SSL}")
    print(f"  from      : {FROM}")
    print(f"  to        : {TO}")
    print(f"───────────────────────────────────────\n")

    if not HOST or not USER or not PASSWORD:
        print("✗ SMTP_HOST / SMTP_USER / SMTP_PASSWORD missing in .env")
        return 1

    msg = EmailMessage()
    msg["Subject"] = "SMTP test — Monitoring System"
    msg["From"] = formataddr((FROM_NAME, FROM))
    msg["To"] = TO
    msg.set_content("If you can read this, your SMTP config works.")
    msg.add_alternative(
        "<p>If you can read this, your <b>SMTP config works</b>.</p>",
        subtype="html",
    )

    try:
        print("→ connecting…")
        if USE_SSL:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(HOST, PORT, timeout=15, context=ctx) as s:
                s.set_debuglevel(1)   # <-- full protocol trace
                print("→ logging in…")
                s.login(USER, PASSWORD)
                print("→ sending…")
                s.send_message(msg)
        else:
            with smtplib.SMTP(HOST, PORT, timeout=15) as s:
                s.set_debuglevel(1)
                s.ehlo()
                if USE_TLS:
                    print("→ STARTTLS…")
                    s.starttls(context=ssl.create_default_context())
                    s.ehlo()
                print("→ logging in…")
                s.login(USER, PASSWORD)
                print("→ sending…")
                s.send_message(msg)

        print("\n✓ Email sent. Check the inbox (and spam folder).")
        return 0

    except smtplib.SMTPAuthenticationError as e:
        print(f"\n✗ AUTH FAILED: {e.smtp_code} {e.smtp_error.decode(errors='replace')}")
        print("  → Wrong username/password, or you need an App Password.")
        return 2
    except smtplib.SMTPConnectError as e:
        print(f"\n✗ CONNECT FAILED: {e}")
        print("  → Host/port wrong, or firewall blocking outbound SMTP.")
        return 3
    except smtplib.SMTPServerDisconnected as e:
        print(f"\n✗ SERVER DISCONNECTED: {e}")
        print("  → TLS/SSL mismatch. Try toggling SMTP_USE_TLS / SMTP_USE_SSL.")
        return 4
    except smtplib.SMTPRecipientsRefused as e:
        print(f"\n✗ RECIPIENT REFUSED: {e}")
        print("  → The 'to' address was rejected by the server.")
        return 5
    except smtplib.SMTPException as e:
        print(f"\n✗ SMTP ERROR: {type(e).__name__}: {e}")
        return 6
    except Exception as e:
        print(f"\n✗ UNEXPECTED: {type(e).__name__}: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())