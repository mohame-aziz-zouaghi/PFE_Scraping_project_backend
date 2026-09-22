"""
Email service — SMTP or console fallback.

Usage:
    await EmailService.send_verification(user, token, base_url)
    await EmailService.send_password_reset(user, token, base_url)
"""
from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional

from app.config import settings
from app.core.logging_config import get_logger
from app.models.user import User

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# HTML templates
# ═══════════════════════════════════════════════════════
_BASE = """<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f4f5f7;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr><td align="center" style="padding:40px 20px;">
      <table role="presentation" width="560" cellpadding="0" cellspacing="0"
             style="background:#fff;border-radius:8px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);overflow:hidden;">
        <tr><td style="padding:32px 40px 8px;">
          <div style="font-size:13px;letter-spacing:.12em;text-transform:uppercase;
                      color:#6b7280;font-weight:600;">{brand}</div>
          <h1 style="margin:8px 0 0;font-size:22px;color:#111827;
                     font-weight:700;">{title}</h1>
        </td></tr>
        <tr><td style="padding:16px 40px 32px;color:#374151;font-size:15px;
                       line-height:1.6;">
          {body}
        </td></tr>
        <tr><td style="padding:16px 40px 32px;border-top:1px solid #f3f4f6;
                       color:#9ca3af;font-size:12px;line-height:1.5;">
          {footer}
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>"""


def _button(url: str, label: str) -> str:
    return (
        f'<p style="margin:24px 0;">'
        f'<a href="{url}" '
        f'style="display:inline-block;padding:12px 24px;background:#111827;'
        f'color:#fff;text-decoration:none;border-radius:6px;font-weight:600;'
        f'font-size:14px;">{label}</a></p>'
    )


def _fallback_url(url: str) -> str:
    return (
        f'<p style="margin:16px 0 0;color:#6b7280;font-size:12px;">'
        f'Button not working? Paste this link into your browser:<br>'
        f'<a href="{url}" style="color:#2563eb;word-break:break-all;">{url}</a>'
        f'</p>'
    )


def render_verification_email(user: User, verify_url: str) -> tuple[str, str, str]:
    """Returns (subject, html, text)."""
    subject = "Verify your email address"
    html = _BASE.format(
        brand="Monitoring System",
        title="Confirm your email",
        body=(
            f"<p>Hi {user.display_name or user.username},</p>"
            f"<p>Thanks for signing up. Confirm your email address to activate "
            f"your account and start receiving price alerts.</p>"
            f"{_button(verify_url, 'Verify email')}"
            f"{_fallback_url(verify_url)}"
            f"<p style='margin-top:24px;color:#6b7280;font-size:13px;'>"
            f"This link expires in {settings.email_verify_token_ttl_hours} hours. "
            f"If you didn't create this account, you can safely ignore this email."
            f"</p>"
        ),
        footer=(
            f"You're receiving this because someone signed up with {user.email}. "
            f"Questions? Reply to this email."
        ),
    )
    text = (
        f"Hi {user.display_name or user.username},\n\n"
        f"Confirm your email address:\n{verify_url}\n\n"
        f"This link expires in {settings.email_verify_token_ttl_hours} hours.\n"
        f"If you didn't sign up, ignore this email."
    )
    return subject, html, text


def render_password_reset_email(user: User, reset_url: str) -> tuple[str, str, str]:
    subject = "Reset your password"
    html = _BASE.format(
        brand="Monitoring System",
        title="Reset your password",
        body=(
            f"<p>Hi {user.display_name or user.username},</p>"
            f"<p>We received a request to reset your password. Click the button "
            f"below to choose a new one.</p>"
            f"{_button(reset_url, 'Reset password')}"
            f"{_fallback_url(reset_url)}"
            f"<p style='margin-top:24px;color:#6b7280;font-size:13px;'>"
            f"This link expires in {settings.password_reset_token_ttl_minutes} "
            f"minutes. If you didn't request a reset, ignore this email — your "
            f"password won't change.</p>"
        ),
        footer=(
            f"For your security, this link can only be used once. "
            f"If you didn't request it, no action is needed."
        ),
    )
    text = (
        f"Hi {user.display_name or user.username},\n\n"
        f"Reset your password:\n{reset_url}\n\n"
        f"Expires in {settings.password_reset_token_ttl_minutes} minutes.\n"
        f"If you didn't request this, ignore this email."
    )
    return subject, html, text


# ═══════════════════════════════════════════════════════
# Low-level send
# ═══════════════════════════════════════════════════════
def _send_sync(
    to_email: str,
    subject: str,
    html: str,
    text: str,
) -> None:
    """Blocking SMTP call — always invoked via asyncio.to_thread."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((settings.email_from_name, settings.email_from))
    msg["To"] = to_email
    if settings.email_reply_to:
        msg["Reply-To"] = settings.email_reply_to

    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    if settings.smtp_use_ssl:
        with smtplib.SMTP_SSL(
            settings.smtp_host,
            settings.smtp_port,
            timeout=settings.smtp_timeout,
        ) as s:
            if settings.smtp_user:
                s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
        return

    with smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=settings.smtp_timeout,
    ) as s:
        s.ehlo()
        if settings.smtp_use_tls:
            s.starttls()
            s.ehlo()
        if settings.smtp_user:
            s.login(settings.smtp_user, settings.smtp_password)
        s.send_message(msg)


class EmailService:
    @staticmethod
    async def send(
        to_email: str,
        subject: str,
        html: str,
        text: str,
    ) -> bool:
        """
        Send an email. In dev mode or when disabled, log to console.
        Never raises — failures are logged so auth flows don't 500.
        """
        if not settings.email_enabled or settings.email_dev_mode:
            logger.info(
                "email_skipped_dev_mode",
                to=to_email,
                subject=subject,
                preview=text[:400],
            )
            return True

        try:
            await asyncio.to_thread(
                _send_sync, to_email, subject, html, text
            )
            logger.info("email_sent", to=to_email, subject=subject)
            return True
        except Exception as e:
            logger.error(
                "email_send_failed",
                to=to_email,
                subject=subject,
                error=str(e),
            )
            return False

    # ── High-level helpers ──────────────────────────────
    @staticmethod
    async def send_verification(user: User, raw_token: str) -> bool:
        url = (
            f"{settings.frontend_url}/verify-email"
            f"?token={raw_token}"
        )
        subject, html, text = render_verification_email(user, url)
        return await EmailService.send(user.email, subject, html, text)

    @staticmethod
    async def send_password_reset(user: User, raw_token: str) -> bool:
        url = (
            f"{settings.frontend_url}/reset-password"
            f"?token={raw_token}"
        )
        subject, html, text = render_password_reset_email(user, url)
        return await EmailService.send(user.email, subject, html, text)