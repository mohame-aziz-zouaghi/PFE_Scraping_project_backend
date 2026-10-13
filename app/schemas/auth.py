"""Auth schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)


# ═══════════════════════════════════════════════════════
# REQUESTS
# ═══════════════════════════════════════════════════════
class UserRegister(BaseModel):
    """Payload for creating a new user account."""
    email: EmailStr
    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="3-50 chars, letters/digits/underscore only",
    )
    password: str = Field(
        min_length=8,
        max_length=72,
        description="Minimum 8 chars. Max 72 due to bcrypt limit.",
    )
    display_name: Optional[str] = Field(default=None, max_length=150)
    phone: Optional[str] = Field(default=None, max_length=30)
    turnstile_token: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)
    turnstile_token: Optional[str] = None
    remember: bool = False


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=72)
    new_password: str = Field(min_length=12, max_length=72)


class VerifyEmailRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=256)


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=256)
    new_password: str = Field(..., min_length=12, max_length=72)

    @field_validator("new_password")
    @classmethod
    def _strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain a letter")
        return v


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=72)
    new_password: str = Field(..., min_length=12, max_length=72)

    @field_validator("new_password")
    @classmethod
    def _strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain a letter")
        return v


# ═══════════════════════════════════════════════════════
# RESPONSES — explicit allow-list of public fields
# ═══════════════════════════════════════════════════════
class UserResponse(BaseModel):
    """Public user profile. NEVER includes tokens, hashes, or
    internal flags that could leak information.

    Every field here is intentional — do not add fields without
    thinking about whether they should be visible to the caller.
    """

    model_config = ConfigDict(from_attributes=True)

    # ── Identity
    id: UUID
    email: EmailStr
    username: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    phone: Optional[str] = None

    # ── Account state
    role: str
    tier: str
    is_active: bool
    is_verified: bool

    # ── Timestamps
    created_at: datetime
    last_login_at: Optional[datetime] = None

    # ── Computed / defaults
    @field_validator("role", mode="before")
    @classmethod
    def _role_default(cls, v):
        return v or "user"

    @field_validator("tier", mode="before")
    @classmethod
    def _tier_default(cls, v):
        return v or "free"

    @field_validator("is_active", mode="before")
    @classmethod
    def _is_active_default(cls, v):
        return True if v is None else bool(v)

    @field_validator("is_verified", mode="before")
    @classmethod
    def _is_verified_default(cls, v):
        return bool(v) if v is not None else False

    @field_validator("created_at", mode="before")
    @classmethod
    def _created_at_default(cls, v):
        if v is None:
            from datetime import timezone
            return datetime.now(timezone.utc)
        return v


class UserPublic(BaseModel):
    """Minimal user info for embedding in other responses.

    This is used when a user is referenced from another resource
    (e.g., a review, comment, or shared item). Includes only what's
    safe to show to any viewer.
    """
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    tier: str


class RegisterResponse(BaseModel):
    """Payload returned by POST /auth/register.

    `verification_expires_at` drives the frontend's countdown.
    `email` is echoed so the wizard can display it for confirmation.
    """
    user_id: UUID
    email: str
    username: str
    verification_expires_at: datetime


class TokenResponse(BaseModel):
    """Response after successful login/refresh."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        description="Seconds until access token expires"
    )


class MessageResponse(BaseModel):
    message: str