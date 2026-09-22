"""Auth schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


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
        min_length=12,
        max_length=72,
        description="Minimum 12 chars. Max 72 due to bcrypt limit.",
    )


class UserLogin(BaseModel):
    """Login credentials."""
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    username: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

    role: str
    tier: str

    is_active: bool
    is_verified: bool
    is_banned: bool
    is_deleted: bool

    last_login_at: Optional[datetime] = None
    created_at: datetime

    @field_validator("role", mode="before")
    @classmethod
    def _role_default(cls, v):
        if v is None:
            return "user"
        return v

    @field_validator("tier", mode="before")
    @classmethod
    def _tier_default(cls, v):
        if v is None:
            return "free"
        return v

    @field_validator(
        "is_active", "is_verified", "is_banned", "is_deleted",
        mode="before",
    )
    @classmethod
    def _bool_flag_defaults(cls, v, info):
        if v is None:
            # is_active defaults True; the rest default False
            return info.field_name == "is_active"
        return bool(v)

    @field_validator("created_at", mode="before")
    @classmethod
    def _created_at_default(cls, v):
        if v is None:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc)
        return v


class UserPublic(BaseModel):
    """Minimal user info for nested responses."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    tier: str


class TokenResponse(BaseModel):
    """Response after successful login/refresh."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until access token expires")


class PasswordChange(BaseModel):
    """Payload for changing a password."""
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
    new_password: str = Field(..., min_length=12, max_length=72)   # ← was 8/128

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
    new_password: str = Field(..., min_length=12, max_length=72)   # ← was 8/128

    @field_validator("new_password")
    @classmethod
    def _strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain a letter")
        return v


class MessageResponse(BaseModel):
    message: str