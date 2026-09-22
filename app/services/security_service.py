"""Security-focused helpers shared by API handlers."""
from app.core.security import decode_token


class SecurityService:
    @staticmethod
    def validate_access_token(token: str) -> dict:
        """Validate an access token and return its payload."""
        return decode_token(token, expected_type="access")