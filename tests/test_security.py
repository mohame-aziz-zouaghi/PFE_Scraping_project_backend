from datetime import timedelta

import pytest

from app.core.exceptions import AuthenticationError
from app.core.security import create_access_token, create_refresh_token, decode_token


def test_access_token_round_trip():
    token = create_access_token("user-1", expires_delta=timedelta(minutes=1))
    assert decode_token(token)["sub"] == "user-1"


def test_refresh_token_is_not_an_access_token():
    with pytest.raises(AuthenticationError, match="Invalid token type"):
        decode_token(create_refresh_token("user-1"))
