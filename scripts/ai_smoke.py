"""
Smoke test for Phase 1.

Usage:
    cd backend
    python -m scripts.ai_smoke
"""
import asyncio

from pydantic import BaseModel

from app.core.llm_client import LLMClient
from app.core.prompt_templates import SMOKE_TEST_V1
from app.db.session import async_session_maker
from app.services.embedding_service import EmbeddingService


class EchoResult(BaseModel):
    echo: str
    ok: bool


async def main():
    print("── LLM chat (JSON) ──────────────────────────")
    async with async_session_maker() as db:
        client = LLMClient(
            db=db,
            endpoint="smoke_test",
            prompt_version=SMOKE_TEST_V1.version,
        )
        result = await client.chat_json(
            system=SMOKE_TEST_V1.system,
            user={"echo": "hello"},
            schema=EchoResult,
            cache_key_parts=["smoke", "hello"],
        )
        print(f"  echo={result.echo!r} ok={result.ok}")

        print("\n── Call again (should hit cache) ────────────")
        result2 = await client.chat_json(
            system=SMOKE_TEST_V1.system,
            user={"echo": "hello"},
            schema=EchoResult,
            cache_key_parts=["smoke", "hello"],
        )
        print(f"  echo={result2.echo!r} ok={result2.ok}")

        print("\n── Embedding ────────────────────────────────")
        svc = EmbeddingService(db=db, endpoint="smoke_embedding")
        vec = await svc.embed_one("iPhone 15 Pro Max 256GB Titanium")
        print(f"  dimensions: {len(vec)}")
        print(f"  first 3:    {vec[:3]}")

        print("\n✓ Phase 1 smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
