"""Optional authenticated REST access to local events, evidence and review history."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .events.store import REVIEW_STATUSES, EventStore

bearer = HTTPBearer(auto_error=False)


class ReviewInput(BaseModel):
    status: str
    note: str = Field(default="", max_length=2000)
    reviewer: str = Field(min_length=1, max_length=100)


def create_app(output_root: Path | None = None, token: str | None = None) -> FastAPI:
    root = (
        output_root.resolve()
        if output_root is not None
        else Path(os.environ.get("UTILITY_SAFETY_OUTPUT_DIR", "outputs")).resolve()
    )
    expected = token if token is not None else os.getenv("UTILITY_SAFETY_API_TOKEN", "")
    if len(expected) < 24:
        raise ValueError(
            "Set UTILITY_SAFETY_API_TOKEN to a random token of at least 24 characters."
        )
    store = EventStore(root / "events.sqlite3")

    def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if credentials is None or not hmac.compare_digest(
            credentials.credentials.encode(), expected.encode()
        ):
            raise HTTPException(
                401, "Bearer token required.", headers={"WWW-Authenticate": "Bearer"}
            )

    app = FastAPI(
        title="Utility Safety Events", version="2.2.0", dependencies=[Depends(authenticate)]
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "delivery": store.delivery_summary()}

    @app.get("/events")
    def events(
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        run_id: str | None = None,
    ):
        rows = store.list_events(after=after, limit=limit, run_id=run_id)
        return {"events": rows, "next_cursor": rows[-1]["sequence"] if rows else after}

    @app.post("/events/{event_id}/reviews", status_code=201)
    def review(event_id: str, decision: ReviewInput):
        if decision.status not in REVIEW_STATUSES:
            raise HTTPException(422, "Unsupported review status.")
        try:
            review_id = store.review(event_id, decision.status, decision.note, decision.reviewer)
        except KeyError:
            raise HTTPException(404, "Event not found.") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"review_id": review_id}

    @app.get("/events/{event_id}/reviews")
    def review_history(event_id: str):
        return store.review_history(event_id)

    @app.get("/events/{event_id}/evidence")
    def evidence(event_id: str):
        try:
            path = store.evidence_path(event_id, root)
        except ValueError:
            raise HTTPException(404, "Evidence not found.") from None
        if path is None:
            raise HTTPException(404, "Evidence not found.")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    return app
