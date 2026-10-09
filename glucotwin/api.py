"""FastAPI backend for the GlucoTwin clinician dashboard.

    uvicorn glucotwin.api:app --port 8000      then open http://localhost:8000
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config as C
from .service import ClinicalService

DASHBOARD = C.ROOT / "dashboard"

app = FastAPI(title="GlucoTwin India API", version="0.1.0",
              description="Hybrid physiological + ML digital twin for Type 2 Diabetes (synthetic data, proof of concept).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def svc() -> ClinicalService:
    return ClinicalService()


def _check(pid: str) -> None:
    if pid not in svc().ward_ids:
        raise HTTPException(404, f"Patient {pid} is not in the virtual ward")


class MealPlan(BaseModel):
    meal_key: str
    portion: float = Field(1.0, ge=0.25, le=3.0)
    offset_min: int = Field(0, ge=0, le=600)


class WalkPlan(BaseModel):
    offset_min: int = Field(0, ge=0, le=600)
    duration_min: int = Field(15, ge=0, le=180)
    cadence: float = Field(100, ge=0, le=160)


class Scenario(BaseModel):
    label: str = "scenario"
    meals: list[MealPlan] = []
    walks: list[WalkPlan] = []


class WhatIfRequest(BaseModel):
    at: str | None = None
    horizon_min: int = Field(300, ge=60, le=720)
    scenarios: list[Scenario]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return svc().meta()


@app.get("/api/ward")
def ward(at: str | None = None):
    return svc().ward(at)


@app.get("/api/patients/{pid}")
def patient(pid: str, at: str | None = None, window_h: float = Query(12, ge=2, le=48)):
    _check(pid)
    return svc().patient(pid, at, window_h)


@app.post("/api/patients/{pid}/whatif")
def whatif(pid: str, req: WhatIfRequest):
    _check(pid)
    from .meals import MEALS
    for sc in req.scenarios:
        for m in sc.meals:
            if m.meal_key not in MEALS:
                raise HTTPException(422, f"Unknown meal '{m.meal_key}'")
    return svc().whatif(pid, req.at, [s.model_dump() for s in req.scenarios], req.horizon_min)


@app.get("/api/patients/{pid}/meal-ranking")
def meal_ranking(pid: str, slot: str = Query("lunch", pattern="^(breakfast|lunch|dinner|snack)$"),
                 at: str | None = None, offset_min: int = 0, walk_min: int = 0):
    _check(pid)
    return svc().meal_ranking(pid, at, slot, offset_min, walk_min)


@app.get("/api/patients/{pid}/fhir")
def fhir(pid: str):
    _check(pid)
    return svc().fhir(pid)


if DASHBOARD.exists():
    app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(DASHBOARD / "index.html")
