from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LeagueInfo(BaseModel):
    key: str
    name: str


class StandingsRow(BaseModel):
    pos: int
    team: str
    pj: int
    gf: int
    ga: int
    dg: int
    pts: int


class StandingsResponse(BaseModel):
    league: str
    rows: List[StandingsRow]


class ModelsResponse(BaseModel):
    models: List[str]


class PredictionRequest(BaseModel):
    league: str = Field(..., description="League key")
    home_team: str
    away_team: str
    model: str = "xgboost"


class PredictionResponse(BaseModel):
    league: str
    home_team: str
    away_team: str
    model: str
    label: str
    probabilities: Dict[str, float]
    acceptance_percent: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None


class CompareRequest(BaseModel):
    league: str = Field(..., description="League key")
    home_team: str
    away_team: str
    model_a: str
    model_b: str


class ModelPrediction(BaseModel):
    model: str
    label: str
    probabilities: Dict[str, float]
    acceptance_percent: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None


class CompareResponse(BaseModel):
    league: str
    home_team: str
    away_team: str
    predictions: List[ModelPrediction]
