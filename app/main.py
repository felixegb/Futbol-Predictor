from __future__ import annotations

from typing import Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .data_sources import LEAGUE_CONFIG, fetch_league_data, fetch_standings_table
from .predictor import PredictionService, MatchPredictor
from .schemas import (
    LeagueInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
    StandingsResponse,
    CompareRequest,
    CompareResponse,
    ModelPrediction,
)

# Aplicación principal de FastAPI.
app = FastAPI(title="Football Predictor")

# Resolver la carpeta base del paquete para servir archivos estáticos y plantillas HTML.
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Caché en memoria de servicios entrenados por liga y modelo.
# Estructura: {league: {model: PredictionService}}
_service_cache: Dict[str, Dict[str, PredictionService]] = {}
# Caché de métricas calculadas al entrenar cada servicio.
# Estructura: {league: {model: {metric_name: value}}}
_metrics_cache: Dict[str, Dict[str, Dict[str, float]]] = {}


def _get_metrics(league: str, model: str) -> Dict[str, float]:
    """Devuelve las métricas cacheadas para una liga y modelo, o un dict vacío."""
    return _metrics_cache.get(league, {}).get(model, {})


def get_or_create_service(league: str, model: str) -> PredictionService:
    """Recupera un servicio ya entrenado o lo crea y entrena si aún no existe."""
    # Obtener o crear el subdiccionario de la liga dentro de la caché.
    league_cache = _service_cache.setdefault(league, {})

    # Si el modelo ya fue entrenado para esa liga, reutilizarlo.
    if model in league_cache:
        return league_cache[model]

    # Descargar y preparar los datos históricos de la liga.
    data = fetch_league_data(league)
    if data.df.empty:
        raise HTTPException(status_code=404, detail="No hay datos disponibles para la liga.")

    # Crear el servicio de predicción con el modelo solicitado.
    service = PredictionService(model_name=model)
    try:
        # Entrenar el modelo y guardar sus métricas para reutilizarlas luego.
        metrics = service.fit(data)
        _metrics_cache.setdefault(league, {})[model] = metrics
    except Exception as exc:
        # Traducir errores internos de entrenamiento a error HTTP del servidor.
        raise HTTPException(status_code=500, detail=str(exc))

    # Guardar el servicio entrenado en caché para futuras predicciones.
    league_cache[model] = service
    return service


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Renderiza la página principal con la lista de ligas disponibles."""
    # Convertir la configuración interna en objetos LeagueInfo para la plantilla.
    leagues = [LeagueInfo(key=k, name=v["name"]) for k, v in LEAGUE_CONFIG.items()]
    return templates.TemplateResponse("index.html", {"request": request, "leagues": leagues})


@app.get("/api/standings/{league}", response_model=StandingsResponse)
async def standings(league: str) -> StandingsResponse:
    """Devuelve la tabla de clasificación de la liga solicitada."""
    table = fetch_standings_table(league)
    if table.empty:
        raise HTTPException(status_code=404, detail="No hay datos para la liga seleccionada.")

    # Convertir el DataFrame a lista de diccionarios para serializarlo como JSON.
    rows = table.to_dict(orient="records")
    return StandingsResponse(league=league, rows=rows)


@app.get("/api/teams/{league}")
async def teams(league: str) -> Dict[str, object]:
    """Devuelve la lista de equipos únicos presentes en los datos de la liga."""
    data = fetch_league_data(league)
    if data.df.empty:
        raise HTTPException(status_code=404, detail="No hay datos disponibles para la liga.")

    # Unir equipos locales y visitantes, eliminar duplicados y ordenar alfabéticamente.
    teams = sorted(set(data.df["home_team"]).union(set(data.df["away_team"])))
    return {"league": league, "teams": teams}


@app.get("/api/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    """Lista los modelos de predicción disponibles en la aplicación."""
    return ModelsResponse(models=MatchPredictor.available_models())


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    """Genera una predicción para un partido usando un único modelo."""
    # Validación básica: un equipo no puede jugar contra sí mismo.
    if payload.home_team == payload.away_team:
        raise HTTPException(status_code=400, detail="Local y visitante no pueden ser el mismo equipo.")

    # Obtener el servicio entrenado para la liga y el modelo solicitados.
    service = get_or_create_service(payload.league, payload.model)
    try:
        # Ejecutar la predicción del partido.
        result = service.predict_match(payload.home_team, payload.away_team)
    except Exception as exc:
        # Errores de validación o predicción se exponen como 400.
        raise HTTPException(status_code=400, detail=str(exc))

    # Recuperar las métricas del modelo para incluirlas en la respuesta.
    metrics = _get_metrics(payload.league, payload.model)
    return PredictionResponse(
        league=payload.league,
        home_team=payload.home_team,
        away_team=payload.away_team,
        model=payload.model,
        label=result["label"],
        probabilities=result["probabilities"],
        acceptance_percent=(
            float(metrics.get("accuracy", 0.0)) * 100.0
            if metrics
            else None
        ),
        metrics=metrics or None,
    )


@app.post("/api/predict-compare", response_model=CompareResponse)
async def predict_compare(payload: CompareRequest) -> CompareResponse:
    """Compara las predicciones de dos modelos para el mismo partido."""
    # Misma validación básica que en el endpoint de predicción simple.
    if payload.home_team == payload.away_team:
        raise HTTPException(status_code=400, detail="Local y visitante no pueden ser el mismo equipo.")

    # Evaluar exactamente dos modelos: el modelo A y el modelo B.
    models = [payload.model_a, payload.model_b]
    predictions = []
    for model in models:
        # Reutilizar o entrenar el servicio asociado a este modelo.
        service = get_or_create_service(payload.league, model)
        try:
            # Generar la predicción del partido con el modelo actual.
            result = service.predict_match(payload.home_team, payload.away_team)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Añadir a la respuesta tanto la predicción como sus métricas de entrenamiento.
        metrics = _get_metrics(payload.league, model)
        predictions.append(
            ModelPrediction(
                model=model,
                label=result["label"],
                probabilities=result["probabilities"],
                acceptance_percent=float(metrics.get("accuracy", 0.0)) * 100.0 if metrics else None,
                metrics=metrics or None,
            )
        )

    return CompareResponse(
        league=payload.league,
        home_team=payload.home_team,
        away_team=payload.away_team,
        predictions=predictions,
    )
