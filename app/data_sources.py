from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from io import StringIO


# Configuración de las ligas soportadas.
# Cada clave es el identificador de ruta (slug) de la liga.
LEAGUE_CONFIG = {
    "premier-league": {
        "name": "Premier League",
        "code": "E0",
        "thesportsdb_id": "4328",
        "seasons": ["2122", "2223", "2324", "2425"],
    },
    "la-liga": {
        "name": "La Liga",
        "code": "SP1",
        "thesportsdb_id": "4335",
        "seasons": ["2122", "2223", "2324", "2425"],
    },
    "bundesliga": {
        "name": "Bundesliga",
        "code": "D1",
        "thesportsdb_id": "4331",
        "seasons": ["2122", "2223", "2324", "2425"],
    },
    "ligue-1": {
        "name": "Ligue 1",
        "code": "F1",
        "thesportsdb_id": "4334",
        "seasons": ["2122", "2223", "2324", "2425"],
    },
    "serie-a": {
        "name": "Serie A",
        "code": "I1",
        "thesportsdb_id": "4332",
        "seasons": ["2122", "2223", "2324", "2425"],
    },
}

THE_SPORTS_DB_BASE = "https://www.thesportsdb.com/api/v1/json/123"


# Contenedor de datos de partidos de una liga.
# - df: DataFrame con todos los partidos normalizados.
# - league: slug de la liga (ej. "premier-league").
# - seasons: lista de temporadas incluidas en el DataFrame.
@dataclass
class MatchData:
    df: pd.DataFrame
    league: str
    seasons: List[str]


def season_url(code: str, season: str) -> str:
    """Construye la URL del CSV de una temporada en football-data.co.uk."""
    return f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"


def ensure_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena el DataFrame de partidos de forma cronológica."""
    if "match_date" in df.columns:
        # Ordena primero por fecha y luego por temporada para desempatar
        return df.sort_values(["match_date", "season"]).reset_index(drop=True)
    # Si no existe la columna de fecha, solo reinicia los índices
    return df.reset_index(drop=True)


def _parse_date(value: object) -> Optional[datetime]:
    # Descartar valores nulos o NaN de pandas/numpy
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    # Probar los tres formatos de fecha que aparecen en los CSVs de football-data.co.uk
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    # Si ningún formato funcionó, devolver None
    return None


def clean_season_data(data: pd.DataFrame, season_label: str) -> pd.DataFrame:
    """Normalize raw football-data.co.uk CSV to standard columns."""

    # Si el CSV llega vacío, devolver un DataFrame vacío inmediatamente
    if data.empty:
        return pd.DataFrame()

    rows = []
    for _, row in data.iterrows():
        try:
            # Leer goles del equipo local (FTHG) y visitante (FTAG)
            fthg = row.get("FTHG")
            ftag = row.get("FTAG")
            # Omitir filas con goles nulos (partidos incompletos o cabecera extra)
            if pd.isna(fthg) or pd.isna(ftag):
                continue

            home_goals = int(fthg)
            away_goals = int(ftag)

            # Leer y limpiar nombres de equipos; omitir si están vacíos
            home_team = str(row.get("HomeTeam", "")).strip()
            away_team = str(row.get("AwayTeam", "")).strip()
            if not home_team or not away_team:
                continue

            # Calcular el resultado: H = local gana, A = visitante gana, D = empate
            if home_goals > away_goals:
                result = "H"
            elif away_goals > home_goals:
                result = "A"
            else:
                result = "D"

            # Parsear la fecha; si no es válida, usar datetime.min como fallback
            match_date = _parse_date(row.get("Date"))

            # Agregar el partido normalizado a la lista
            rows.append(
                {
                    "season": season_label,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "match_date": match_date or datetime.min,
                }
            )
        except (ValueError, TypeError):
            # Si alguna conversión falla por datos corruptos, omitir la fila
            continue

    return pd.DataFrame(rows)


def fetch_league_data(league_key: str) -> MatchData:
    # Validar que la liga existe en la configuración
    if league_key not in LEAGUE_CONFIG:
        raise ValueError("Liga no soportada.")

    config = LEAGUE_CONFIG[league_key]
    code = config["code"]
    seasons = config["seasons"]

    all_matches: List[pd.DataFrame] = []
    for season in seasons:
        url = season_url(code, season)
        try:
            # Descargar el CSV de la temporada con timeout de 15 segundos
            response = requests.get(url, timeout=15)
            response.raise_for_status()  # Lanza error si el servidor devuelve 4xx/5xx
            season_df = pd.read_csv(StringIO(response.text))
            # Limpiar y normalizar el CSV descargado
            cleaned = clean_season_data(season_df, season)
            if not cleaned.empty:
                all_matches.append(cleaned)
        except Exception:
            # Si la temporada falla (red, CSV corrupto, etc.), continuar con la siguiente
            continue

    # Si no se pudo descargar ninguna temporada, devolver un MatchData vacío
    if not all_matches:
        return MatchData(df=pd.DataFrame(), league=league_key, seasons=seasons)

    # Combinar todas las temporadas en un único DataFrame y ordenar cronológicamente
    data = pd.concat(all_matches, ignore_index=True)
    data = ensure_chronological(data)
    return MatchData(df=data, league=league_key, seasons=seasons)


def _to_int(value: object) -> int:
    """Convierte un valor a entero de forma segura; devuelve 0 si falla."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fetch_standings_thesportsdb(league_key: str) -> pd.DataFrame:
    """Obtiene la tabla de clasificación en tiempo real desde la API de TheSportsDB."""
    config = LEAGUE_CONFIG[league_key]
    league_id = config.get("thesportsdb_id")
    # Si la liga no tiene ID configurado, devolver DataFrame vacío
    if not league_id:
        return pd.DataFrame(columns=["pos", "team", "pj", "gf", "ga", "dg", "pts"])

    # Llamar al endpoint de clasificación con el ID de la liga
    url = f"{THE_SPORTS_DB_BASE}/lookuptable.php?l={league_id}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()
    # La API puede devolver la tabla bajo la clave "table" o "standings"
    table = payload.get("table") or payload.get("standings") or []

    rows = []
    for idx, entry in enumerate(table, start=1):
        # Ignorar entradas que no sean diccionarios (datos corruptos)
        if not isinstance(entry, dict):
            continue

        # Extraer cada campo probando distintos nombres de clave posibles en la API
        pos = _to_int(entry.get("intRank") or entry.get("rank") or idx)
        team = entry.get("strTeam") or entry.get("team") or entry.get("name") or "Unknown"
        pj = _to_int(entry.get("intPlayed") or entry.get("played") or entry.get("matchesPlayed"))
        gf = _to_int(entry.get("intGoalsFor") or entry.get("goalsFor") or entry.get("gf"))
        ga = _to_int(entry.get("intGoalsAgainst") or entry.get("goalsAgainst") or entry.get("ga"))
        # Diferencia de goles: usar el campo de la API o calcularlo si no viene
        dg = _to_int(entry.get("intGoalDifference") or entry.get("goalDifference") or (gf - ga))
        pts = _to_int(entry.get("intPoints") or entry.get("points"))

        rows.append(
            {
                "pos": pos or idx,  # Usar idx como fallback si pos es 0
                "team": str(team),
                "pj": pj,
                "gf": gf,
                "ga": ga,
                "dg": dg,
                "pts": pts,
            }
        )

    # Ordenar filas por posición antes de construir el DataFrame
    rows = sorted(rows, key=lambda r: r["pos"])
    return pd.DataFrame(rows, columns=["pos", "team", "pj", "gf", "ga", "dg", "pts"])


def compute_standings(match_data: MatchData) -> pd.DataFrame:
    """Compute a current standings table from match results."""

    df = match_data.df.copy()
    # Si no hay partidos, devolver tabla vacía con las columnas correctas
    if df.empty:
        return pd.DataFrame(columns=["pos", "team", "pj", "gf", "ga", "dg", "pts"])

    # Asignar puntos al equipo local y visitante según el resultado de cada partido
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0}).astype(int)
    df["away_pts"] = df["result"].map({"A": 3, "D": 1, "H": 0}).astype(int)

    # Agregar estadísticas como equipo local: partidos, goles a favor, en contra y puntos
    home = df.groupby("home_team").agg(
        pj=("home_team", "count"),
        gf=("home_goals", "sum"),
        ga=("away_goals", "sum"),
        pts=("home_pts", "sum"),
    )
    # Agregar estadísticas como equipo visitante
    away = df.groupby("away_team").agg(
        pj=("away_team", "count"),
        gf=("away_goals", "sum"),
        ga=("home_goals", "sum"),
        pts=("away_pts", "sum"),
    )

    # Sumar local + visitante para obtener el total de cada equipo
    # fill_value=0 maneja equipos que sólo aparecen en uno de los dos grupos
    table = home.add(away, fill_value=0).reset_index().rename(columns={"index": "team"})

    # Calcular diferencia de goles
    table["dg"] = table["gf"] - table["ga"]

    # Ordenar por: puntos (desc) → diferencia de goles (desc) → goles a favor (desc)
    table = table.sort_values(["pts", "dg", "gf"], ascending=[False, False, False]).reset_index(drop=True)

    # Insertar columna de posición basada en el orden resultante
    table.insert(0, "pos", table.index + 1)

    return table[["pos", "team", "pj", "gf", "ga", "dg", "pts"]]


def fetch_standings_table(league_key: str) -> pd.DataFrame:
    """Fetch standings from external API with CSV fallback."""

    try:
        # Intentar primero con la API en tiempo real de TheSportsDB
        table = _fetch_standings_thesportsdb(league_key)
        if not table.empty:
            return table
    except Exception:
        # Si la API falla (error de red, respuesta inválida, etc.), usar el fallback
        pass

    # Fallback: calcular la clasificación a partir de los CSVs históricos descargados
    data = fetch_league_data(league_key)
    return compute_standings(data)
