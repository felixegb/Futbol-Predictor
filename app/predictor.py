from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .data_sources import MatchData, ensure_chronological


# Mapeos entre las etiquetas de resultado del fútbol y enteros para entrenar modelos.
# H = victoria local, D = empate, A = victoria visitante.
LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {0: "H", 1: "D", 2: "A"}


@dataclass
class EloConfig:
    # Intensidad del ajuste tras cada partido.
    k_factor: float = 20.0
    # Rating inicial que recibe cualquier equipo nuevo.
    base_rating: float = 1500.0
    # Ventaja fija añadida al equipo local al calcular expectativas.
    home_advantage: float = 100.0


class EloTracker:
   
    def __init__(self, config: Optional[EloConfig] = None) -> None:
        # Si no se pasa configuración, usar la configuración por defecto.
        self.config = config or EloConfig()
        # Diccionario con el rating actual de cada equipo.
        self.team_ratings: Dict[str, float] = {}

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcula la probabilidad esperada de que A puntúe mejor que B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, home_team: str, away_team: str, result: str) -> Tuple[float, float]:
        # Obtener rating actual de ambos equipos o el rating base si aún no existen.
        home_rating = self.team_ratings.get(home_team, self.config.base_rating)
        away_rating = self.team_ratings.get(away_team, self.config.base_rating)

        # Calcular el resultado esperado teniendo en cuenta la ventaja de jugar en casa.
        expected_home = self._expected_score(home_rating + self.config.home_advantage, away_rating)

        # Convertir el resultado real a puntuación ELO para el equipo local.
        if result == "H":
            actual_home = 1.0
        elif result == "D":
            actual_home = 0.5
        else:
            actual_home = 0.0

        # Ajustar los ratings: si el local rinde mejor de lo esperado, sube; si no, baja.
        delta = self.config.k_factor * (actual_home - expected_home)
        self.team_ratings[home_team] = home_rating + delta
        self.team_ratings[away_team] = away_rating - delta

        # Devolver los ratings previos al partido, que son los que se usan como feature.
        return home_rating, away_rating


class FeatureEngineer:
    def __init__(self, elo_config: Optional[EloConfig] = None) -> None:
        # Permite personalizar el cálculo ELO o usar los valores por defecto.
        self.elo_config = elo_config or EloConfig()

    def compute_features(self, data: MatchData) -> MatchData:
   
        # Asegurar orden temporal correcto para que las features no miren al futuro.
        df = ensure_chronological(data.df)

        # ELO (sequential, but optimized with list appends)
        tracker = EloTracker(self.elo_config)
        home_elo: List[float] = []
        away_elo: List[float] = []
        for row in df.itertuples(index=False):
            # update devuelve los ratings antes del partido y luego actualiza el estado interno.
            h_elo, a_elo = tracker.update(row.home_team, row.away_team, row.result)
            home_elo.append(h_elo)
            away_elo.append(a_elo)

        df = df.copy()
        df["home_elo"] = home_elo
        df["away_elo"] = away_elo

        # Convertir el resultado del partido en puntos para local y visitante.
        df["h_points"] = df["result"].map({"H": 3, "D": 1, "A": 0}).astype(float)
        df["a_points"] = df["result"].map({"A": 3, "D": 1, "H": 0}).astype(float)

        # Pasar a formato largo para calcular estadísticas móviles por equipo,
        # tratando por separado las apariciones como local y visitante.
        home_df = df[["season", "home_team", "h_points", "home_goals", "away_goals", "match_date"]].rename(
            columns={
                "home_team": "team",
                "h_points": "points",
                "home_goals": "gf",
                "away_goals": "ga",
            }
        )
        home_df["is_home"] = 1
        home_df["match_index"] = df.index

        away_df = df[["season", "away_team", "a_points", "away_goals", "home_goals", "match_date"]].rename(
            columns={
                "away_team": "team",
                "a_points": "points",
                "away_goals": "gf",
                "home_goals": "ga",
            }
        )
        away_df["is_home"] = 0
        away_df["match_index"] = df.index

        # Unir ambos lados del partido en un único historial por equipo.
        team_stats = pd.concat([home_df, away_df], ignore_index=True)
        team_stats = team_stats.sort_values(["team", "match_date", "match_index"])

        # Calcular forma reciente y medias móviles usando solo partidos anteriores (shift).
        grouped = team_stats.groupby("team", sort=False)
        team_stats["form_l5"] = grouped["points"].transform(
            lambda x: x.shift().rolling(window=5, min_periods=1).sum()
        )
        team_stats["gf_l5"] = grouped["gf"].transform(
            lambda x: x.shift().rolling(window=5, min_periods=1).mean()
        )
        team_stats["ga_l5"] = grouped["ga"].transform(
            lambda x: x.shift().rolling(window=5, min_periods=1).mean()
        )

        team_stats = team_stats.fillna(0.0)

    # Separar otra vez las estadísticas según el equipo jugó como local o visitante.
        stats_home = team_stats[team_stats["is_home"] == 1].set_index("match_index")
        stats_away = team_stats[team_stats["is_home"] == 0].set_index("match_index")

    # Copiar las features calculadas al DataFrame original de partidos.
        df["home_form"] = stats_home["form_l5"].astype(float)
        df["home_gf_avg"] = stats_home["gf_l5"].astype(float)
        df["home_ga_avg"] = stats_home["ga_l5"].astype(float)
        df["away_form"] = stats_away["form_l5"].astype(float)
        df["away_gf_avg"] = stats_away["gf_l5"].astype(float)
        df["away_ga_avg"] = stats_away["ga_l5"].astype(float)

        return MatchData(df=df, league=data.league, seasons=data.seasons)


class MatchPredictor:
    """Train and predict match outcomes for a league."""

    def __init__(self, model_name: str = "xgboost") -> None:
        # Guardar el nombre del modelo y construir la instancia concreta.
        self.model_name = model_name
        self.model = self._build_model(model_name, use_early_stopping=True)
        # Lista fija de columnas de entrada que usarán todos los modelos.
        self.features = [
            "home_elo",
            "away_elo",
            "home_form",
            "away_form",
            "home_gf_avg",
            "away_gf_avg",
            "home_ga_avg",
            "away_ga_avg",
        ]
        # Bandera para impedir predicciones antes del entrenamiento.
        self.trained = False

    @staticmethod
    def available_models() -> List[str]:
        """Devuelve los nombres de modelos que el sistema sabe construir."""
        return ["xgboost", "logistic", "random_forest"]

    def _build_model(self, model_name: str, use_early_stopping: bool = True):
        """Construye la instancia del modelo de ML según el nombre indicado."""
        factories: Dict[str, Callable[[], object]] = {
            "xgboost": lambda: XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                early_stopping_rounds=80 if use_early_stopping else None,
                n_estimators=800,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                gamma=0.3,
                reg_alpha=0.4,
                reg_lambda=1.5,
                random_state=42,
            ),
            "logistic": lambda: LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
            ),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced",
            ),
        }

        # Validar que el nombre del modelo esté soportado.
        if model_name not in factories:
            raise ValueError(f"Modelo no soportado: {model_name}")
        return factories[model_name]()

    def _rolling_cv(self, df: pd.DataFrame, n_splits: int = 3) -> Dict[str, float]:
        """Evalúa el modelo con validación cruzada temporal sobre el histórico."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = df[self.features].reset_index(drop=True)
        y = df["result"].map(LABEL_TO_INT).reset_index(drop=True)

        accs: List[float] = []
        losses: List[float] = []

        for train_idx, test_idx in tscv.split(X):
            # Cada split respeta el orden temporal: entrenar con pasado y evaluar en futuro.
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Crear un modelo nuevo para cada fold y evitar contaminar el estado entre splits.
            model = self._build_model(self.model_name, use_early_stopping=False)
            model.fit(X_train, y_train)

            # Calcular métricas de clasificación y probabilidad para ese fold.
            probs = model.predict_proba(X_test)
            preds = model.predict(X_test)
            accs.append(accuracy_score(y_test, preds))
            losses.append(log_loss(y_test, probs, labels=[0, 1, 2]))

        return {
            "cv_accuracy": float(sum(accs) / len(accs)),
            "cv_log_loss": float(sum(losses) / len(losses)),
        }

    def train(self, data: MatchData) -> Dict[str, float]:
        if data.df.empty:
            raise ValueError("No hay partidos para entrenar.")

        # Trabajar sobre una copia para no mutar los datos originales.
        df = data.df.copy()

        # Descartar los primeros partidos como warm-up porque las features históricas aún son débiles.
        df = df.iloc[50:].reset_index(drop=True)
        if df.empty:
            raise ValueError("No hay suficientes partidos despues del warm-up.")

        # Separar por temporadas para hacer un split temporal limpio: train / val / test.
        seasons = sorted(df["season"].unique())
        if len(seasons) < 3:
            raise ValueError("Se requieren al menos 3 temporadas para split temporal.")

        train_seasons = seasons[:-2]
        val_season = seasons[-2]
        test_season = seasons[-1]

        train_df = df[df["season"].isin(train_seasons)].reset_index(drop=True)
        val_df = df[df["season"] == val_season].reset_index(drop=True)
        test_df = df[df["season"] == test_season].reset_index(drop=True)

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("El split temporal genero un conjunto vacio.")

        # Preparar matrices de features y etiquetas para cada subconjunto.
        X_train = train_df[self.features]
        y_train = train_df["result"].map(LABEL_TO_INT)
        X_val = val_df[self.features]
        y_val = val_df["result"].map(LABEL_TO_INT)
        X_test = test_df[self.features]
        y_test = test_df["result"].map(LABEL_TO_INT)

        if y_train.isna().any() or y_val.isna().any() or y_test.isna().any():
            raise ValueError("Se encontraron etiquetas inesperadas fuera de H/D/A.")

        # XGBoost aprovecha el conjunto de validación para early stopping; los demás no.
        if self.model_name == "xgboost":
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

        # Intentar calibrar las probabilidades con el split de validación.
        # Primero isotonic, y si falla, sigmoid. Si ambos fallan, se conserva el modelo original.
        try:
            calibrator = CalibratedClassifierCV(self.model, method="isotonic", cv="prefit")
            calibrator.fit(X_val, y_val)
            self.model = calibrator
        except Exception:
            try:
                calibrator = CalibratedClassifierCV(self.model, method="sigmoid", cv="prefit")
                calibrator.fit(X_val, y_val)
                self.model = calibrator
            except Exception:
                pass

        # Evaluar el modelo final en el conjunto de test temporal.
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, probs, labels=[0, 1, 2])

        # Añadir una validación cruzada temporal adicional para tener métricas más robustas.
        rolling = self._rolling_cv(df)

        self.trained = True
        return {
            "accuracy": float(acc),
            "log_loss": float(loss),
            "train_seasons": len(train_seasons),
            "val_season": val_season,
            "test_season": test_season,
            **rolling,
        }

    def predict(self, features_row: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
     
        if not self.trained:
            raise RuntimeError("El modelo no esta entrenado.")

        # Obtener la clase predicha y las probabilidades por cada resultado posible.
        pred_int = int(self.model.predict(features_row)[0])
        probs = self.model.predict_proba(features_row)[0]

        # Reconstruir el diccionario de probabilidades usando las clases reales del modelo.
        probs_map = {INT_TO_LABEL[i]: float(probs[idx]) for idx, i in enumerate(self.model.classes_)}
        return INT_TO_LABEL[pred_int], probs_map


class PredictionService:

    def __init__(self, model_name: str = "xgboost") -> None:
        # Encapsula todo el pipeline: features + modelo de predicción.
        self.model_name = model_name
        self.feature_engineer = FeatureEngineer()
        self.predictor = MatchPredictor(model_name=model_name)
        # Aquí se guarda el MatchData enriquecido tras el entrenamiento.
        self.data: Optional[MatchData] = None

    def fit(self, data: MatchData) -> Dict[str, float]:
        """Calcula features sobre los datos y entrena el predictor."""
        self.data = self.feature_engineer.compute_features(data)
        return self.predictor.train(self.data)

    def available_teams(self) -> List[str]:
        """Lista los equipos disponibles en los datos ya cargados."""
        if self.data is None:
            return []
        teams = sorted(set(self.data.df["home_team"]).union(self.data.df["away_team"]))
        return teams

    def latest_team_features(self, team_name: str) -> Dict[str, float]:
        """Recupera las últimas features conocidas de un equipo para predecir su próximo partido."""
        if self.data is None:
            raise RuntimeError("Datos no disponibles.")

        df = self.data.df
        # Buscar todos los partidos en los que aparece el equipo, como local o visitante.
        team_matches = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)]
        if team_matches.empty:
            raise ValueError(f"Equipo no encontrado: {team_name}")

        # Usar el partido más reciente para extraer el estado actual del equipo.
        last_match = team_matches.iloc[-1]
        if last_match["home_team"] == team_name:
            return {
                "elo": float(last_match["home_elo"]),
                "form": float(last_match["home_form"]),
                "gf": float(last_match["home_gf_avg"]),
                "ga": float(last_match["home_ga_avg"]),
            }

        return {
            "elo": float(last_match["away_elo"]),
            "form": float(last_match["away_form"]),
            "gf": float(last_match["away_gf_avg"]),
            "ga": float(last_match["away_ga_avg"]),
        }

    def predict_match(self, home_team: str, away_team: str) -> Dict[str, object]:
        """Construye las features de un partido futuro y devuelve su predicción."""
        if self.data is None:
            raise RuntimeError("Datos no disponibles.")

        # Obtener el estado reciente de ambos equipos a partir del histórico entrenado.
        home_stats = self.latest_team_features(home_team)
        away_stats = self.latest_team_features(away_team)

        # Construir una fila artificial con las mismas columnas que se usaron al entrenar.
        match_features = pd.DataFrame(
            {
                "home_elo": [home_stats["elo"]],
                "away_elo": [away_stats["elo"]],
                "home_form": [home_stats["form"]],
                "away_form": [away_stats["form"]],
                "home_gf_avg": [home_stats["gf"]],
                "away_gf_avg": [away_stats["gf"]],
                "home_ga_avg": [home_stats["ga"]],
                "away_ga_avg": [away_stats["ga"]],
            }
        )

        # Delegar la predicción al modelo ya entrenado.
        label, probs = self.predictor.predict(match_features)
        return {
            "label": label,
            "probabilities": probs,
            "home_team": home_team,
            "away_team": away_team,
        }
