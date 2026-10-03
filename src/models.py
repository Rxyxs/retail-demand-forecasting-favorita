"""Modelos, con una linea base que hay que ganarle de verdad.

El orden importa: primero el naive estacional, que en retail es un rival serio
porque la demanda semanal es fuertemente periodica, y recien despues los
modelos. Un gradient boosting que no le gana al naive estacional no esta
aprendiendo demanda, esta aprendiendo el promedio.
"""
from __future__ import annotations

import numpy as np
import polars as pl

SEASON = 7  # la semana es el ciclo dominante en supermercados


def add_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Calendario, rezagos y medias moviles, todos calculados hacia atras.

    Cada rezago se toma dentro de su propia serie (local x familia) y desplazado
    al menos `horizon` dias, de modo que ninguna fila use informacion que no
    estaria disponible al momento de decidir el pedido. Es la diferencia entre
    un backtest y una filtracion.
    """
    return (
        panel.sort(["store_nbr", "family", "date"])
        .with_columns(
            pl.col("date").dt.weekday().alias("dow"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day_of_month"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
            # Quincena y fin de mes: en Ecuador, como en Chile, el sueldo se paga
            # el 15 y el ultimo dia habil, y la demanda de supermercado lo sigue.
            ((pl.col("date").dt.day() == 15) | (pl.col("date").dt.month_end().dt.day() == pl.col("date").dt.day()))
            .cast(pl.Int8)
            .alias("payday"),
        )
        .with_columns(
            [
                pl.col("sales").shift(lag).over(["store_nbr", "family"]).alias(f"lag_{lag}")
                for lag in (7, 14, 21, 28)
            ]
            + [
                pl.col("sales").shift(7).rolling_mean(window).over(["store_nbr", "family"]).alias(f"roll_mean_{window}")
                for window in (7, 28)
            ]
            + [
                pl.col("sales").shift(7).rolling_std(28).over(["store_nbr", "family"]).alias("roll_std_28"),
                (pl.col("sales").shift(7) == 0).cast(pl.Float64).rolling_mean(28)
                .over(["store_nbr", "family"]).alias("zero_rate_28"),
                pl.col("onpromotion").shift(0).over(["store_nbr", "family"]).alias("promo_today"),
            ]
        )
    )


FEATURES = [
    "dow", "month", "day_of_month", "day_of_year", "payday",
    "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_28", "zero_rate_28",
    "promo_today", "onpromotion", "oil_price", "is_holiday", "cluster",
]


def seasonal_naive(train: pl.DataFrame, valid: pl.DataFrame, season: int = SEASON) -> np.ndarray:
    """Repite el mismo dia de la semana anterior disponible.

    Es la linea base honesta en retail: captura el ciclo semanal entero sin
    estimar nada, y cualquier modelo tiene que justificarse contra ella.
    """
    last = (
        train.sort("date")
        .group_by(["store_nbr", "family", train["date"].dt.weekday().alias("dow")])
        .agg(pl.col("sales").last().alias("naive"))
    )
    merged = valid.with_columns(pl.col("date").dt.weekday().alias("dow")).join(
        last, on=["store_nbr", "family", "dow"], how="left"
    )
    return merged["naive"].fill_null(0.0).to_numpy()


def train_lightgbm(
    train: pl.DataFrame,
    features: list[str] = None,
    objective: str = "regression",
    alpha: float | None = None,
    seed: int = 42,
):
    """LightGBM sobre log1p(ventas), con objetivo puntual o de cuantil.

    Se entrena en espacio logaritmico porque la metrica de la competencia es
    RMSLE y porque la demanda por familia abarca tres ordenes de magnitud; en
    espacio original las familias grandes dominarian el gradiente.
    """
    import lightgbm as lgb

    features = features or FEATURES
    df = train.drop_nulls(subset=features + ["sales"])
    X = df.select(features).to_pandas()
    y = np.log1p(np.maximum(df["sales"].to_numpy(), 0.0))

    params = {
        "objective": objective,
        "learning_rate": 0.06,
        "num_leaves": 96,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
        "num_threads": 0,
    }
    if objective == "quantile":
        if alpha is None:
            raise ValueError("objective='quantile' requiere alpha")
        params["alpha"] = alpha

    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=450)


def predict(model, frame: pl.DataFrame, features: list[str] = None) -> np.ndarray:
    """Devuelve la prediccion en unidades, no en logaritmo."""
    features = features or FEATURES
    X = frame.select(features).to_pandas()
    return np.maximum(np.expm1(model.predict(X)), 0.0)
