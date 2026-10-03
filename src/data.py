"""Carga del panel de Favorita y las decisiones de limpieza que lo definen.

El dataset trae una trampa que no se ve en el esquema: 8 de los 54 locales
abren despues del inicio del panel, y mientras no existen aparecen con ventas
en cero para las 33 familias, todos los dias. Son 222.057 filas (7,4% del
panel) y el 23,6% de todos los ceros del dataset. Un modelo entrenado sobre
ellas aprende "este local no vende" de un local que todavia no habia abierto,
y una metrica calculada sobre ellas se acredita aciertos por predecir cero
donde no habia nada que predecir.

Este modulo las marca explicitamente en vez de borrarlas en silencio, para que
cada paso posterior decida si las usa y el README pueda mostrar las dos
versiones del mismo numero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Terremoto de Ecuador, 16 de abril de 2016 (magnitud 7.8). Altero la demanda
# de forma abrupta -- compras de panico de agua y alimentos, locales cerrados en
# la costa -- durante semanas. No se elimina: se marca, porque un modelo que no
# sabe que paso lo aprende como estacionalidad y lo repite cada abril.
EARTHQUAKE_DATE = pl.date(2016, 4, 16)
EARTHQUAKE_WINDOW_DAYS = 30


@dataclass(frozen=True)
class PanelSummary:
    """Las cifras que el README cita, calculadas una sola vez y en un lugar."""

    total_rows: int
    pre_opening_rows: int
    zero_rows: int
    zero_rows_pre_opening: int
    series: int
    stores: int
    families: int
    date_min: str
    date_max: str

    @property
    def pre_opening_share(self) -> float:
        return self.pre_opening_rows / self.total_rows

    @property
    def zero_share_raw(self) -> float:
        return self.zero_rows / self.total_rows

    @property
    def zero_share_after_opening(self) -> float:
        valid = self.total_rows - self.pre_opening_rows
        return (self.zero_rows - self.zero_rows_pre_opening) / valid

    @property
    def zeros_explained_by_openings(self) -> float:
        return self.zero_rows_pre_opening / self.zero_rows


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    """Lee los CSV crudos tal cual vienen de Kaggle, sin transformarlos."""
    needed = ["train", "stores", "oil", "holidays_events", "transactions"]
    missing = [n for n in needed if not (raw_dir / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {missing} en {raw_dir}. Descargalos primero:\n"
            "  kaggle competitions download -c store-sales-time-series-forecasting "
            "-p data/raw --unzip"
        )
    return {n: pl.read_csv(raw_dir / f"{n}.csv", try_parse_dates=True) for n in needed}


def store_opening_dates(train: pl.DataFrame) -> pl.DataFrame:
    """Primera fecha con venta estrictamente positiva por local.

    Se usa la primera venta y no la primera fila porque la fila existe desde el
    inicio del panel para todos los locales: el dataset esta completado con
    ceros, no recortado por fecha de apertura.
    """
    return (
        train.filter(pl.col("sales") > 0)
        .group_by("store_nbr")
        .agg(pl.col("date").min().alias("opened_on"))
        .sort("store_nbr")
    )


def build_panel(raw: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """Panel diario local x familia con covariables y banderas de calidad."""
    raw = raw or load_raw()
    train = raw["train"]

    openings = store_opening_dates(train)

    # El precio del petroleo viene con huecos (solo dias habiles de mercado) y
    # algunos nulos. Se rellena hacia adelante: el precio relevante para un
    # domingo es el del viernes anterior, no un promedio que mira al futuro.
    oil = (
        raw["oil"]
        .sort("date")
        .with_columns(pl.col("dcoilwtico").forward_fill().alias("oil_price"))
        .select(["date", "oil_price"])
    )

    # Solo feriados efectivamente celebrados: las filas transferidas marcan el
    # dia que se movio, no el que se trabaja.
    holidays = (
        raw["holidays_events"]
        .filter((pl.col("transferred") == False) & (pl.col("type") != "Work Day"))  # noqa: E712
        .select(["date", "type", "locale"])
        .unique(subset=["date"], keep="first")
        .with_columns(pl.lit(1).alias("is_holiday"))
        .select(["date", "is_holiday", "locale"])
        .rename({"locale": "holiday_locale"})
    )

    panel = (
        train.join(openings, on="store_nbr", how="left")
        .join(raw["stores"], on="store_nbr", how="left")
        .join(oil, on="date", how="left")
        .join(holidays, on="date", how="left")
        .with_columns(
            pl.col("is_holiday").fill_null(0),
            pl.col("oil_price").forward_fill().backward_fill(),
            # La bandera central de este modulo.
            (pl.col("date") < pl.col("opened_on")).alias("pre_opening"),
            (
                (pl.col("date") >= EARTHQUAKE_DATE)
                & (pl.col("date") < EARTHQUAKE_DATE.dt.offset_by(f"{EARTHQUAKE_WINDOW_DAYS}d"))
            ).alias("post_earthquake"),
        )
        .sort(["store_nbr", "family", "date"])
    )
    return panel


def summarize(panel: pl.DataFrame) -> PanelSummary:
    """Las cifras de encabezado, derivadas del panel y no escritas a mano."""
    zero = pl.col("sales") == 0
    return PanelSummary(
        total_rows=panel.height,
        pre_opening_rows=int(panel.select(pl.col("pre_opening").sum()).item()),
        zero_rows=int(panel.select(zero.sum()).item()),
        zero_rows_pre_opening=int(
            panel.filter(pl.col("pre_opening")).select(zero.sum()).item()
        ),
        series=panel.select(["store_nbr", "family"]).unique().height,
        stores=panel["store_nbr"].n_unique(),
        families=panel["family"].n_unique(),
        date_min=str(panel["date"].min()),
        date_max=str(panel["date"].max()),
    )


def intermittency_by_series(panel: pl.DataFrame, exclude_pre_opening: bool = True) -> pl.DataFrame:
    """Fraccion de dias sin venta por serie, que es lo que decide el metodo.

    Una serie con 90% de ceros no se pronostica con el mismo instrumento que una
    que vende todos los dias, y el promedio del panel esconde esa diferencia.
    """
    df = panel.filter(~pl.col("pre_opening")) if exclude_pre_opening else panel
    return (
        df.group_by(["store_nbr", "family"])
        .agg(
            (pl.col("sales") == 0).mean().alias("zero_fraction"),
            pl.col("sales").mean().alias("mean_sales"),
            pl.col("sales").len().alias("observations"),
        )
        .sort("zero_fraction", descending=True)
    )
