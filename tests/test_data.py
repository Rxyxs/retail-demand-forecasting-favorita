"""Tests de la carga del panel, con foco en la deteccion de pre-apertura.

Se construyen paneles sinteticos chicos donde la respuesta correcta se conoce de
antemano, en vez de verificar contra el dataset real: un test que depende de los
3 millones de filas descargadas no corre en CI y falla por motivos equivocados.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.data import build_panel, intermittency_by_series, store_opening_dates, summarize


def _toy_train() -> pl.DataFrame:
    """Dos locales: el 1 abre el dia uno, el 2 recien al cuarto dia."""
    dates = pl.date_range(dt.date(2020, 1, 1), dt.date(2020, 1, 6), "1d", eager=True)
    rows = []
    for d_i, d in enumerate(dates):
        for store, opens_at in ((1, 0), (2, 3)):
            for fam in ("A", "B"):
                sales = 0.0 if d_i < opens_at else float(10 + d_i)
                rows.append({"id": len(rows), "date": d, "store_nbr": store,
                             "family": fam, "sales": sales, "onpromotion": 0})
    return pl.DataFrame(rows)


def _toy_raw() -> dict[str, pl.DataFrame]:
    train = _toy_train()
    return {
        "train": train,
        "stores": pl.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Guayaquil"],
                                 "state": ["P", "G"], "type": ["A", "B"], "cluster": [1, 2]}),
        "oil": pl.DataFrame({"date": [dt.date(2020, 1, 1), dt.date(2020, 1, 4)],
                              "dcoilwtico": [50.0, 55.0]}),
        "holidays_events": pl.DataFrame({
            "date": [dt.date(2020, 1, 2)], "type": ["Holiday"], "locale": ["National"],
            "locale_name": ["Ecuador"], "description": ["x"], "transferred": [False],
        }),
        "transactions": pl.DataFrame({"date": [dt.date(2020, 1, 1)], "store_nbr": [1],
                                       "transactions": [100]}),
    }


def test_opening_date_is_the_first_positive_sale_not_the_first_row():
    # El panel trae filas para el local 2 desde el dia uno, con ventas en cero.
    # Tomar la primera fila daria 2020-01-01 y borraria todo el hallazgo.
    openings = store_opening_dates(_toy_train())
    by_store = {r["store_nbr"]: r["opened_on"] for r in openings.to_dicts()}
    assert str(by_store[1]) == "2020-01-01"
    assert str(by_store[2]) == "2020-01-04"


def test_pre_opening_rows_are_flagged_and_are_all_zero():
    panel = build_panel(_toy_raw())
    pre = panel.filter(pl.col("pre_opening"))
    # Local 2, dias 1 a 3, dos familias = 6 filas.
    assert pre.height == 6
    assert (pre["sales"] == 0).all()
    assert set(pre["store_nbr"].unique().to_list()) == {2}


def test_summary_separates_the_two_zero_rates():
    s = summarize(build_panel(_toy_raw()))
    # 24 filas totales, 6 pre-apertura, y los unicos ceros son esas 6.
    assert s.total_rows == 24
    assert s.pre_opening_rows == 6
    assert s.zero_rows == 6
    assert s.zeros_explained_by_openings == pytest.approx(1.0)
    assert s.zero_share_raw == pytest.approx(6 / 24)
    # Sacando las filas pre-apertura no queda ningun cero.
    assert s.zero_share_after_opening == pytest.approx(0.0)


def test_intermittency_changes_when_pre_opening_is_counted():
    panel = build_panel(_toy_raw())
    con = intermittency_by_series(panel, exclude_pre_opening=False)
    sin = intermittency_by_series(panel, exclude_pre_opening=True)
    store2_con = con.filter(pl.col("store_nbr") == 2)["zero_fraction"].max()
    store2_sin = sin.filter(pl.col("store_nbr") == 2)["zero_fraction"].max()
    # Contando pre-apertura el local 2 parece intermitente; sin contarla, no.
    assert store2_con == pytest.approx(0.5)
    assert store2_sin == pytest.approx(0.0)


def test_oil_price_is_forward_filled_never_interpolated_from_the_future():
    panel = build_panel(_toy_raw())
    by_date = {str(r["date"]): r["oil_price"] for r in
               panel.filter(pl.col("store_nbr") == 1).filter(pl.col("family") == "A").to_dicts()}
    # El 2 y el 3 de enero no tienen precio propio: heredan el del 1, no el del 4.
    assert by_date["2020-01-02"] == pytest.approx(50.0)
    assert by_date["2020-01-03"] == pytest.approx(50.0)
    assert by_date["2020-01-04"] == pytest.approx(55.0)


def test_missing_files_raise_a_message_that_says_how_to_fix_it():
    from pathlib import Path
    with pytest.raises(FileNotFoundError, match="kaggle competitions download"):
        build_panel.__wrapped__ if hasattr(build_panel, "__wrapped__") else None
        from src.data import load_raw
        load_raw(Path("no/existe"))
