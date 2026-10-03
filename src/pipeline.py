"""Pipeline de punta a punta: `python -m src.pipeline`.

Divide el panel por fecha (no al azar), entrena las tres variantes, y evalua
cada una dos veces: como pronostico y como decision de compra. Escribe un JSON
con todo lo que el README cita, para que ninguna cifra del documento este
escrita a mano.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from src.data import build_panel, intermittency_by_series, summarize
from src.evaluation import (
    InventoryOutcome,
    mase,
    newsvendor_quantile,
    pinball_loss,
    rmsle,
    simulate_inventory,
    mae,
)
from src.models import FEATURES, add_features, predict, seasonal_naive, train_lightgbm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

# Ultimos 16 dias como validacion: es el horizonte que la competencia pide
# pronosticar, asi que el backtest mide lo mismo que el problema plantea.
VALID_DAYS = 16

# Razon de costos del escenario: quedarse corto cuesta 4 veces lo que sobrar.
# Es una suposicion declarada, no un dato del dataset -- en abarrotes de alta
# rotacion y buen margen la venta perdida domina a la merma, y el analisis de
# sensibilidad sobre esta razon esta en los resultados.
UNDERSTOCK_COST = 4.0
OVERSTOCK_COST = 1.0


def split_by_date(panel: pl.DataFrame, valid_days: int = VALID_DAYS):
    cutoff = panel["date"].max() - pl.duration(days=valid_days - 1)
    return panel.filter(pl.col("date") < cutoff), panel.filter(pl.col("date") >= cutoff)


def _outcome_dict(o: InventoryOutcome) -> dict:
    return {
        "order_total": o.order_quantity_total,
        "demand_total": o.demand_total,
        "units_short": o.units_short,
        "units_excess": o.units_excess,
        "understock_cost": o.understock_cost,
        "overstock_cost": o.overstock_cost,
        "total_cost": o.total_cost,
        "service_level": o.service_level,
        "fill_rate": o.fill_rate,
    }


def run() -> dict:
    print("=== 1/5 Cargando panel y marcando filas pre-apertura")
    panel = build_panel()
    s = summarize(panel)
    print(f"  {s.total_rows:,} filas, {s.series} series, {s.date_min} -> {s.date_max}")
    print(f"  pre-apertura: {s.pre_opening_rows:,} ({s.pre_opening_share:.1%}), "
          f"{s.zeros_explained_by_openings:.1%} de todos los ceros")
    print(f"  ceros: {s.zero_share_raw:.1%} crudo -> {s.zero_share_after_opening:.1%} sobre filas validas")

    inter_raw = intermittency_by_series(panel, exclude_pre_opening=False)
    inter = intermittency_by_series(panel, exclude_pre_opening=True)
    severe_raw = int((inter_raw["zero_fraction"] > 0.9).sum())
    severe = int((inter["zero_fraction"] > 0.9).sum())
    print(f"  series >90% ceros: {severe_raw} contando pre-apertura -> {severe} sin contarla")

    print("\n=== 2/5 Construyendo features (rezagos dentro de cada serie)")
    # Las filas pre-apertura se excluyen del entrenamiento: no son observaciones
    # de demanda, son ausencia de local.
    featured = add_features(panel.filter(~pl.col("pre_opening")))
    train, valid = split_by_date(featured)
    valid = valid.drop_nulls(subset=FEATURES)
    print(f"  train: {train.height:,} filas hasta {train['date'].max()}")
    print(f"  valid: {valid.height:,} filas desde {valid['date'].min()} ({VALID_DAYS} dias)")

    y_valid = valid["sales"].to_numpy()

    print("\n=== 3/5 Entrenando modelos")
    print("  linea base: naive estacional (mismo dia de la semana anterior)")
    pred_naive = seasonal_naive(train, valid)

    print("  LightGBM puntual (objetivo: la media)")
    model_mean = train_lightgbm(train, objective="regression")
    pred_mean = predict(model_mean, valid)

    critical_q = newsvendor_quantile(UNDERSTOCK_COST, OVERSTOCK_COST)
    print(f"  LightGBM cuantil {critical_q:.0%} (objetivo: la decision de pedido)")
    model_q = train_lightgbm(train, objective="quantile", alpha=critical_q)
    pred_q = predict(model_q, valid)

    print("\n=== 4/5 Exactitud")
    y_train_series = train["sales"].to_numpy()
    accuracy = {}
    for name, pred in [("seasonal_naive", pred_naive), ("lgbm_mean", pred_mean),
                       ("lgbm_quantile", pred_q)]:
        accuracy[name] = {
            "rmsle": rmsle(y_valid, pred),
            "mae": mae(y_valid, pred),
            "mase": mase(y_valid, pred, y_train_series),
            "pinball_at_critical_q": pinball_loss(y_valid, pred, critical_q),
        }
        a = accuracy[name]
        print(f"  {name:16s} RMSLE={a['rmsle']:.4f}  MAE={a['mae']:7.2f}  "
              f"MASE={a['mase']:.3f}  pinball@{critical_q:.0%}={a['pinball_at_critical_q']:.3f}")

    print("\n=== 5/5 La misma prediccion, evaluada como decision de compra")
    print(f"  costo de quiebre {UNDERSTOCK_COST:.0f}x el de sobrestock -> cuantil critico {critical_q:.0%}")
    inventory = {}
    for name, pred in [("seasonal_naive", pred_naive), ("lgbm_mean", pred_mean),
                       ("lgbm_quantile", pred_q)]:
        o = simulate_inventory(y_valid, pred, UNDERSTOCK_COST, OVERSTOCK_COST)
        inventory[name] = _outcome_dict(o)
        print(f"  {name:16s} servicio={o.service_level:6.1%}  fill={o.fill_rate:6.1%}  "
              f"costo={o.total_cost:12,.0f}  (quiebre {o.understock_cost:,.0f} / sobra {o.overstock_cost:,.0f})")

    # Sensibilidad: el cuantil optimo depende de una razon de costos que es una
    # suposicion. Si la conclusion se da vuelta con una razon razonable, hay que
    # decirlo.
    print("\n  sensibilidad a la razon de costos:")
    sensitivity = []
    for ratio in (1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0):
        q = newsvendor_quantile(ratio, 1.0)
        m = train_lightgbm(train, objective="quantile", alpha=q) if ratio != UNDERSTOCK_COST else model_q
        p = predict(m, valid) if ratio != UNDERSTOCK_COST else pred_q
        o_q = simulate_inventory(y_valid, p, ratio, 1.0)
        o_m = simulate_inventory(y_valid, pred_mean, ratio, 1.0)
        saving = 1 - o_q.total_cost / o_m.total_cost if o_m.total_cost else 0.0
        sensitivity.append({
            "cost_ratio": ratio, "critical_quantile": q,
            "quantile_cost": o_q.total_cost, "mean_cost": o_m.total_cost,
            "saving_vs_mean": saving,
            "quantile_service": o_q.service_level, "mean_service": o_m.service_level,
        })
        print(f"    {ratio:g}:1 -> cuantil {q:.0%}  costo cuantil={o_q.total_cost:12,.0f}  "
              f"vs media={o_m.total_cost:12,.0f}  ahorro={saving:+6.1%}")

    report = {
        "panel": {
            "total_rows": s.total_rows, "series": s.series, "stores": s.stores,
            "families": s.families, "date_min": s.date_min, "date_max": s.date_max,
            "pre_opening_rows": s.pre_opening_rows,
            "pre_opening_share": s.pre_opening_share,
            "zero_rows": s.zero_rows,
            "zero_rows_pre_opening": s.zero_rows_pre_opening,
            "zero_share_raw": s.zero_share_raw,
            "zero_share_after_opening": s.zero_share_after_opening,
            "zeros_explained_by_openings": s.zeros_explained_by_openings,
            "severe_intermittent_raw": severe_raw,
            "severe_intermittent_clean": severe,
        },
        "split": {
            "valid_days": VALID_DAYS,
            "train_rows": train.height, "valid_rows": valid.height,
            "train_end": str(train["date"].max()), "valid_start": str(valid["date"].min()),
        },
        "costs": {"understock": UNDERSTOCK_COST, "overstock": OVERSTOCK_COST,
                  "critical_quantile": critical_q},
        "accuracy": accuracy,
        "inventory": inventory,
        "sensitivity": sensitivity,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "results.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReporte escrito en {path}")
    return report


if __name__ == "__main__":
    run()
