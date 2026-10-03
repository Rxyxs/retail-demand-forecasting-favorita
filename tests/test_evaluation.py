"""Tests de las metricas y del simulador de inventario.

Todo lo de aca es aritmetica exacta o una identidad de libro, a proposito: estas
funciones son las que deciden que modelo gana, asi que un error de signo en la
pinball o en el cuantil critico invertiria la conclusion del proyecto entero sin
que ninguna corrida se caiga.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    mae,
    mase,
    newsvendor_quantile,
    pinball_loss,
    rmsle,
    simulate_inventory,
)


def test_rmsle_is_zero_for_a_perfect_forecast():
    y = np.array([0.0, 3.0, 17.0, 250.0])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_handles_zeros_without_blowing_up():
    # El 25,8% de las observaciones validas del panel son cero. Una metrica que
    # divide por el valor real (MAPE) es indefinida aca; RMSLE no.
    assert np.isfinite(rmsle(np.zeros(5), np.array([0.0, 1.0, 2.0, 0.0, 3.0])))


def test_rmsle_penalises_relative_error_not_absolute():
    # Errar 2 sobre una venta de 3 tiene que pesar mas que errar 2 sobre 300.
    small = rmsle(np.array([3.0]), np.array([5.0]))
    large = rmsle(np.array([300.0]), np.array([302.0]))
    assert small > large


def test_rmsle_clamps_negative_predictions():
    # log1p de un negativo es NaN; el modelo puede predecir negativo y la metrica
    # no debe propagarlo en silencio.
    assert np.isfinite(rmsle(np.array([2.0]), np.array([-5.0])))


def test_pinball_at_median_is_half_the_absolute_error():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    assert pinball_loss(y, p, 0.5) == pytest.approx(mae(y, p) / 2.0)


def test_pinball_is_asymmetric_in_the_direction_the_quantile_says():
    y = np.array([100.0])
    short = pinball_loss(y, np.array([90.0]), 0.9)   # quedarse corto
    over = pinball_loss(y, np.array([110.0]), 0.9)   # pasarse
    # Con quantile=0.9, quedarse corto debe costar 9 veces pasarse.
    assert short == pytest.approx(9.0 * over)


def test_newsvendor_quantile_is_the_critical_ratio():
    assert newsvendor_quantile(4.0, 1.0) == pytest.approx(0.8)
    assert newsvendor_quantile(1.0, 1.0) == pytest.approx(0.5)
    assert newsvendor_quantile(9.0, 1.0) == pytest.approx(0.9)


def test_newsvendor_quantile_rejects_degenerate_costs():
    with pytest.raises(ValueError):
        newsvendor_quantile(0.0, 0.0)


def test_mase_of_the_seasonal_naive_is_one_by_construction():
    # Serie con ciclo semanal perfecto mas una tendencia, de modo que el naive
    # estacional comete un error constante: su MASE tiene que dar exactamente 1.
    rng = np.random.default_rng(0)
    week = rng.uniform(5, 50, size=7)
    train = np.concatenate([week + i for i in range(10)])
    y_true = week + 10
    y_pred = week + 9  # el naive repite la semana anterior
    assert mase(y_true, y_pred, train, season=7) == pytest.approx(1.0)


def test_inventory_splits_the_two_errors_and_never_mixes_them():
    demand = np.array([10.0, 10.0, 10.0])
    order = np.array([8.0, 10.0, 15.0])  # falta 2, exacto, sobra 5
    o = simulate_inventory(demand, order, unit_understock_cost=4.0, unit_overstock_cost=1.0)

    assert o.units_short == pytest.approx(2.0)
    assert o.units_excess == pytest.approx(5.0)
    assert o.understock_cost == pytest.approx(8.0)
    assert o.overstock_cost == pytest.approx(5.0)
    assert o.total_cost == pytest.approx(13.0)
    assert o.service_level == pytest.approx(2 / 3)       # 2 de 3 dias sin quiebre
    assert o.fill_rate == pytest.approx(28.0 / 30.0)     # 28 de 30 unidades servidas


def test_ordering_the_mean_leaves_service_near_one_half_on_symmetric_demand():
    # El argumento central del proyecto, como test: con demanda simetrica, pedir
    # la media deja quiebre aproximadamente la mitad de los dias, por mucho que
    # el pronostico de la media sea exacto.
    rng = np.random.default_rng(7)
    demand = rng.normal(100, 20, size=20_000)
    o = simulate_inventory(demand, np.full_like(demand, 100.0), 4.0, 1.0)
    assert 0.45 < o.service_level < 0.55


def test_ordering_the_critical_quantile_hits_its_target_service():
    rng = np.random.default_rng(7)
    demand = rng.normal(100, 20, size=20_000)
    q = newsvendor_quantile(4.0, 1.0)  # 0.80
    order = np.full_like(demand, float(np.quantile(demand, q)))
    o = simulate_inventory(demand, order, 4.0, 1.0)
    assert o.service_level == pytest.approx(q, abs=0.02)


def test_inventory_treats_negative_orders_as_zero():
    o = simulate_inventory(np.array([5.0]), np.array([-3.0]), 4.0, 1.0)
    assert o.units_short == pytest.approx(5.0)
    assert o.units_excess == pytest.approx(0.0)
