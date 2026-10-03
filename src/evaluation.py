"""Metricas, y la razon por la que la metrica de la competencia no alcanza.

La competencia puntua con RMSLE sobre la prediccion puntual. Eso ordena modelos,
pero no es la decision que un retailer toma: nadie "pide la media". Se pide una
cantidad, y equivocarse por debajo (quiebre de stock, venta perdida) no cuesta lo
mismo que equivocarse por arriba (sobrestock, merma, capital inmovilizado).

Ese es un problema de newsvendor, y su solucion no es la media sino un cuantil
determinado por la razon de costos. Este modulo implementa las dos familias:
las metricas de exactitud que ordenan modelos, y las de decision que dicen si el
pronostico sirve para pedir mercaderia.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Exactitud
# ---------------------------------------------------------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Metrica oficial de la competencia.

    Se aplica sobre log1p, asi que castiga el error relativo y no el absoluto:
    errar 2 unidades sobre una familia que vende 3 pesa mucho mas que errar 2
    sobre una que vende 300. Para un panel donde las familias van de ventas de
    un digito a miles, esa es la eleccion correcta -- y es tambien la razon de
    que un RMSE crudo sobre este panel este dominado por media docena de series
    grandes.
    """
    y_true = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 7) -> float:
    """MAE escalado por el MAE del naive estacional sobre el train.

    Se usa en vez de MAPE porque MAPE es indefinido con ventas en cero, y en este
    panel el 25,8% de las observaciones validas lo son. Un MASE de 1 significa
    "igual de bueno que repetir la semana pasada"; por encima de 1, peor.
    """
    y_train = np.asarray(y_train, float)
    if y_train.size <= season:
        return float("nan")
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale


# ---------------------------------------------------------------------------
# Decision: cuantiles e inventario
# ---------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Perdida pinball, que es la funcion que un pronostico de cuantil minimiza.

    Es asimetrica a proposito: con quantile=0.9 quedarse corto cuesta 9 veces mas
    que pasarse, que es justamente la estructura de costos de un quiebre frente a
    un sobrestock cuando el margen es alto.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    delta = y_true - y_pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def newsvendor_quantile(unit_understock_cost: float, unit_overstock_cost: float) -> float:
    """Cuantil critico: Cu / (Cu + Co).

    Es el resultado clasico del newsvendor y lo que conecta una razon de costos
    con un pronostico. Si quedarse corto cuesta 4 veces lo que sobrar, el pedido
    optimo es el percentil 80 de la demanda, no su media -- y pedir la media deja
    quiebre uno de cada dos dias por construccion.
    """
    total = unit_understock_cost + unit_overstock_cost
    if total <= 0:
        raise ValueError("Los costos unitarios deben sumar un valor positivo")
    return unit_understock_cost / total


@dataclass(frozen=True)
class InventoryOutcome:
    """Lo que un comprador mira: cuanto se perdio y por que lado."""

    order_quantity_total: float
    demand_total: float
    units_short: float
    units_excess: float
    understock_cost: float
    overstock_cost: float
    service_level: float  # fraccion de dias servidos completos
    fill_rate: float      # fraccion de unidades demandadas efectivamente servidas

    @property
    def total_cost(self) -> float:
        return self.understock_cost + self.overstock_cost


def simulate_inventory(
    demand: np.ndarray,
    order: np.ndarray,
    unit_understock_cost: float,
    unit_overstock_cost: float,
) -> InventoryOutcome:
    """Evalua un pronostico como lo que es: una decision de cuanto pedir.

    Modelo de un periodo (newsvendor), sin arrastre de inventario entre dias.
    Es una simplificacion y conviene nombrarla: con arrastre, el sobrestock de
    hoy amortigua el quiebre de manana y las dos columnas de costo bajan. Lo que
    el modelo si captura, y es el punto, es la asimetria entre los dos errores.
    """
    demand = np.maximum(np.asarray(demand, float), 0.0)
    order = np.maximum(np.asarray(order, float), 0.0)

    short = np.maximum(demand - order, 0.0)
    excess = np.maximum(order - demand, 0.0)
    served = np.minimum(demand, order)

    demand_total = float(demand.sum())
    return InventoryOutcome(
        order_quantity_total=float(order.sum()),
        demand_total=demand_total,
        units_short=float(short.sum()),
        units_excess=float(excess.sum()),
        understock_cost=float(short.sum() * unit_understock_cost),
        overstock_cost=float(excess.sum() * unit_overstock_cost),
        service_level=float(np.mean(short <= 1e-9)),
        fill_rate=float(served.sum() / demand_total) if demand_total > 0 else 1.0,
    )
