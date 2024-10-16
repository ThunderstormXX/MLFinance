
from typing import Union, Optional, Callable
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from ipywidgets import interact
from ipywidgets import widgets
from tqdm.auto import tqdm

FloatArray = npt.NDArray[np.float_]
Floats = Union[float, FloatArray]

@dataclass
class MarketState:
    stock_price: Floats
    interest_rate: Floats
    time: Floats = 0

@dataclass
class StockOption:
    strike_price: Floats
    expiration_time: Floats  # in years
    is_call: Union[bool, npt.NDArray[np.bool_]]

    def payoff(self, stock_price: Floats) -> Floats:
        call_payoff = np.maximum(0, stock_price - self.strike_price)
        put_payoff = np.maximum(0, self.strike_price - stock_price)
        return np.where(self.is_call, call_payoff, put_payoff)

class CallStockOption(StockOption):
    def __init__(self, strike_price, expiration_time):
        super().__init__(strike_price, expiration_time, True)

class PutStockOption(StockOption):
    def __init__(self, strike_price, expiration_time):
        super().__init__(strike_price, expiration_time, False)

@dataclass
class BSParams:
    volatility: Floats

def dt(option: StockOption, ms: MarketState):
    return np.maximum(option.expiration_time - ms.time, np.finfo(np.float64).eps)


def d1(option: StockOption, ms: MarketState, params: BSParams):
    return 1 / (params.volatility * np.sqrt(dt(option, ms)))\
                * (np.log(ms.stock_price / option.strike_price)
                   + (ms.interest_rate + params.volatility ** 2 / 2) * dt(option, ms))


def d2(option: StockOption, ms: MarketState, params: BSParams):
    return d1(option, ms, params) - params.volatility * np.sqrt(dt(option, ms))


def price(option: StockOption, ms: MarketState, params: BSParams):
    discount_factor = np.exp(-ms.interest_rate * (dt(option, ms)))

    call_price = stats.norm.cdf(d1(option, ms, params)) * ms.stock_price\
            - stats.norm.cdf(d2(option, ms, params)) * option.strike_price * discount_factor
    put_price = stats.norm.cdf(-d2(option, ms, params)) * option.strike_price * discount_factor\
        - stats.norm.cdf(-d1(option, ms, params)) * ms.stock_price

    return np.where(option.is_call, call_price, put_price)

def delta(option: StockOption, ms: MarketState, params: BSParams):
    nd1 = stats.norm.cdf(d1(option, ms, params))
    return np.where(option.is_call, nd1, nd1 - 1)


def gamma(option: StockOption, ms: MarketState, params: BSParams):
    return stats.norm.pdf(d1(option, ms, params)) / (ms.stock_price * params.volatility * np.sqrt(dt(option, ms)))


def theta(option: StockOption, ms: MarketState, params: BSParams):
    a = -ms.stock_price * stats.norm.pdf(d1(option, ms, params)) * params.volatility\
        / (2 * np.sqrt(dt(option, ms)))
    d_discount_factor = ms.interest_rate * np.exp(-ms.interest_rate * (dt(option, ms)))

    call_theta = a - option.strike_price * d_discount_factor * stats.norm.cdf(d2(option, ms, params))
    put_theta = a + option.strike_price * d_discount_factor * stats.norm.cdf(-d2(option, ms, params))
    return np.where(option.is_call, call_theta, put_theta)


def vega(option: StockOption, ms: MarketState, params: BSParams):
    return ms.stock_price * stats.norm.pdf(d1(option, ms, params)) * np.sqrt(dt(option, ms))


def rho(option: StockOption, ms: MarketState, params: BSParams):
    discount_factor = np.exp(-ms.interest_rate * (dt(option, ms)))
    call_rho = option.strike_price * dt(option, ms) * discount_factor * stats.norm.cdf(d2(option, ms, params))
    put_rho = -option.strike_price * dt(option, ms) * discount_factor * stats.norm.cdf(-d2(option, ms, params))
    return np.where(option.is_call, call_rho, put_rho)