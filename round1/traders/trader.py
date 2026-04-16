from __future__ import annotations

import json
from typing import Any

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logger (compatible with Prosperity visualiser)
# ──────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs: str = ""
        self.max_log_length: int = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self._to_json(
                [
                    self._compress_state(state, ""),
                    self._compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self._to_json(
                [
                    self._compress_state(state, self._truncate(state.traderData, max_item_length)),
                    self._compress_orders(orders),
                    conversions,
                    self._truncate(trader_data, max_item_length),
                    self._truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    # ── compression helpers ───────────────────────────────────────────────────

    def _compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self._compress_listings(state.listings),
            self._compress_order_depths(state.order_depths),
            self._compress_trades(state.own_trades),
            self._compress_trades(state.market_trades),
            state.position,
            self._compress_observations(state.observations),
        ]

    def _compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def _compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {
            sym: [od.buy_orders, od.sell_orders]
            for sym, od in order_depths.items()
        }

    def _compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def _compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs: dict[str, list] = {}
        for product, obs in observations.conversionObservations.items():
            conv_obs[product] = [
                obs.bidPrice,
                obs.askPrice,
                obs.transportFees,
                obs.exportTariff,
                obs.importTariff,
                obs.sugarPrice,
                obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, conv_obs]

    def _compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for o in arr:
                compressed.append([o.symbol, o.price, o.quantity])
        return compressed

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def _truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
POSITION_LIMIT = 80      
DRIFT_PER_STEP = 0.001     


# ──────────────────────────────────────────────────────────────────────────────
# Trader
# ──────────────────────────────────────────────────────────────────────────────

class Trader:
    def __init__(self) -> None:
        self.pepper_open: float | None = None

    def get_position(self, product):
        positions = {"INTARIAN_PEPPER_ROOT": 4,
                     "ASH_COATED_OSMIUM": 6}
    
        return positions[product]
    
    def best_bid_ask(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None, 0, 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_amount = order_depth.buy_orders.get(best_bid, 0)
        ask_amount = abs(order_depth.sell_orders.get(best_ask, 0))
        return best_bid, best_ask, bid_amount, ask_amount
    
    def basic_market_making(self, product: str, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        order_depth = state.order_depths.get(product)
        best_bid, best_ask, _, _ = self.best_bid_ask(order_depth)
        if not best_bid or not best_ask:
            return orders
        position = self.get_position(product)
        orders.append(Order(product, best_bid+1, position))
        orders.append(Order(product, best_ask-1, -position))
        return orders
    
    def buy_n_hold(self, product: str, state: TradingState) -> list[Order]:
        order_depth = state.order_depths[product]
        orders: list[Order] = []

        if not order_depth.buy_orders and not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        # Record the open price on the first tick where both sides are quoted
        if self.pepper_open is None and best_bid is not None and best_ask is not None:
            self.pepper_open = (best_bid + best_ask) / 2
            logger.print(f"[{product}] Open price: {self.pepper_open}")

        if self.pepper_open is None:
            return orders

        # Fair value drifts linearly upward from the open
        fv = self.pepper_open + DRIFT_PER_STEP * state.timestamp
        current_pos = state.position.get(product, 0)
        remaining_capacity = POSITION_LIMIT - current_pos

        logger.print(f"[{product}] t={state.timestamp} fv={fv:.1f} pos={current_pos}")

        # Hit every ask at or below fair value until the position limit is reached
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if remaining_capacity <= 0:
                break
            if True:
                vol = min(-order_depth.sell_orders[ask_price], remaining_capacity)
                logger.print(f"[{product}] BUY {vol}x @ {ask_price}")
                orders.append(Order(product, ask_price, vol))
                remaining_capacity -= vol


        return orders
    
    def build_orders_for_product(self, product:str, state: TradingState) -> list[Order]:
        match product:
            case "INTARIAN_PEPPER_ROOT":    
                return self.buy_n_hold(product, state)
            case "ASH_COATED_OSMIUM":
                return self.basic_market_making(product, state)
            case _:
                return []
    
    
    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}

        for product in state.order_depths:
            result[product] = self.build_orders_for_product(product, state)

        trader_data = ""
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
