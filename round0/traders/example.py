from __future__ import annotations

import json
import math
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


class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        acceptable_price_dict = {"EMERALDS": 10000, "TOMATOES": 5000}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []
            acceptable_price = acceptable_price_dict[product]  # Participant should calculate this value
            logger.print("Acceptable price : " + str(acceptable_price))
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    logger.print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
        trader_data = ""  # No state needed - we check position directly
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    





