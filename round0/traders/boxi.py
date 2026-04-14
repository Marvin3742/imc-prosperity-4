import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, Tuple


class Logger:

    def __init__(self) -> None:
        self.logs: str = ""
        self.max_log_length: int = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
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

    def _compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {sym: [od.buy_orders, od.sell_orders] for sym, od in order_depths.items()}

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


class Trader:

    def __init__(self) -> None:
        self.logger = Logger()
        self.pos_limits: Dict[str, int] = {
            "EMERALDS": 50,
            "TOMATOES": 60,
        }
        self.ema_mid: Dict[str, float] = {}
        self._ema_updates: Dict[str, int] = {}

        self.ema_alpha_base = 0.12
        self.ema_alpha_fast = 0.30
        self.warmup_ticks = 45
        self.warmup_passive_extra = 0.4 

        self.take_spread_frac = 0.38
        self.take_edge_min = 0.55
        self.take_edge_max = 2.8

        self.passive_half_spread_frac = 0.85
        self.passive_offset_min = 0.45
        self.passive_offset_max = 1.85

        self.inv_skew_coeff = 0.42

        self.micro_tilt_scale = 0.35

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _best_bid_ask(order_depth: OrderDepth) -> Tuple:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None, 0, 0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders.get(best_bid, 0)
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        return best_bid, best_ask, bid_vol, ask_vol

    def _fair_value(self, product: str, order_depth: OrderDepth) -> float:
        best_bid, best_ask, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        
        # ZAWSZE aktualizuj licznik ticków dla każdego produktu na samym początku!
        n = self._ema_updates.get(product, 0) + 1
        self._ema_updates[product] = n
        
        if best_bid is None or best_ask is None:
            return 10000.0 if product == "EMERALDS" else self.ema_mid.get(product, 0.0)

        mid = (best_bid + best_ask) / 2.0

        denom = bid_vol + ask_vol
        if denom > 0:
            micro = (best_bid * ask_vol + best_ask * bid_vol) / denom
            imbalance_tilt = self.micro_tilt_scale * (micro - mid)
        else:
            imbalance_tilt = 0.0

        # Szmaragdy dostają stałą bazę + tilt
        if product == "EMERALDS":
            return 10000.0 + imbalance_tilt

        # Pomidory i reszta liczą EMA
        alpha = self.ema_alpha_fast if n <= self.warmup_ticks else self.ema_alpha_base

        if product not in self.ema_mid:
            self.ema_mid[product] = mid
        else:
            self.ema_mid[product] = (1.0 - alpha) * self.ema_mid[product] + alpha * mid

        return self.ema_mid[product] + imbalance_tilt

    def _edges_from_spread(self, spread: float) -> Tuple[float, float]:
        """Returns (take_edge, passive_offset_from_fair per side)."""
        take_edge = self._clamp(
            self.take_spread_frac * spread,
            self.take_edge_min,
            self.take_edge_max,
        )
        half = max(spread / 2.0, 0.5)
        passive_off = self._clamp(
            self.passive_half_spread_frac * half,
            self.passive_offset_min,
            self.passive_offset_max,
        )
        return take_edge, passive_off

    def _build_orders_for_product(
        self,
        product: str,
        state: TradingState,
        order_depth: OrderDepth,
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.pos_limits.get(product, 20)
        pos = state.position.get(product, 0)

        fair = self._fair_value(product, order_depth)
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)

        if best_bid is None or best_ask is None:
            return orders

        spread = float(best_ask - best_bid)
        if spread <= 0:
            return orders

        updates = self._ema_updates.get(product, 0)
        in_warmup = updates <= self.warmup_ticks
        take_edge, passive_off = self._edges_from_spread(spread)
        if in_warmup:
            passive_off += self.warmup_passive_extra

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        norm = self._clamp(pos / float(limit), -1.0, 1.0) if limit else 0.0
        inv_skew = self.inv_skew_coeff * norm * max(spread, 1.0)

        buy_threshold = fair - take_edge
        sell_threshold = fair + take_edge

        if not in_warmup:
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if ask_px <= buy_threshold and buy_cap > 0:
                    ask_qty = abs(order_depth.sell_orders[ask_px])
                    take = min(buy_cap, ask_qty)
                    if take > 0:
                        orders.append(Order(product, int(ask_px), int(take)))
                        buy_cap -= take

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_px >= sell_threshold and sell_cap > 0:
                    bid_qty = order_depth.buy_orders[bid_px]
                    hit = min(sell_cap, bid_qty)
                    if hit > 0:
                        orders.append(Order(product, int(bid_px), int(-hit)))
                        sell_cap -= hit

        bid_raw = fair - passive_off - inv_skew
        ask_raw = fair + passive_off - inv_skew

        bid_quote = int(round(bid_raw))
        ask_quote = int(round(ask_raw))

        bid_quote = min(bid_quote, int(best_bid) + 1)
        ask_quote = max(ask_quote, int(best_ask) - 1)

        bid_quote = min(bid_quote, int(best_ask) - 1)
        ask_quote = max(ask_quote, int(best_bid) + 1)

        base_lot = max(4, limit // 6)
        if in_warmup:
            lot = max(1, min(base_lot, limit // 10))
        else:
            lot = base_lot

        if buy_cap > 0 and bid_quote < best_ask:
            post_buy = min(buy_cap, lot)
            if post_buy > 0:
                orders.append(Order(product, bid_quote, int(post_buy)))

        if sell_cap > 0 and ask_quote > best_bid:
            post_sell = min(sell_cap, lot)
            if post_sell > 0:
                orders.append(Order(product, ask_quote, int(-post_sell)))

        return orders

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product in self.pos_limits:
                result[product] = self._build_orders_for_product(product, state, order_depth)
            else:
                result[product] = []

        conversions = 0
        trader_data = "baseline-mm-v2-stabilized"
        self.logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data