#!/usr/bin/env python3
"""
Polymarket Mid-Window Momentum Scalper
========================================
A genuinely different strategy from the other bots in this project. Instead
of deciding once at window open, this watches an ALREADY-RUNNING 5-minute
window continuously, looking for a specific setup:

    Polymarket's own price for one side has been moving cleanly in one
    direction over the last SIGNAL_LOOKBACK_SEC seconds. If that move is
    large and clean enough to plausibly continue by MOVE_TARGET (10c) in
    the next 10-15 seconds, buy that side at its current price, and
    immediately rest a sell order PROFIT_MARGIN (5c) above entry — a
    conservative partial capture of the predicted move, not the whole thing.

Takes at most MAX_TRADES_PER_WINDOW entries per 5-minute window. Starts with
BTC only, per explicit request.

IMPORTANT — read before running live:
  This asks for something genuinely harder to predict than the other bots in
  this project: not just direction, but direction AND magnitude AND timing,
  continuously re-evaluated throughout the window. This has NOT been
  validated with real data. "Enters most windows" is a real, explicit design
  goal here, but the signal must still clear its bar — this bot will NOT
  force an entry with no real momentum behind it just to hit a trade count.
  Run --dry-run for a meaningful sample before ever using --live.

Modes:
  --dry-run   No real orders. Polls the REAL, LIVE order book and computes
              what WOULD have happened using real market data.
  --live      Places real orders using your Polymarket deposit wallet.

Usage:
  python momentum_bot.py --dry-run
  python momentum_bot.py --live --amount 2
"""

import time
import json
import csv
import argparse
import threading
import os
import collections
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

MARKETS = {
    "btc-updown-5m": "BTC",   # BTC only, per explicit request — extend later once validated
}

SIGNAL_LOOKBACK_SEC   = 15    # how far back to look for recent momentum
MIN_MOVE_TO_TRUST     = 0.03  # minimum real move (in price, not %) over the lookback before trusting direction —
                                # a starting hypothesis, not a calibrated number. Prevents acting on pure noise.
CLEANLINESS_MIN_RATIO = 0.5   # same concept as the predict-variant bot: net move vs total high-low range over
                                # the lookback. A low ratio means whipsaw, not a real trend — same logic that
                                # helped catch the delta bot's biggest-signal loss.

MOVE_TARGET   = 0.10  # the move we're trying to catch signs of (informational — not directly enforced,
                        # since we can't verify the FULL 10c move happens, only that momentum looks real)
PROFIT_MARGIN = 0.05   # conservative partial capture — sell trigger is entry price + this, not the full predicted move

BUY_CEILING_BUFFER = 0.02  # willing to pay up to (observed price + this) to actually get filled, since price is moving
BUY_TIMEOUT_SEC     = 3.0

MAX_TRADES_PER_WINDOW = 2

MONITOR_INTERVAL = 2.0   # how often to check for a new entry opportunity throughout the window
# Force-exit removed entirely per explicit request — unsold positions now ride
# to actual market resolution instead of being sold early at a fixed cutoff.
# This means an unsold trade can win fully or lose fully, not a bounded loss.

POLL_INTERVAL_SLOW = 1.0

# ─── UTILITIES ───────────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def ts_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg, crypto=""):
    prefix = f"[{crypto}] " if crypto else ""
    with _print_lock:
        print(f"[{ts_str()}] {prefix}{msg}", flush=True)

def now_unix():
    return time.time()


def get_window_market(slug_prefix: str, start_ts: int) -> dict | None:
    """Find the Up/Down market for a window starting at start_ts."""
    slug = f"{slug_prefix}-{start_ts}"
    try:
        r = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=3)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        event = data[0]
    except Exception:
        return None

    markets = event.get("markets", [])
    if not markets:
        return None
    market = markets[0]

    try:
        outcomes       = json.loads(market.get("outcomes", "[]"))
        clob_token_ids = json.loads(market.get("clobTokenIds", "[]"))
    except Exception:
        return None

    if len(outcomes) < 2 or len(clob_token_ids) < 2:
        return None

    tokens = dict(zip(outcomes, clob_token_ids))
    if "Down" not in tokens or "Up" not in tokens:
        return None

    return {
        "slug":         slug,
        "crypto":       MARKETS[slug_prefix],
        "start_ts":     start_ts,
        "close_ts":     start_ts + 300,
        "down_token":   tokens["Down"],
        "up_token":     tokens["Up"],
        "condition_id": market.get("conditionId", ""),
        "title":        event.get("title", ""),
    }


def get_order_book(token_id: str) -> dict:
    """Raw public order book fetch — no auth required."""
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=2)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def best_ask(book: dict):
    asks = book.get("asks", [])
    if not asks:
        return None, None
    cheapest = min(asks, key=lambda a: float(a["price"]))
    return float(cheapest["price"]), float(cheapest["size"])


def best_bid(book: dict):
    bids = book.get("bids", [])
    if not bids:
        return None, None
    highest = max(bids, key=lambda b: float(b["price"]))
    return float(highest["price"]), float(highest["size"])


def mid_price(book: dict) -> float | None:
    """Midpoint of best bid/ask — used as this bot's own price-history signal."""
    bid, _ = best_bid(book)
    ask, _ = best_ask(book)
    if bid is None or ask is None:
        return None
    return round((bid + ask) / 2, 4)


def next_window_start(now: float) -> int:
    return int((now // 300) + 1) * 300


class PriceHistory:
    """
    Maintains this bot's OWN rolling buffer of a token's recent mid-price,
    built from direct polling — not dependent on any external kline interval
    we can't verify. Used to compute momentum/magnitude/cleanliness over the
    last SIGNAL_LOOKBACK_SEC seconds, directly on Polymarket's own price.
    """
    def __init__(self, lookback_sec: float):
        self.lookback_sec = lookback_sec
        self.buffer = collections.deque()  # (timestamp, price)

    def add(self, price: float):
        now = now_unix()
        self.buffer.append((now, price))
        cutoff = now - self.lookback_sec
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def signal(self) -> dict:
        """
        Returns a dict describing whether current momentum is strong and
        clean enough to trust, using the same magnitude+cleanliness concept
        already validated on the predict-variant spread bot — applied here
        to Polymarket's own short-term price history instead of a 5-minute
        window's prior-close data.
        """
        result = {"side": None, "move": 0.0, "cleanliness": 0.0, "reason": ""}
        if len(self.buffer) < 3:
            result["reason"] = "insufficient price history yet"
            return result

        prices = [p for _, p in self.buffer]
        oldest, current = prices[0], prices[-1]
        move = current - oldest
        price_range = max(prices) - min(prices)
        cleanliness = round(abs(move) / price_range, 4) if price_range > 0 else 1.0

        result["move"] = round(move, 4)
        result["cleanliness"] = cleanliness

        if abs(move) < MIN_MOVE_TO_TRUST:
            result["reason"] = f"move {move:+.4f} < {MIN_MOVE_TO_TRUST} — too weak to trust"
            return result
        if cleanliness < CLEANLINESS_MIN_RATIO:
            result["reason"] = f"move {move:+.4f} OK, but cleanliness {cleanliness:.2f} < {CLEANLINESS_MIN_RATIO} — too much whipsaw"
            return result

        result["side"] = "Up" if move > 0 else "Down"
        result["reason"] = f"move {move:+.4f}, cleanliness {cleanliness:.2f} — both pass"
        return result


# ─── PERSISTENT CSV LOG ──────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "bot_name", "mode", "crypto", "slug", "trade_num_this_window",
    "signal_side", "signal_move", "signal_cleanliness", "signal_reason",
    "buy_result", "buy_price", "buy_shares", "buy_elapsed_ms",
    "sell_result", "sell_price", "pnl_usd", "notes",
]

class TradeLogger:
    def __init__(self, bot_name: str):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades_log.csv")
        self.lock = threading.Lock()
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def write(self, row: dict):
        row = {**{k: "" for k in CSV_FIELDS}, **row}
        with self.lock:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in CSV_FIELDS])


# ─── CORE BOT ────────────────────────────────────────────────────────────────

class MomentumBot:
    def __init__(self, dry_run: bool, amount: float):
        self.dry_run  = dry_run
        self.amount   = amount
        self.bot_name = os.getenv("BOT_NAME", "momentum_bot")
        self.mode_str = "dry_run" if dry_run else "live"
        self.stop_event = threading.Event()
        self.trades = []
        self.trades_lock = threading.Lock()
        self.logger = TradeLogger(self.bot_name)

        self.client = None
        if not dry_run:
            self._init_client()

        log("=" * 70)
        log(f"Momentum Scalper | {self.mode_str.upper()} | ${amount:.2f}/trade | bot_name={self.bot_name}")
        log(f"Signal: {SIGNAL_LOOKBACK_SEC}s lookback, min move {MIN_MOVE_TO_TRUST}, cleanliness >= {CLEANLINESS_MIN_RATIO}")
        log(f"Sell trigger: entry + ${PROFIT_MARGIN} | unsold positions hold to actual resolution (no force-exit) | max {MAX_TRADES_PER_WINDOW} trades/window")
        log(f"Trade log: {self.logger.path}")
        log("=" * 70)

    def _init_client(self):
        from py_clob_client_v2 import ClobClient, AssetType, BalanceAllowanceParams
        signature_type = int(os.getenv("POLY_SIGNATURE_TYPE", "3"))
        self.client = ClobClient(
            host=CLOB_API,
            key=os.environ["POLY_PRIVATE_KEY"],
            chain_id=137,
            signature_type=signature_type,
            funder=os.environ["POLY_PROXY_WALLET"],
        )
        self.client.set_api_creds(self.client.create_or_derive_api_key())
        self.client.update_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=signature_type,
        ))

    # ── BUY ──────────────────────────────────────────────────────────────────

    def _attempt_buy(self, token: str, observed_price: float, crypto: str) -> dict:
        ceiling = round(observed_price + BUY_CEILING_BUFFER, 4)
        MIN_SHARES = 5  # CONFIRMED via a real live API error on the other bots in this project: "Size (4) lower than the minimum: 5"

        if self.dry_run:
            book = get_order_book(token)
            price, size = best_ask(book)
            if price is not None and price <= ceiling:
                shares = max(MIN_SHARES, round(self.amount / price))
                log(f"[DRY] BUY would fill: ask ${price:.3f} (size {size})", crypto)
                return {"result": "bought", "price": price, "shares": shares}
            log(f"[DRY] BUY missed: no ask <= ${ceiling}", crypto)
            return {"result": "missed", "price": None, "shares": 0}

        from py_clob_client_v2 import OrderArgsV2, Side, OrderType, OrderPayload
        size = max(MIN_SHARES, round(self.amount / ceiling))
        try:
            resp = self.client.create_and_post_order(
                OrderArgsV2(token_id=token, price=ceiling, size=size, side=Side.BUY),
                order_type=OrderType.GTC,
            )
        except Exception as e:
            log(f"❌ BUY order failed to submit: {e}", crypto)
            return {"result": "error", "price": None, "shares": 0}

        order_id = resp.get("orderID", "")
        deadline = now_unix() + BUY_TIMEOUT_SEC
        last_known_size = 0.0
        while now_unix() < deadline:
            try:
                detail = self.client.get_order(order_id)
            except Exception:
                detail = None
            if detail is None:
                break
            try:
                current_size = float(detail.get("size_matched", 0))
                if current_size > last_known_size:
                    last_known_size = current_size
            except (TypeError, ValueError):
                pass
            time.sleep(0.25)

        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
        except Exception:
            pass

        if last_known_size <= 0:
            # Final independent safety check — same fix applied to the other
            # bots after a real, confirmed lag was observed between get_order()
            # and actual fill state. Field name for balance is NOT confirmed
            # from docs — logging raw response if this doesn't parse.
            try:
                from py_clob_client_v2 import AssetType, BalanceAllowanceParams
                bal_resp = self.client.get_balance_allowance(BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token,
                    signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "3")),
                ))
                real_balance = float(bal_resp.get("balance", 0)) / 1_000_000
                if real_balance >= 0.5:
                    log(f"⚠️ get_order() showed no fill, but balance check found {real_balance} shares — correcting course", crypto)
                    return {"result": "bought", "price": ceiling, "shares": real_balance}
            except Exception as e:
                log(f"⚠️ Final balance safety-check failed ({e})", crypto)
            log(f"❌ BUY timed out with no confirmed fill after {BUY_TIMEOUT_SEC}s", crypto)
            return {"result": "missed", "price": None, "shares": 0}

        log(f"✅ BUY confirmed: {last_known_size} shares at ceiling ${ceiling}, order {order_id[:16]}...", crypto)
        return {"result": "bought", "price": ceiling, "shares": last_known_size}

    # ── SELL ─────────────────────────────────────────────────────────────────

    def _watch_for_sell(self, token: str, buy_price: float, raw_shares: float, close_ts: float, crypto: str) -> dict:
        shares = int(raw_shares)
        if shares != raw_shares:
            log(f"⚠️ Buy partially filled: held {raw_shares}, flooring to {shares} whole shares to keep sells valid", crypto)
        if shares < 1:
            # NOTE: this specific case is NOT a premature time-based exit — a
            # sub-1-share dust fill can never be sold as a normal limit order
            # regardless of how much time is given, so exiting it immediately
            # is the only option, not the thing that was removed below.
            log("⚠️ Partial fill left less than 1 whole share — forcing immediate exit", crypto)
            exit_result = self._force_exit(token, raw_shares, crypto)
            pnl = -round(buy_price * raw_shares, 4)
            return {**exit_result, "pnl_usd": pnl, "notes": "sub-1-share partial fill"}

        sell_trigger = round(buy_price + PROFIT_MARGIN, 4)
        log(f"Sell trigger: ${sell_trigger} (bought ${buy_price} + ${PROFIT_MARGIN})", crypto)

        if not self.dry_run:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            try:
                self.client.update_balance_allowance(BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token,
                    signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "3")),
                ))
            except Exception as e:
                log(f"⚠️ Could not sync conditional balance ({e})", crypto)

            from py_clob_client_v2 import OrderArgsV2, Side, OrderType, OrderPayload
            try:
                resp = self.client.create_and_post_order(
                    OrderArgsV2(token_id=token, price=sell_trigger, size=shares, side=Side.SELL),
                    order_type=OrderType.GTC,
                )
                sell_order_id = resp.get("orderID", "")
                log(f"Resting SELL placed at ${sell_trigger}, order {sell_order_id[:16]}...", crypto)
            except Exception as e:
                # Order placement itself failing is a real error to recover
                # from, not the time-based premature exit that was removed —
                # keeping a fallback here is a different thing than force-exit.
                log(f"⚠️ Could not place resting sell ({e}) — exiting immediately since we can't rest a sell at all", crypto)
                exit_result = self._force_exit(token, shares, crypto)
                pnl = round((exit_result["price"] - buy_price) * shares, 4) if exit_result["price"] is not None else -round(buy_price * shares, 4)
                return {**exit_result, "pnl_usd": pnl, "notes": "resting sell placement failed"}

            # No more time-based force-exit. Keep watching until either the
            # resting order fully fills, or the window actually closes — at
            # which point the position rides to real market resolution
            # instead of being sold early. This means the true outcome of an
            # unsold position is NOT known immediately anymore; it depends
            # on the actual settlement, which this bot does not yet check.
            last_known_sold = 0.0
            while now_unix() < close_ts:
                try:
                    detail = self.client.get_order(sell_order_id)
                except Exception:
                    detail = None
                if detail is None:
                    last_known_sold = shares
                    break
                try:
                    current_sold = float(detail.get("size_matched", 0))
                    if current_sold > last_known_sold:
                        last_known_sold = current_sold
                except (TypeError, ValueError):
                    pass
                time.sleep(POLL_INTERVAL_SLOW)

            if last_known_sold >= shares:
                pnl = round((sell_trigger - buy_price) * shares, 4)
                return {"result": "sold", "price": sell_trigger, "pnl_usd": pnl, "notes": "sold via resting order"}

            remaining = round(shares - last_known_sold, 4)
            # Cancel the now-stale resting sell BEFORE holding to resolution —
            # otherwise it could still get matched at the old, low sell_trigger
            # price right as the market resolves, accidentally giving away the
            # exact upside this decision was meant to hold for. We keep the
            # underlying shares; we just remove the order offering to sell them cheap.
            try:
                self.client.cancel_order(OrderPayload(orderID=sell_order_id))
            except Exception:
                pass  # may already be gone if it just fully filled in the final instant — fine
            log(f"⏰ Window closed with {remaining} shares unsold — cancelled the resting sell, holding to actual "
                f"market resolution, true outcome not yet known", crypto)
            sold_pnl = round((sell_trigger - buy_price) * last_known_sold, 4) if last_known_sold > 0 else 0.0
            return {"result": "holding_to_resolution", "price": None, "pnl_usd": sold_pnl,
                    "notes": f"{remaining} shares still unsold at window close — pnl on those is pending actual resolution, not reflected above"}

        # DRY-RUN: poll-based simulation. Same change — keep watching until
        # sold or the window closes, no simulated forced exit.
        while now_unix() < close_ts:
            book = get_order_book(token)
            price, size = best_bid(book)
            if price is not None and price >= sell_trigger and size >= shares:
                log(f"[DRY] SELL would fill: bid ${price:.3f}", crypto)
                pnl = round((price - buy_price) * shares, 4)
                return {"result": "sold", "price": price, "pnl_usd": pnl, "notes": "sold"}
            time.sleep(POLL_INTERVAL_SLOW)

        log(f"⏰ Window closed, still holding — this would ride to actual resolution (not simulated here)", crypto)
        return {"result": "holding_to_resolution", "price": None, "pnl_usd": 0.0,
                "notes": "never reached sell trigger before window close — true outcome not simulated"}

    def _force_exit(self, token: str, shares: float, crypto: str) -> dict:
        if self.dry_run:
            book = get_order_book(token)
            price, size = best_bid(book)
            if price is None:
                log("[DRY] No bids at all for force-exit — total loss this trade", crypto)
                return {"result": "no_bids", "price": None}
            return {"result": "exited", "price": price}

        from py_clob_client_v2 import MarketOrderArgsV2, Side, OrderType
        try:
            resp = self.client.create_and_post_market_order(
                MarketOrderArgsV2(token_id=token, amount=shares, side=Side.SELL),
                order_type=OrderType.FAK,
            )
        except Exception as e:
            log(f"⚠️ Force-exit order failed: {e}", crypto)
            return {"result": "error", "price": None}
        status = str(resp.get("status", "")).lower()
        if status == "matched":
            try:
                cost = float(resp.get("makingAmount", 0)) / 1_000_000
                exit_price = round(cost / shares, 4) if shares else None
            except Exception:
                exit_price = None
            return {"result": "exited", "price": exit_price}
        return {"result": "unmatched", "price": None}

    # ── WINDOW LOOP ──────────────────────────────────────────────────────────

    def _monitor_window(self, slug_prefix: str, start_ts: int):
        crypto = MARKETS[slug_prefix]
        close_ts = start_ts + 300

        market = None
        find_deadline = now_unix() + 5
        while now_unix() < find_deadline:
            market = get_window_market(slug_prefix, start_ts)
            if market:
                break
            time.sleep(0.5)
        if not market:
            log(f"Could not find market for window starting {start_ts} — skipping entire window", crypto)
            return

        up_history   = PriceHistory(SIGNAL_LOOKBACK_SEC)
        down_history = PriceHistory(SIGNAL_LOOKBACK_SEC)
        trades_this_window = 0

        while now_unix() < close_ts and trades_this_window < MAX_TRADES_PER_WINDOW:
            if self.stop_event.is_set():
                return

            up_book, down_book = get_order_book(market["up_token"]), get_order_book(market["down_token"])
            up_price, down_price = mid_price(up_book), mid_price(down_book)
            if up_price is not None:
                up_history.add(up_price)
            if down_price is not None:
                down_history.add(down_price)

            up_signal   = up_history.signal()
            down_signal = down_history.signal()

            # Only one side can genuinely be trending at a time in a two-outcome
            # market (they're complementary) — but check both independently
            # since our own price histories could briefly disagree due to timing.
            chosen = None
            if up_signal["side"] == "Up":
                chosen = ("Up", market["up_token"], up_price, up_signal)
            elif down_signal["side"] == "Down":
                chosen = ("Down", market["down_token"], down_price, down_signal)

            if chosen:
                side, token, price, signal = chosen
                trades_this_window += 1
                log(f"Signal fired (trade {trades_this_window}/{MAX_TRADES_PER_WINDOW}): {signal['reason']} -> buying {side} @ ~${price}", crypto)

                buy_info = self._attempt_buy(token, price, crypto)
                row = {
                    "timestamp": ts_str(), "bot_name": self.bot_name, "mode": self.mode_str,
                    "crypto": crypto, "slug": market["slug"], "trade_num_this_window": trades_this_window,
                    "signal_side": side, "signal_move": signal["move"], "signal_cleanliness": signal["cleanliness"],
                    "signal_reason": signal["reason"], "buy_result": buy_info["result"],
                    "buy_price": buy_info["price"], "buy_shares": buy_info["shares"],
                }

                if buy_info["result"] != "bought":
                    row.update({"sell_result": "n/a", "sell_price": "", "pnl_usd": 0, "notes": "no buy fill"})
                    self._record(row)
                    continue

                sell_info = self._watch_for_sell(token, buy_info["price"], buy_info["shares"], close_ts, crypto)
                row.update({
                    "sell_result": sell_info["result"], "sell_price": sell_info["price"],
                    "pnl_usd": sell_info["pnl_usd"], "notes": sell_info["notes"],
                })
                self._record(row)

            time.sleep(MONITOR_INTERVAL)

    def _record(self, row: dict):
        with self.trades_lock:
            self.trades.append(row)
        self.logger.write(row)
        pnl = row.get("pnl_usd", 0)
        sign = "+" if isinstance(pnl, (int, float)) and pnl >= 0 else ""
        log(f"RECORDED: side={row['signal_side']} | buy={row['buy_result']}@{row['buy_price']} | "
            f"sell={row['sell_result']}@{row['sell_price']} | pnl={sign}${pnl}", row["crypto"])

    def _asset_loop(self, slug_prefix: str):
        crypto = MARKETS[slug_prefix]
        while not self.stop_event.is_set():
            start_ts = next_window_start(now_unix())
            while now_unix() < start_ts and not self.stop_event.is_set():
                time.sleep(1)
            if self.stop_event.is_set():
                break
            log(f"Monitoring window starting {datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%H:%M:%S')} UTC", crypto)
            try:
                self._monitor_window(slug_prefix, start_ts)
            except Exception as e:
                log(f"⚠️ Unhandled error this window: {e}", crypto)
            time.sleep(2)

    def run(self):
        threads = [threading.Thread(target=self._asset_loop, args=(prefix,), daemon=True) for prefix in MARKETS]
        for t in threads:
            t.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log("Stopping...")
            self.stop_event.set()
            self._print_summary()

    def _print_summary(self):
        log("-" * 70)
        with self.trades_lock:
            trades = list(self.trades)
        bought  = [t for t in trades if t["buy_result"] == "bought"]
        sold    = [t for t in bought if t["sell_result"] == "sold"]
        pending = [t for t in bought if t["sell_result"] == "holding_to_resolution"]
        dust    = [t for t in bought if t["sell_result"] not in ("sold", "holding_to_resolution")]
        total_pnl = sum(float(t["pnl_usd"] or 0) for t in trades)
        log(f"SUMMARY — {len(trades)} signals fired, {len(bought)} buy fills")
        log(f"  Sold at margin: {len(sold)}")
        log(f"  Holding to actual resolution (outcome not yet known): {len(pending)}")
        if dust:
            log(f"  Other (dust/error handling): {len(dust)}")
        log(f"  Total PnL shown here EXCLUDES pending positions — check your actual account for their real outcome")
        log(f"  Realized PnL so far: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f}")
        log("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Mid-Window Momentum Scalper")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--amount", type=float, default=2.0)
    args = parser.parse_args()

    bot = MomentumBot(dry_run=args.dry_run, amount=args.amount)
    bot.run()
