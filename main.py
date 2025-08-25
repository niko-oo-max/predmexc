import os
import json
import logging
import threading
from datetime import datetime
import asyncio
import time

import requests
import pandas as pd
import ta

from flask import Flask, jsonify
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ==============================
# CONFIG (env overrides supported)
# ==============================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BINANCE_BASE = "https://api.binance.com"
TIMEFRAME = os.getenv("TIMEFRAME", "5m")  # 1m, 5m, 15m, 1h, etc
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",") if s.strip()]
BROADCAST_INTERVAL = int(os.getenv("BROADCAST_INTERVAL", "3600"))  # seconds between auto alerts
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "80"))  # only broadcast signals >= this
SUBSCRIBERS_FILE = os.getenv("SUBSCRIBERS_FILE", "subscribers.json")

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("render-tv-bot")

# ==============================
# FLASK (keep-alive endpoint)
# ==============================
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "service": "TV Signals Bot",
        "status": "running",
        "time_utc": datetime.utcnow().isoformat() + "Z",
        "endpoints": ["/", "/ping"]
    })

@app.route("/ping")
def ping():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat() + "Z"})

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ==============================
# SUBSCRIBERS
# ==============================
def load_subscribers() -> set[int]:
    try:
        if os.path.exists(SUBSCRIBERS_FILE):
            with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(int(x) for x in data if isinstance(x, int) or str(x).isdigit())
    except Exception as e:
        logger.warning(f"Failed to load subscribers: {e}")
    return set()

def save_subscribers(subs: set[int]):
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(subs)), f)
    except Exception as e:
        logger.warning(f"Failed to save subscribers: {e}")

SUBSCRIBERS = load_subscribers()

# ==============================
# BINANCE DATA
# ==============================
def get_price(symbol: str) -> float | None:
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/price", params={"symbol": symbol}, timeout=7)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        logger.error(f"Price fetch error {symbol}: {e}")
        return None

def get_ohlcv(symbol: str, interval: str = TIMEFRAME, limit: int = 200) -> pd.DataFrame | None:
    """Return DataFrame with columns: time, open, high, low, close, volume (floats)"""
    try:
        url = f"{BINANCE_BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(
            data,
            columns=[
                "time","open","high","low","close","volume",
                "close_time","qav","trades","taker_base","taker_quote","ignore"
            ],
        )
        df = df[["time","open","high","low","close","volume"]].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        logger.error(f"OHLCV fetch error {symbol}: {e}")
        return None

# ==============================
# INDICATORS
# ==============================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    """Adds EMA9/20/50, RSI14/7, Stochastic, MACD, BB, Volume SMA, ATR."""
    try:
        # EMAs
        df["ema9"]  = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

        # RSI
        df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi7"]  = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

        # Stochastic (fast %K)
        stoch = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()  # 0-100

        # MACD
        macd = ta.trend.MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"]  = bb.bollinger_lband()
        df["bb_mid"]  = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"] * 100  # %

        # Volume filters
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma20"]

        # ATR for SL/TP
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range()

        # Support/Resistance proxies
        df["support20"]    = df["low"].rolling(20).min()
        df["resistance20"] = df["high"].rolling(20).max()

        return df
    except Exception as e:
        logger.error(f"Indicator calc error: {e}")
        return None

# ==============================
# SIGNAL ENGINE
# ==============================
def calc_sl_tp(price: float, atr: float, side: str, conf: float) -> dict:
    """ATR-based dynamic stops/targets scaled by confidence & volatility."""
    # wider stops/targets at higher confidence
    base = 1.3 + (conf/100)  # 1.3 .. 2.3
    stop_mult = base
    tp_mult   = base * 2.0

    if side == "LONG":
        sl  = price - stop_mult * atr
        tp1 = price + tp_mult * atr
        tp2 = price + tp_mult * 1.5 * atr
    else:
        sl  = price + stop_mult * atr
        tp1 = price - tp_mult * atr
        tp2 = price - tp_mult * 1.5 * atr

    # risk:reward based on TP1
    risk  = abs(price - sl)
    reward = abs(tp1 - price)
    rr = round((reward / risk), 2) if risk > 0 else 0.0
    return {
        "stop_loss": round(sl, 6),
        "take_profit_1": round(tp1, 6),
        "take_profit_2": round(tp2, 6),
        "risk_reward": rr
    }

def evaluate_signal(df: pd.DataFrame) -> dict | None:
    """Return best of LONG/SHORT candidates with confidence & targets, else None."""
    if df is None or len(df) < 60:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Conditions
    ema_trend_up   = last["ema20"] > last["ema50"]
    ema_trend_down = last["ema20"] < last["ema50"]

    ema9_cross_up   = prev["ema9"] <= prev["ema20"] and last["ema9"] > last["ema20"]
    ema9_cross_down = prev["ema9"] >= prev["ema20"] and last["ema9"] < last["ema20"]

    rsi_bull = (last["rsi14"] < 40) or (last["rsi7"] < 30)
    rsi_bear = (last["rsi14"] > 60) or (last["rsi7"] > 70)

    stoch_bull = last["stoch_k"] < 25
    stoch_bear = last["stoch_k"] > 75

    macd_bull = (last["macd"] > last["macd_signal"]) and (last["macd_hist"] > prev["macd_hist"])
    macd_bear = (last["macd"] < last["macd_signal"]) and (last["macd_hist"] < prev["macd_hist"])

    # Price near BB edges
    bb_low_prox  = ((last["close"] - last["bb_low"]) / max(last["bb_low"], 1e-9)) * 100  # %
    bb_high_prox = ((last["bb_high"] - last["close"]) / max(last["close"], 1e-9)) * 100
    near_lower = bb_low_prox < 2.0
    near_upper = bb_high_prox < 2.0

    # Filters
    vol_ok = (last["vol_ratio"] > 1.2)  # 20%+ above avg
    volat_ok = (last["bb_width"] > 2.0)  # avoid dead range

    # Build scores
    long_checks = [
        ema9_cross_up,               # entry trigger
        ema_trend_up or (last["ema20"] > prev["ema20"]),
        rsi_bull,
        stoch_bull,
        near_lower,
        macd_bull or (last["macd_hist"] > 0),
        vol_ok,
        volat_ok
    ]
    short_checks = [
        ema9_cross_down,             # entry trigger
        ema_trend_down or (last["ema20"] < prev["ema20"]),
        rsi_bear,
        stoch_bear,
        near_upper,
        macd_bear or (last["macd_hist"] < 0),
        vol_ok,
        volat_ok
    ]

    long_conf  = round(sum(bool(x) for x in long_checks)  / len(long_checks)  * 100, 1)
    short_conf = round(sum(bool(x) for x in short_checks) / len(short_checks) * 100, 1)

    side = None
    conf = 0.0
    if long_conf >= short_conf and long_conf >= 50:
        side = "LONG";  conf = long_conf
    elif short_conf > long_conf and short_conf >= 50:
        side = "SHORT"; conf = short_conf
    else:
        return None

    price = float(last["close"])
    atr   = float(last["atr"]) if not pd.isna(last["atr"]) else None
    if atr is None or atr <= 0:
        return None

    targets = calc_sl_tp(price, atr, side, conf)
    # sanity: require at least 1.5:1
    if targets["risk_reward"] < 1.5:
        return None

    return {
        "side": side,
        "price": price,
        "confidence": conf,
        "rsi14": float(last["rsi14"]),
        "rsi7": float(last["rsi7"]),
        "stoch_k": float(last["stoch_k"]),
        "macd": float(last["macd"]),
        "vol_ratio": float(last["vol_ratio"]),
        "bb_width": float(last["bb_width"]),
        **targets
    }

def build_message(symbol: str, signal: dict) -> str:
    emoji = "🔥" if signal["confidence"] >= 85 else ("⚡" if signal["confidence"] >= 80 else "📊")
    return (
        f"🚨 Enhanced Technical Signal\n\n"
        f"{emoji} Pair: {symbol}\n"
        f"🔄 Signal: {signal['side']}\n"
        f"💰 Price: ${signal['price']:.6f}\n"
        f"🎯 Confidence: {signal['confidence']}%\n\n"
        f"📊 Indicators:\n"
        f"• RSI(14): {signal['rsi14']:.1f} | RSI(7): {signal['rsi7']:.1f}\n"
        f"• Stoch %K: {signal['stoch_k']:.1f}\n"
        f"• MACD: {signal['macd']:.6f}\n"
        f"• Volume Ratio: {signal['vol_ratio']:.2f}x\n"
        f"• BB Width: {signal['bb_width']:.2f}%\n\n"
        f"🛡 Risk:\n"
        f"• Stop Loss: ${signal['stop_loss']:.6f}\n"
        f"• Take Profit 1: ${signal['take_profit_1']:.6f}\n"
        f"• Take Profit 2: ${signal['take_profit_2']:.6f}\n"
        f"• Risk/Reward: {signal['risk_reward']}:1\n\n"
        f"⏰ TF: {TIMEFRAME} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"⚠️ Confidence is a rules score (not guaranteed win rate). Manage risk."
    )

def analyze_symbol(symbol: str) -> dict | None:
    df = get_ohlcv(symbol, interval=TIMEFRAME, limit=200)
    if df is None or len(df) < 60:
        return None
    df = calculate_indicators(df)
    if df is None:
        return None
    return evaluate_signal(df)

# ==============================
# TELEGRAM COMMANDS
# ==============================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Welcome!\n"
        "Use /subscribe to get auto alerts (only high-confidence).\n"
        "Try /signal (default pair) or /ta (checks multiple pairs)."
    )

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Optional symbol argument, else first configured symbol
    symbol = (context.args[0].upper() if context.args else SYMBOLS[0]).replace("/", "")
    sig = analyze_symbol(symbol)
    if sig:
        await update.message.reply_text(build_message(symbol, sig))
    else:
        await update.message.reply_text(f"📊 No qualifying setup for {symbol} right now.")

async def cmd_ta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    found_any = False
    for sym in SYMBOLS:
        sig = analyze_symbol(sym)
        if sig:
            found_any = True
            await update.message.reply_text(build_message(sym, sig))
    if not found_any:
        await update.message.reply_text("📊 TA complete: no high-quality setups right now.")

async def cmd_sub(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    SUBSCRIBERS.add(chat_id)
    save_subscribers(SUBSCRIBERS)
    await update.message.reply_text("✅ Subscribed. You’ll receive auto alerts (min confidence filter applied).")

async def cmd_unsub(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    SUBSCRIBERS.discard(chat_id)
    save_subscribers(SUBSCRIBERS)
    await update.message.reply_text("❌ Unsubscribed.")

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"🆔 Your chat ID: {update.message.chat_id}")

# ==============================
# AUTO BROADCAST (only high-confidence)
# ==============================
async def job_broadcast(context: ContextTypes.DEFAULT_TYPE):
    for sym in SYMBOLS:
        try:
            sig = analyze_symbol(sym)
            if not sig or sig["confidence"] < MIN_CONFIDENCE:
                continue
            msg = build_message(sym, sig)
            for chat_id in list(SUBSCRIBERS):
                try:
                    await context.bot.send_message(chat_id=chat_id, text=msg)
                except Exception as e:
                    logger.warning(f"Send fail {chat_id}: {e}")
        except Exception as e:
            logger.error(f"Broadcast error {sym}: {e}")

# ==============================
# MAIN
# ==============================
def run_bot():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN missing. Set it in Render env vars.")
        raise SystemExit(1)

    app_tg = Application.builder().token(BOT_TOKEN).build()

    app_tg.add_handler(CommandHandler("start", cmd_start))
    app_tg.add_handler(CommandHandler("signal", cmd_signal))
    app_tg.add_handler(CommandHandler("ta", cmd_ta))
    app_tg.add_handler(CommandHandler("subscribe", cmd_sub))
    app_tg.add_handler(CommandHandler("unsubscribe", cmd_unsub))
    app_tg.add_handler(CommandHandler("id", cmd_id))

    # schedule repeating job
    app_tg.job_queue.run_repeating(job_broadcast, interval=BROADCAST_INTERVAL, first=10)

    logger.info("Starting Telegram bot polling…")
    app_tg.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    # start Flask server in background
    threading.Thread(target=run_flask, daemon=True).start()
    # start bot
    run_bot()
