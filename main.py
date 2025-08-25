import os
import logging
import threading
import time
from datetime import datetime
from flask import Flask, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import ccxt
import pandas as pd
import ta
import requests

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------------------
# Config
# -------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")  # Optional: for broadcast jobs
TIMEFRAME = "5m"
TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'DOT/USDT', 'ADA/USDT']

# -------------------------------
# Exchange setup
# -------------------------------
def get_exchange():
    for ex_name, ex_class in [('binance', ccxt.binance), ('bybit', ccxt.bybit), ('okx', ccxt.okx), ('kucoin', ccxt.kucoin)]:
        try:
            ex = ex_class({'enableRateLimit': True})
            ex.load_markets()
            logger.info(f"Connected to {ex_name}")
            return ex
        except Exception as e:
            logger.warning(f"Failed to connect to {ex_name}: {e}")
    return None

exchange = get_exchange()
if exchange is None:
    logger.error("No working exchange found. Exiting.")
    exit(1)

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "running", "time_utc": datetime.utcnow().isoformat() + "Z"})

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat(), "message": "Bot running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "TradingView Signal Bot", "uptime": time.time()})

def run_flask():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, use_reloader=False)

# -------------------------------
# Helper functions
# -------------------------------
def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

def calculate_indicators(df):
    df['ma9'] = df['close'].rolling(9).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close'] * 100
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
    df['macd_histogram'] = ta.trend.MACD(df['close']).macd_diff()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    return df

def check_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signal = None
    confidence = 0

    if last['ma9'] > last['ma20'] > last['ma50'] and last['rsi14'] < 60:
        signal = 'LONG'
        confidence = 70
    elif last['ma9'] < last['ma20'] < last['ma50'] and last['rsi14'] > 40:
        signal = 'SHORT'
        confidence = 70

    if signal:
        stop_loss = last['close'] - last['atr'] if signal == 'LONG' else last['close'] + last['atr']
        take_profit = last['close'] + last['atr']*2 if signal == 'LONG' else last['close'] - last['atr']*2
        rr = (take_profit - last['close'])/(last['close'] - stop_loss) if signal == 'LONG' else (last['close'] - take_profit)/(stop_loss - last['close'])
        return {
            'signal': signal,
            'price': last['close'],
            'stop_loss': round(stop_loss, 6),
            'take_profit': round(take_profit, 6),
            'risk_reward': round(rr, 2),
            'confidence': confidence
        }
    return None

# -------------------------------
# Telegram bot commands
# -------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Bot is running and fetching live market data.")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages = []
    for symbol in TRADING_PAIRS:
        df = fetch_ohlcv(symbol)
        if df is not None:
            df = calculate_indicators(df)
            signal = check_signal(df)
            if signal:
                messages.append(f"{symbol} - {signal['signal']} @ {signal['price']:.2f} | TP: {signal['take_profit']:.2f} | SL: {signal['stop_loss']:.2f} | R:R: {signal['risk_reward']}")
    if messages:
        await update.message.reply_text("\n".join(messages))
    else:
        await update.message.reply_text("No high-confidence signals found.")

# -------------------------------
# Bot runner
# -------------------------------
def run_bot():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set in environment!")
        return
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("signal", signal_command))
    application.run_polling()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Start Flask in separate thread
    threading.Thread(target=run_flask, daemon=True).start()
    # Start Telegram bot
    run_bot()
