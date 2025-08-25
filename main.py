import os
import logging
import threading
import time
from datetime import datetime
import requests
import ccxt
import pandas as pd
import ta
from flask import Flask, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Flask server
app = Flask(__name__)

# Trading pairs
TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
TA_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAME = '5m'

# Environment
BOT_TOKEN = os.getenv('BOT_TOKEN')

# ----- Exchange setup -----
def get_working_exchange():
    exchanges_to_try = [
        ccxt.binance,
        ccxt.bybit,
        ccxt.okx,
        ccxt.kucoin,
        ccxt.mexc
    ]
    for ex_class in exchanges_to_try:
        try:
            ex = ex_class({'enableRateLimit': True})
            ex.load_markets()
            logger.info(f"Connected to {ex.id}")
            return ex
        except Exception as e:
            logger.warning(f"Failed {ex_class.__name__}: {e}")
    return None

exchange = get_working_exchange()

# ----- Price fetch -----
def get_real_price(symbol):
    try:
        data = exchange.fetch_ticker(symbol)
        return float(data['last'])
    except Exception as e:
        logger.warning(f"Failed fetching price for {symbol}: {e}")
        return None

# ----- OHLCV fetch -----
def fetch_ohlcv(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Failed fetching OHLCV {symbol}: {e}")
        return None

# ----- Indicators -----
def calculate_indicators(df):
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['rsi14'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
    return df

# ----- Telegram commands -----
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Bot is running. Use /signal or /ta for analysis.")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = TRADING_PAIRS[0]
    price = get_real_price(pair)
    await update.message.reply_text(f"📊 Current price of {pair}: ${price}")

async def ta_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📊 TA Signals:\n"
    for sym in TA_SYMBOLS:
        df = fetch_ohlcv(sym)
        if df is not None:
            df = calculate_indicators(df)
            last = df.iloc[-1]
            msg += f"{sym}: Close={last['close']:.2f} RSI={last['rsi14']:.1f}\n"
    await update.message.reply_text(msg)

# ----- Flask routes -----
@app.route('/ping')
def ping():
    return jsonify({'status':'ok','timestamp':datetime.now().isoformat()})

# ----- Bot runner -----
def run_bot():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set!")
        return
    app_tg = Application.builder().token(BOT_TOKEN).build()
    app_tg.job_queue = None  # Prevent weakref error on Render
    app_tg.add_handler(CommandHandler("start", start_command))
    app_tg.add_handler(CommandHandler("signal", signal_command))
    app_tg.add_handler(CommandHandler("ta", ta_signal_command))
    logger.info("Starting bot...")
    app_tg.run_polling()

# ----- Main -----
if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False), daemon=True).start()
    time.sleep(2)
    run_bot()
