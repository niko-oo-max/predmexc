import os
import time
import asyncio
import logging
from datetime import datetime
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import ccxtpro
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
TA_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
TIMEFRAME = "5m"

# -----------------------
# Flask server
# -----------------------
app = Flask(__name__)

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/')
def home():
    return jsonify({'service': 'TradingView Signal Bot', 'status': 'running'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'uptime': time.time()})

def run_flask_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# -----------------------
# CCXTPro live OHLCV
# -----------------------
async def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100):
    exchange = ccxtpro.binance({'enableRateLimit': True})
    await exchange.load_markets()
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    await exchange.close()
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# -----------------------
# Technical indicators
# -----------------------
def calculate_indicators(df):
    df['rsi14'] = pd.Series(df['close']).rolling(14).apply(lambda x: 100 - (100 / (1 + ((x.diff().fillna(0) > 0).sum() / 14))), raw=False)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    return df

def check_technical_signal(df):
    last = df.iloc[-1]
    if last['rsi14'] < 30 and last['close'] > last['ma20']:
        return {'signal': 'LONG', 'price': last['close'], 'confidence': 85, 
                'stop_loss': last['close'] - last['atr'], 'take_profit_1': last['close'] + last['atr'], 
                'rsi14': last['rsi14'], 'ma20': last['ma20'], 'ma50': last['ma50'], 'atr': last['atr']}
    elif last['rsi14'] > 70 and last['close'] < last['ma20']:
        return {'signal': 'SHORT', 'price': last['close'], 'confidence': 85, 
                'stop_loss': last['close'] + last['atr'], 'take_profit_1': last['close'] - last['atr'], 
                'rsi14': last['rsi14'], 'ma20': last['ma20'], 'ma50': last['ma50'], 'atr': last['atr']}
    return None

# -----------------------
# Telegram Commands
# -----------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Trading Signal Bot is online!")

async def ta_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Analyzing markets for signals...")
    signals_found = []

    for symbol in TA_SYMBOLS:
        try:
            df = await fetch_ohlcv(symbol)
            df = calculate_indicators(df)
            signal = check_technical_signal(df)
            if signal:
                signals_found.append((symbol, signal))
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")

    if signals_found:
        for symbol, data in signals_found:
            emoji = "🔥" if data['confidence'] >= 80 else "⚡"
            msg = f"""
🚨 Signal: {symbol}
{emoji} Type: {data['signal']}
💰 Entry: ${data['price']:.2f}
🎯 Confidence: {data['confidence']}%
📊 RSI14: {data['rsi14']:.1f} | MA20: {data['ma20']:.2f} | MA50: {data['ma50']:.2f}
⚡ Stop Loss: ${data['stop_loss']:.2f} | Take Profit: ${data['take_profit_1']:.2f}
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            await update.message.reply_text(msg)
    else:
        await update.message.reply_text("❌ No high-confidence signals at the moment.")

async def overview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Fetching market overview...")
    messages = []
    for symbol in TA_SYMBOLS:
        try:
            df = await fetch_ohlcv(symbol)
            df = calculate_indicators(df)
            last = df.iloc[-1]
            trend = "📈" if last['ma20'] > last['ma50'] else "📉"
            rsi_status = "🔴" if last['rsi14'] > 70 else "🟢" if last['rsi14'] < 30 else "🟡"
            msg = f"{trend} {symbol} | Price: ${last['close']:.2f} | RSI: {last['rsi14']:.1f} | MA20: {last['ma20']:.2f} | MA50: {last['ma50']:.2f} | {rsi_status}"
            messages.append(msg)
        except Exception as e:
            logger.error(f"Overview error {symbol}: {e}")
    await update.message.reply_text("\n".join(messages))

# -----------------------
# Run Telegram bot
# -----------------------
def run_bot():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN is missing!")
        return
    app_tg = Application.builder().token(BOT_TOKEN).build()
    app_tg.add_handler(CommandHandler("start", start_command))
    app_tg.add_handler(CommandHandler("ta", ta_signal_command))
    app_tg.add_handler(CommandHandler("overview", overview_command))
    logger.info("Telegram bot running...")
    app_tg.run_polling()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Start Flask server in background
    import threading
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    
    # Run Telegram bot
    run_bot()
