# main.py
import os
import logging
import threading
import time
from datetime import datetime
import requests
import random

from flask import Flask, jsonify
import ccxt
import pandas as pd
import numpy as np
import ta

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ----------------- Logging -----------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ----------------- Flask App -----------------
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'service': 'TradingView Signal Bot',
        'status': 'running',
        'commands': ['/start','/signal','/ta','/overview','/backtest']
    })

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok','timestamp': datetime.utcnow().isoformat()})

@app.route('/health')
def health():
    return jsonify({'status':'healthy','service':'TradingView Signal Bot','version':'2.0.0'})

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ----------------- Config -----------------
TRADING_PAIRS = ['BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','XRPUSDT','SOLUSDT','DOTUSDT','LINKUSDT','LTCUSDT','AVAXUSDT','MATICUSDT','UNIUSDT','ATOMUSDT','FILUSDT','TRXUSDT']
TA_SYMBOLS = ['BTC/USDT','ETH/USDT','SOL/USDT']
TIMEFRAME = '5m'

# ----------------- Exchange -----------------
def get_exchange():
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        exchange.load_markets()
        logger.info("Connected to Binance")
        return exchange
    except Exception as e:
        logger.error(f"Exchange error: {e}")
        return None

exchange = get_exchange()

# ----------------- Price & OHLCV -----------------
def fetch_ohlcv(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

# ----------------- Indicators -----------------
def calculate_indicators(df):
    df['ma9'] = df['close'].rolling(9).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['rsi14'] = ta.momentum.RSIIndicator(df['close'],14).rsi()
    df['rsi7'] = ta.momentum.RSIIndicator(df['close'],7).rsi()
    bb = ta.volatility.BollingerBands(df['close'],20,2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_high']-df['bb_low'])/df['bb_mid']*100
    stoch = ta.momentum.StochasticOscillator(df['high'],df['low'],df['close'],14,3)
    df['stoch_rsi'] = stoch.stoch()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume']/df['volume_sma']
    df['atr'] = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close'],14).average_true_range()
    return df

# ----------------- Signal Calculation -----------------
def calculate_sl_tp(df, signal_type, confidence):
    last = df.iloc[-1]
    price = last['close']
    atr = last['atr']
    factor = 1.5 + (confidence/100)
    vol_adj = min(last['bb_width']/10,0.5)
    atr_mult = factor + vol_adj
    tp_mult = atr_mult*2
    if signal_type=='LONG':
        stop = price - atr*atr_mult
        tp1 = price + atr*tp_mult
        tp2 = price + atr*tp_mult*1.5
    else:
        stop = price + atr*atr_mult
        tp1 = price - atr*tp_mult
        tp2 = price - atr*tp_mult*1.5
    risk = abs(price-stop)
    reward = abs(tp1-price)
    rr = reward/risk if risk>0 else 0
    return {'stop_loss':round(stop,6),'take_profit_1':round(tp1,6),'take_profit_2':round(tp2,6),'risk_reward':round(rr,2),'atr':round(atr,6)}

def check_signal(df):
    if len(df)<51: return None
    last, prev = df.iloc[-1], df.iloc[-2]
    ma9_up = last['ma9']>last['ma20']
    ma9_down = last['ma9']<last['ma20']
    rsi_overb = last['rsi14']>65 or last['rsi7']>75
    rsi_overs = last['rsi14']<35 or last['rsi7']<25
    stoch_overb = last['stoch_rsi']>75
    stoch_overs = last['stoch_rsi']<25
    vol_confirm = last['volume_ratio']>1.2
    volat = last['bb_width']>2
    confidence_long = sum([ma9_up,rsi_overs,stoch_overs,vol_confirm,volat])/5*100
    confidence_short = sum([ma9_down,rsi_overb,stoch_overb,vol_confirm,volat])/5*100
    min_conf = 60
    if confidence_long>=min_conf:
        sltp = calculate_sl_tp(df,'LONG',confidence_long)
        if sltp['risk_reward']>=1.5:
            return {'signal':'LONG','price':last['close'],'confidence':round(confidence_long,1),**sltp}
    if confidence_short>=min_conf:
        sltp = calculate_sl_tp(df,'SHORT',confidence_short)
        if sltp['risk_reward']>=1.5:
            return {'signal':'SHORT','price':last['close'],'confidence':round(confidence_short,1),**sltp}
    return None

# ----------------- Telegram Commands -----------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 Welcome to TradingView Signal Bot!\nCommands: /signal /ta /overview /backtest")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pair = random.choice(TRADING_PAIRS)
    df = fetch_ohlcv(pair)
    if df is not None:
        df = calculate_indicators(df)
        sig = check_signal(df)
        if sig:
            msg=f"🚨 Signal: {sig['signal']}\n💰 Price: {sig['price']}\n🛑 SL: {sig['stop_loss']}\n✅ TP1: {sig['take_profit_1']}\n🎯 RR: {sig['risk_reward']}\nConfidence: {sig['confidence']}%"
            await update.message.reply_text(msg)
        else:
            await update.message.reply_text("📊 No high-confidence signal detected.")
    else:
        await update.message.reply_text("❌ Unable to fetch data.")

async def ta_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg="🔍 TA Analysis:\n"
    for sym in TA_SYMBOLS:
        df = fetch_ohlcv(sym)
        if df is not None:
            df = calculate_indicators(df)
            sig = check_signal(df)
            if sig:
                msg+=f"\n{sym}: {sig['signal']} Price:{sig['price']} SL:{sig['stop_loss']} TP:{sig['take_profit_1']} RR:{sig['risk_reward']} Confidence:{sig['confidence']}%"
    await update.message.reply_text(msg or "📊 No signals found.")

async def overview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg="📊 Market Overview:\n"
    for sym in TA_SYMBOLS:
        df = fetch_ohlcv(sym)
        if df is not None:
            df = calculate_indicators(df)
            last = df.iloc[-1]
            trend="📈" if last['ma20']>last['ma50'] else "📉"
            rsi="🟢" if last['rsi14']<30 else "🔴" if last['rsi14']>70 else "🟡"
            msg+=f"\n{trend} {sym} Price:{last['close']} MA20:{last['ma20']:.2f} MA50:{last['ma50']:.2f} RSI:{last['rsi14']:.1f} {rsi}"
    await update.message.reply_text(msg)

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg="📊 Backtest results not implemented in this version."
    await update.message.reply_text(msg)

# ----------------- Run Telegram Bot -----------------
def run_bot():
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN missing")
        return
    app_tg = Application.builder().token(BOT_TOKEN).build()
    app_tg.add_handler(CommandHandler("start", start_command))
    app_tg.add_handler(CommandHandler("signal", signal_command))
    app_tg.add_handler(CommandHandler("ta", ta_signal_command))
    app_tg.add_handler(CommandHandler("overview", overview_command))
    app_tg.add_handler(CommandHandler("backtest", backtest_command))
    app_tg.run_polling()

# ----------------- Main -----------------
if __name__=="__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)
    run_bot()
