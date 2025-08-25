# main.py
import os
import logging
import threading
import time
from datetime import datetime
from flask import Flask, jsonify
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, JobQueue
import ccxt
import pandas as pd
import ta

# -------------------------------
# CONFIG
# -------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")  # Chat ID for broadcasting signals
BROADCAST_INTERVAL = 60  # seconds
TIMEFRAME = '5m'

TRADING_PAIRS = ['BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','XRPUSDT',
                 'SOLUSDT','DOTUSDT','LINKUSDT','LTCUSDT','AVAXUSDT',
                 'MATICUSDT','UNIUSDT','ATOMUSDT','FILUSDT','TRXUSDT']
TA_SYMBOLS = ['BTC/USDT','SOL/USDT','ETH/USDT']

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# FLASK APP
# -------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"running","time_utc":datetime.utcnow().isoformat() + "Z"})

@app.route("/ping")
def ping():
    return jsonify({"status":"ok","timestamp":datetime.utcnow().isoformat(),
                    "message":"TradingView Signal Bot is running"})

# -------------------------------
# EXCHANGE
# -------------------------------
def get_working_exchange():
    exchanges_to_try = [('bybit', ccxt.bybit), ('okx', ccxt.okx), ('kucoin', ccxt.kucoin), ('mexc', ccxt.mexc)]
    for name, ex_class in exchanges_to_try:
        try:
            ex = ex_class({'enableRateLimit': True, 'sandbox': False})
            ex.load_markets()
            logger.info(f"Connected to {name}")
            return ex
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
    return None

exchange = get_working_exchange()

def fetch_ohlcv(symbol):
    try:
        if exchange is None: return None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None

def calculate_indicators(df):
    try:
        df['ma9'] = df['close'].rolling(9).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low'])/df['bb_mid']*100
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_rsi'] = stoch.stoch()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume']/df['volume_sma']
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['support'] = df['low'].rolling(20).min()
        df['resistance'] = df['high'].rolling(20).max()
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

# -------------------------------
# SIGNAL LOGIC
# -------------------------------
def check_technical_signal(df):
    try:
        if len(df)<51: return None
        last, prev = df.iloc[-1], df.iloc[-2]
        ma9_cross_up = prev['ma9']<=prev['ma20'] and last['ma9']>last['ma20']
        ma9_cross_down = prev['ma9']>=prev['ma20'] and last['ma9']<last['ma20']
        rsi_oversold = last['rsi14']<35 or last['rsi7']<25
        rsi_overbought = last['rsi14']>65 or last['rsi7']>75
        stoch_oversold = last['stoch_rsi']<25
        stoch_overbought = last['stoch_rsi']>75
        bb_width_ok = last['bb_width']>2
        long_cond = [ma9_cross_up, rsi_oversold, stoch_oversold, bb_width_ok]
        short_cond = [ma9_cross_down, rsi_overbought, stoch_overbought, bb_width_ok]
        long_score = sum(long_cond)/len(long_cond)*100
        short_score = sum(short_cond)/len(short_cond)*100
        min_conf=60
        def sl_tp(price,atr,typ,conf):
            mult = 1.5 + conf/100
            if typ=='LONG':
                sl=price-atr*mult; tp1=price+atr*mult*2; tp2=price+atr*mult*3
            else: sl=price+atr*mult; tp1=price-atr*mult*2; tp2=price-atr*mult*3
            rr=abs(tp1-price)/abs(price-sl) if abs(price-sl)>0 else 0
            return {'stop_loss':round(sl,6),'take_profit_1':round(tp1,6),'take_profit_2':round(tp2,6),'risk_reward':round(rr,2),'atr':round(atr,6)}
        if long_score>=min_conf: return {'signal':'LONG','price':last['close'],'confidence':round(long_score,1),**sl_tp(last['close'],last['atr'],'LONG',long_score)}
        elif short_score>=min_conf: return {'signal':'SHORT','price':last['close'],'confidence':round(short_score,1),**sl_tp(last['close'],last['atr'],'SHORT',short_score)}
        return None
    except Exception as e:
        logger.error(f"Signal check error: {e}")
        return None

def calculate_signal_probability(df,sig):
    try:
        if not sig: return None
        last=df.iloc[-1]
        base_prob=sig['confidence']
        vol_factor=min(last['bb_width']/5,1.0)
        rr_factor=min(sig['risk_reward']/3,1.0)
        occurrence=min(base_prob*(0.7+0.3*vol_factor),95)
        win_prob=min(base_prob*0.6+rr_factor*20+min(sig.get('volume_ratio',1)/2*15,15),85)
        market_strength="STRONG" if vol_factor>0.6 else "MODERATE" if vol_factor>0.3 else "WEAK"
        return {'occurrence_probability':round(occurrence,1),'win_probability':round(win_prob,1),'market_strength':market_strength}
    except Exception as e:
        logger.error(f"Probability calc error: {e}")
        return None

# -------------------------------
# TELEGRAM COMMANDS
# -------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📈 TradingView Signal Bot is running!")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol='BTCUSDT'
    df=fetch_ohlcv(symbol)
    if df is None: await update.message.reply_text("❌ Unable to fetch live data."); return
    df=calculate_indicators(df)
    sig=check_technical_signal(df)
    prob=calculate_signal_probability(df,sig)
    msg=f"🚨 Signal: {sig['signal'] if sig else 'NONE'}\nPrice: {sig['price'] if sig else 'N/A'}\nConfidence: {sig['confidence'] if sig else 'N/A'}\nWin Chance: {prob['win_probability'] if prob else 'N/A'}%"
    await update.message.reply_text(msg)

async def ta_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Analyzing TA signals...")
    for symbol in TA_SYMBOLS:
        df=fetch_ohlcv(symbol)
        if df is None: continue
        df=calculate_indicators(df)
        sig=check_technical_signal(df)
        if sig:
            prob=calculate_signal_probability(df,sig)
            msg=f"🚨 {symbol} Signal: {sig['signal']}\nPrice: {sig['price']}\nConfidence: {sig['confidence']}%\nWin Chance: {prob['win_probability']}%" if prob else ""
            await update.message.reply_text(msg)

async def market_overview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Market Overview:")
    for symbol in TA_SYMBOLS:
        df=fetch_ohlcv(symbol)
        if df is None: continue
        df=calculate_indicators(df)
        last=df.iloc[-1]
        trend="📈" if last['ma20']>last['ma50'] else "📉"
        rsi_status="🔴" if last['rsi14']>70 else "🟢" if last['rsi14']<30 else "🟡"
        await update.message.reply_text(f"{trend} {symbol}\nPrice: {last['close']:.2f}\nRSI: {last['rsi14']:.1f} {rsi_status}")

# -------------------------------
# RUN BOT
# -------------------------------
def run_telegram_bot():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not found")
        return
    app_bot = ApplicationBuilder().token(BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start_command))
    app_bot.add_handler(CommandHandler("signal", signal_command))
    app_bot.add_handler(CommandHandler("ta", ta_signal_command))
    app_bot.add_handler(CommandHandler("overview", market_overview_command))
    logger.info("Starting Telegram bot...")
    app_bot.run_polling()

def run_flask_server():
    logger.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# -------------------------------
# MAIN
# -------------------------------
def main():
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    time.sleep(2)
    run_telegram_bot()

if __name__=="__main__":
    main()
