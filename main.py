import os
import time
import threading
import logging
from datetime import datetime

import asyncio
import pandas as pd
import numpy as np
import ccxt
import ta
import requests
from flask import Flask, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- Constants -------------------
TA_SYMBOLS = ["BTC/USDT", "ETH/USDT"]  # Add symbols you want
TIMEFRAME = "5m"

# ------------------- Exchange -------------------
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

# ------------------- Flask App -------------------
app = Flask(__name__)

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat(), 'message': 'TradingView Signal Bot running'})

@app.route('/')
def home():
    return jsonify({
        'service': 'TradingView Signal Bot',
        'status': 'running',
        'bot_commands': ['/start', '/signal', '/ta', '/overview', '/backtest']
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'uptime': time.time(),
        'service': 'TradingView Signal Bot',
        'version': '2.0.0',
        'data_source': 'Live Binance via CCXT'
    })

def run_flask_server():
    try:
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask error: {e}")

# ------------------- Telegram Bot Commands -------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 TradingView Signal Bot running! Use /signal, /ta, /overview, /backtest")

def fetch_ohlcv(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Fetching OHLCV failed for {symbol}: {e}")
        return None

def calculate_indicators(df):
    try:
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma9'] = df['close'].rolling(9).mean()
        df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid'] * 100
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_rsi'] = stoch.stoch()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        return df
    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")
        return df

def calculate_stop_loss_take_profit(df, signal_type, confidence_score):
    try:
        last = df.iloc[-1]
        price = last['close']
        atr = last['atr']
        bb_width = last['bb_width']
        multiplier = 1.5 + (confidence_score/100) + min(bb_width/10,0.5)
        tp_multiplier = multiplier*2
        if signal_type == 'LONG':
            sl = price - atr*multiplier
            tp1 = price + atr*tp_multiplier
            tp2 = price + atr*tp_multiplier*1.5
        else:
            sl = price + atr*multiplier
            tp1 = price - atr*tp_multiplier
            tp2 = price - atr*tp_multiplier*1.5
        rr = abs(tp1-price)/abs(price-sl) if price-sl !=0 else 0
        return {'stop_loss':round(sl,6),'take_profit_1':round(tp1,6),
                'take_profit_2':round(tp2,6),'risk_reward':round(rr,2),'atr':round(atr,6)}
    except Exception as e:
        logger.error(f"SL/TP calculation failed: {e}")
        return None

def check_technical_signal(df):
    try:
        if len(df)<51: return None
        last = df.iloc[-1]; prev=df.iloc[-2]
        ma_cross_up = prev['ma9']<=prev['ma20'] and last['ma9']>last['ma20']
        ma_cross_down = prev['ma9']>=prev['ma20'] and last['ma9']<last['ma20']
        rsi14_oversold = last['rsi14']<35
        rsi14_overbought = last['rsi14']>65
        stoch_oversold = last['stoch_rsi']<25
        stoch_overbought = last['stoch_rsi']>75
        volume_ok = last['volume_ratio']>1.2
        vol_ok = last['bb_width']>2
        long_conditions = [ma_cross_up, rsi14_oversold, stoch_oversold, volume_ok, vol_ok]
        short_conditions = [ma_cross_down, rsi14_overbought, stoch_overbought, volume_ok, vol_ok]
        long_score = sum(long_conditions)/len(long_conditions)*100
        short_score = sum(short_conditions)/len(short_conditions)*100
        min_conf = 60
        if long_score>=min_conf:
            sltp = calculate_stop_loss_take_profit(df,'LONG',long_score)
            if sltp and sltp['risk_reward']>=1.5:
                return {'signal':'LONG','price':last['close'],'confidence':round(long_score,1),
                        'rsi14':last['rsi14'],'rsi7':last['rsi7'],'stoch_rsi':last['stoch_rsi'],
                        'macd':last['macd'],'volume_ratio':last['volume_ratio'],'bb_width':last['bb_width'],**sltp}
        elif short_score>=min_conf:
            sltp = calculate_stop_loss_take_profit(df,'SHORT',short_score)
            if sltp and sltp['risk_reward']>=1.5:
                return {'signal':'SHORT','price':last['close'],'confidence':round(short_score,1),
                        'rsi14':last['rsi14'],'rsi7':last['rsi7'],'stoch_rsi':last['stoch_rsi'],
                        'macd':last['macd'],'volume_ratio':last['volume_ratio'],'bb_width':last['bb_width'],**sltp}
        return None
    except Exception as e:
        logger.error(f"Signal check failed: {e}")
        return None

def calculate_signal_probability(df, signal_data):
    try:
        if signal_data is None: return None
        last = df.iloc[-1]
        base_prob = signal_data['confidence']
        vol_factor = min(last['bb_width']/5,1)
        volu_factor = min(signal_data['volume_ratio']/2,1)
        rr_factor = min(signal_data['risk_reward']/3,1)
        occ_prob = min(base_prob*(0.7+0.3*vol_factor),95)
        win_prob = min(base_prob*0.6 + rr_factor*20 + volu_factor*15,85)
        market_strength = "STRONG" if vol_factor>0.6 and volu_factor>0.6 else "MODERATE" if vol_factor>0.3 else "WEAK"
        return {'occurrence_probability':round(occ_prob,1),'win_probability':round(win_prob,1),
                'market_strength':market_strength,'volatility_factor':round(vol_factor*100,1),'volume_factor':round(volu_factor*100,1)}
    except Exception as e:
        logger.error(f"Probability calc failed: {e}")
        return None

async def ta_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Analyzing markets...")
    signals_found=[]
    for symbol in TA_SYMBOLS:
        df=fetch_ohlcv(symbol)
        if df is not None:
            df=calculate_indicators(df)
            signal=check_technical_signal(df)
            if signal: signals_found.append((symbol,signal))
    if signals_found:
        for symbol,data in signals_found:
            df=fetch_ohlcv(symbol); df=calculate_indicators(df)
            prob_data=calculate_signal_probability(df,data)
            confidence_emoji = "🔥" if data['confidence']>=80 else "⚡" if data['confidence']>=70 else "📊"
            prob_section=""
            if prob_data:
                prob_emoji = "🟢" if prob_data['win_probability']>=70 else "🟡" if prob_data['win_probability']>=60 else "🔴"
                prob_section=f"🎲 Signal Probability Analysis:\n• {prob_emoji} Win Chance: {prob_data['win_probability']}%\n• 📈 Occurrence: {prob_data['occurrence_probability']}%\n• 💪 Market Strength: {prob_data['market_strength']}\n• 🌊 Volatility Factor: {prob_data['volatility_factor']}%\n• 📊 Volume Factor: {prob_data['volume_factor']}%\n"
            msg=f"🚨 TA Signal\n{confidence_emoji} Pair: {symbol}\n🔄 Signal: {data['signal']}\n💰 Entry: ${data['price']:.6f}\n🎯 Confidence: {data['confidence']}%\n{prob_section}🎯 Risk Management:\n• SL: ${data['stop_loss']:.6f}\n• TP1: ${data['take_profit_1']:.6f}\n• TP2: ${data['take_profit_2']:.6f}\n• RR: {data['risk_reward']}:1\n• ATR: ${data['atr']:.6f}\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            await update.message.reply_text(msg)
    else:
        await update.message.reply_text("❌ No high-confidence signals found. Try again later.")

async def market_overview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Fetching market overview...")
    overview=[]
    for symbol in TA_SYMBOLS:
        df=fetch_ohlcv(symbol)
        if df is not None:
            df=calculate_indicators(df)
            if len(df)>50:
                last=df.iloc[-1]
                overview.append({'symbol':symbol,'price':last['close'],'rsi':last['rsi14'],'ma20':last['ma20'],'ma50':last['ma50']})
    if overview:
        msg="📊 Market Overview\n"
        for d in overview:
            trend="📈" if d['ma20']>d['ma50'] else "📉"
            rsi_status="🔴" if d['rsi']>70 else "🟢" if d['rsi']<30 else "🟡"
            msg+=f"\n{trend} {d['symbol']}\n💰 Price: ${d['price']:.4f}\n📈 MA20: ${d['ma20']:.4f}\n📊 MA50: ${d['ma50']:.4f}\n{rsi_status} RSI: {d['rsi']:.1f}\n---"
        msg+=f"\n⏰ Updated: {datetime.now().strftime('%H:%M:%S UTC')}"
        await update.message.reply_text(msg)
    else:
        await update.message.reply_text("❌ Unable to fetch market data.")

# ------------------- Bot Runner -------------------
def run_bot():
    BOT_TOKEN=os.getenv("BOT_TOKEN")
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN missing!")
        return

    async def main():
        app_tg = Application.builder().token(BOT_TOKEN).build()
        app_tg.add_handler(CommandHandler("start", start_command))
        app_tg.add_handler(CommandHandler("ta", ta_signal_command))
        app_tg.add_handler(CommandHandler("overview", market_overview_command))
        await app_tg.run_polling()
    asyncio.run(main())

# ------------------- Main -------------------
if __name__ == "__main__":
    # Start Flask server
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    time.sleep(2)
    # Start Telegram bot
    run_bot()
