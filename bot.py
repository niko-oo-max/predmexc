import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import random

BOT_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')  # Get from environment or use placeholder

def generate_signal():
    direction = random.choice(["LONG", "SHORT"])
    entry = round(random.uniform(1.0, 2.0), 3)
    stop_loss = round(entry * (0.98 if direction == "LONG" else 1.02), 3)
    take_profit = round(entry * (1.05 if direction == "LONG" else 0.95), 3)
    return f"""
🚨 Signal Alert 🚨
Direction: {direction}
Entry Price: {entry}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to MEXC Signal Bot!")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    signal_text = generate_signal()
    await update.message.reply_text(signal_text)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("signal", signal))

app.run_polling()