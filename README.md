# MEXC Trading Signal Telegram Bot

A Telegram bot that generates random trading signals for MEXC exchange with a Flask web server for 24/7 uptime monitoring.

## Features

🤖 **Telegram Bot Commands:**
- `/start` - Welcome message and bot introduction
- `/signal` - Generate random trading signals (LONG/SHORT with entry, SL, TP)

🌐 **Web Server Endpoints:**
- `/` - Home page with service information
- `/ping` - Health check for uptime monitoring
- `/health` - Detailed health status

📊 **Signal Features:**
- Random LONG/SHORT signals
- Popular crypto trading pairs (BTC, ETH, BNB, etc.)
- Entry price, Stop Loss, Take Profit levels
- Leverage recommendations (5x-20x)
- Risk management tips

## Setup Instructions

### 1. Get a Telegram Bot Token
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` command
3. Follow instructions to create your bot
4. Copy the bot token provided

### 2. Replit Deployment
1. Create a new Python Repl
2. Upload/paste the code files
3. Go to "Secrets" tab in Replit
4. Add secret: `BOT_TOKEN` = your bot token
5. Click the green "Run" button

### 3. Uptime Monitoring (Optional)
Set up monitoring with UptimeRobot or cron-job.org:
- Monitor URL: `https://your-repl-name.your-username.repl.co/ping`
- Check interval: 5-10 minutes

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BOT_TOKEN` | Telegram bot token from @BotFather | Yes |

## Technical Details

- **Backend**: Python with Flask and python-telegram-bot
- **Hosting**: Optimized for Replit free tier
- **Uptime**: Flask server keeps Repl active 24/7
- **Threading**: Concurrent bot and web server operation
- **Logging**: Comprehensive logging for debugging

## Usage

1. Start a chat with your bot on Telegram
2. Send `/start` to see welcome message
3. Send `/signal` to get trading signals
4. Monitor uptime via web endpoints

## Disclaimer

⚠️ **Important**: This bot generates random signals for educational purposes only. Always do your own research and risk management before trading. Never risk more than you can afford to lose.

## Support

For issues or questions:
1. Check the logs in Replit console
2. Verify BOT_TOKEN is set correctly
3. Ensure bot is active with @BotFather

## File Structure

