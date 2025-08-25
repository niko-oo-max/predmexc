# MEXC Trading Signal Telegram Bot

## Overview

This is a Telegram bot that generates random trading signals for MEXC exchange, designed to run 24/7 with a Flask web server for uptime monitoring. The bot provides users with mock trading signals including entry points, stop losses, and take profit levels for popular cryptocurrency trading pairs. The application combines a Telegram bot interface with a web server to ensure reliable operation and monitoring capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Structure
The project follows a dual-service architecture combining a Telegram bot with a Flask web server:

- **main.py**: Primary application file containing the complete bot implementation with Flask integration
- **bot.py**: Simplified standalone bot version (appears to be a basic template)
- **Flask Web Server**: Provides health monitoring endpoints for uptime services

### Bot Architecture
- **Framework**: Built using python-telegram-bot library with async/await patterns
- **Command Handlers**: Implements command-based interaction model with `/start` and `/signal` commands
- **Signal Generation**: Uses Python's random module to generate mock trading signals
- **Trading Pairs**: Predefined list of popular cryptocurrency pairs (BTCUSDT, ETHUSDT, etc.)

### Web Server Integration
- **Health Monitoring**: Flask endpoints (`/`, `/ping`, `/health`) for uptime monitoring services
- **Threading**: Runs Telegram bot and Flask server concurrently using Python threading
- **Logging**: Comprehensive logging system for monitoring and debugging

### Signal Generation Logic
- **Random Direction**: Generates LONG or SHORT positions randomly
- **Price Calculations**: Entry prices, stop losses, and take profit levels calculated with basic percentage-based risk management
- **Risk Management**: Includes leverage recommendations and educational disclaimers

### Deployment Strategy
- **Replit-Optimized**: Designed specifically for Replit deployment with environment variable configuration
- **24/7 Operation**: Integrated with external uptime monitoring services (UptimeRobot, cron-job.org)
- **Environment Configuration**: Uses Replit's secrets management for secure token storage

## External Dependencies

### Core Dependencies
- **python-telegram-bot**: Telegram Bot API integration
- **Flask**: Web server framework for health monitoring
- **Standard Library**: threading, logging, random, datetime, asyncio, os

### Telegram Integration
- **Bot Token**: Requires Telegram bot token from @BotFather
- **Webhook/Polling**: Uses polling mode for receiving updates

### Monitoring Services
- **UptimeRobot**: External uptime monitoring service integration
- **cron-job.org**: Alternative uptime monitoring option
- **Health Endpoints**: Custom endpoints for monitoring service integration

### Deployment Platform
- **Replit**: Primary deployment platform with specific optimizations
- **Environment Variables**: BOT_TOKEN stored securely in Replit secrets
- **Port Configuration**: Automatic port assignment for web server