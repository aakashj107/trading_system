import os
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# This function runs when someone sends /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am alive and working!")


def main():
    # Token comes from the TELEGRAM_BOT_TOKEN environment variable (.env)
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set. Add it to your .env file.")
        sys.exit(1)

    # Start the bot
    app = Application.builder().token(TOKEN).build()

    # Tell the bot to watch for the /start command
    app.add_handler(CommandHandler("start", start))

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()


import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)


if __name__ == "__main__":
    main()
