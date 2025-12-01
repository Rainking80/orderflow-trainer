"""
Simple test script to verify Telegram notification functionality
"""
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def test_telegram_message():
    """Test sending a simple message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Telegram credentials not configured in .env file")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return False
    
    print(f"Testing Telegram notification...")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:10]}... (truncated)")
    print(f"Chat ID: {TELEGRAM_CHAT_ID}")
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": "🧪 <b>Test Message</b>\n\nTelegram integration is working correctly!",
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("✓ Telegram message sent successfully!")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Failed to send Telegram message: {e}")
        return False

if __name__ == "__main__":
    test_telegram_message()
