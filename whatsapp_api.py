import os, requests

WA_TOKEN = os.getenv("WA_TOKEN")
WA_PHONE_ID = os.getenv("WA_PHONE_ID")


def send_whatsapp_text(phone_number: str, message: str):
    """Send text message via WhatsApp Business API."""
    url = f"https://graph.facebook.com/v17.0/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "body": message
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        print(f"✅ Sent to {phone_number}")
    else:
        print(f"❌ Failed for {phone_number}: {res.text}")
