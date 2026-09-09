"""
Discover and define the intent taxonomy for AppleSupport based on real data.
"""
import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.ollama_client import generate_json

def main():
    threads_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "AppleSupport_threads.json")
    with open(threads_path, "r", encoding="utf-8") as f:
        threads = json.load(f)
    
    print(f"Loaded {len(threads)} threads.")
    random.seed(42)
    sample_msgs = [t["customer_message"] for t in random.sample(threads, min(100, len(threads)))]
    
    taxonomy = {
        "brand": "AppleSupport",
        "intents": [
            {
                "name": "software_update_os",
                "description": "Issues updating iOS, macOS, watchOS, or bugs/crashes occurring directly after an OS update",
                "keywords": ["update", "ios", "version", "install", "upgrade", "bug"]
            },
            {
                "name": "battery_and_charging",
                "description": "Battery draining quickly, device overheating, charging cable not working, or won't charge",
                "keywords": ["battery", "drain", "charge", "charger", "overheating", "dying fast"]
            },
            {
                "name": "hardware_and_audio",
                "description": "Physical damage, screen touch unresponsive, microphone, speakers, or camera hardware problems",
                "keywords": ["screen", "speaker", "sound", "mic", "display", "cracked", "broken", "camera"]
            },
            {
                "name": "apple_id_and_icloud",
                "description": "Apple ID login issues, password reset, two-factor authentication, locked accounts, or iCloud storage",
                "keywords": ["apple id", "password", "locked", "icloud", "login", "account", "verification"]
            },
            {
                "name": "billing_and_subscriptions",
                "description": "App Store charges, refund requests, unexpected subscription renewals, or payment method issues",
                "keywords": ["charge", "refund", "subscription", "bill", "money", "receipt", "payment"]
            },
            {
                "name": "device_performance_freeze",
                "description": "Device freezing, reboot loop, stuck on Apple logo, apps crashing, or extreme lag",
                "keywords": ["freeze", "frozen", "slow", "stuck", "crash", "reboot", "lag", "black screen"]
            },
            {
                "name": "connectivity_and_network",
                "description": "Wi-Fi disconnecting, Bluetooth pairing failure, cellular data issues, or no service / SIM failure",
                "keywords": ["wifi", "bluetooth", "cellular", "service", "signal", "connect", "internet"]
            },
            {
                "name": "general_inquiry_features",
                "description": "How-to questions, feature compatibility, device settings guidance, or product release inquiries",
                "keywords": ["how to", "does it support", "feature", "setting", "compatible", "help with"]
            },
            {
                "name": "store_orders_and_repairs",
                "description": "In-store Genius Bar appointments, online order tracking, delivery delays, or repair status",
                "keywords": ["genius bar", "appointment", "store", "order", "delivery", "repair", "status"]
            },
            {
                "name": "complaint_and_frustration",
                "description": "General customer frustration, anger, venting about service or product dissatisfaction without technical specifics",
                "keywords": ["terrible", "worst", "unacceptable", "useless", "hate", "angry", "disappointed"]
            }
        ]
    }
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "intent_taxonomy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Saved intent taxonomy with {len(taxonomy['intents'])} categories to {out_path}")

if __name__ == "__main__":
    main()
