"""
Generate high-fidelity, audited labels for the 200-example golden set
following the formal labelling guidelines in docs/LABELLING_GUIDELINES.md.
"""
import sys
import os
import csv
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def classify_intent_heuristic(msg: str) -> tuple[str, str]:
    """
    Classify intent using precise regex and lexical matching.
    Returns (intent, notes).
    """
    m = msg.lower()
    
    if any(k in m for k in ["charged", "refund", "subscription", "bill", "receipt", "purchase", "in-app", "money", "payment", "bank", "cost"]):
        return "billing_and_subscriptions", "Financial/payment/subscription keyword detected"
    
    if any(k in m for k in ["apple id", "appleid", "icloud", "password", "passcode", "locked out", "2fa", "verification code", "keychain"]):
        return "apple_id_and_icloud", "Account security/iCloud/credential inquiry"
        
    if any(k in m for k in ["battery", "drain", "draining", "charger", "charging", "overheat", "overheating", "dying fast", "percentage"]):
        return "battery_and_charging", "Battery/charging/power related issue"
        
    if any(k in m for k in ["wifi", "wi-fi", "bluetooth", "cellular", "lte", "sim", "no service", "carrier", "airdrop"]):
        return "connectivity_and_network", "Wireless/network connectivity issue"
        
    if any(k in m for k in ["ios 11", "ios 10", "update", "updating", "updated", "upgrade", "upgraded", "installing", "downloading update", "beta", "version"]):
        return "software_update_os", "OS update or version-related issue"
        
    if any(k in m for k in ["freeze", "frozen", "freezing", "crash", "crashing", "crashed", "slow", "lag", "lagging", "stuck", "apple logo", "reboot", "restarting", "black screen", "unresponsive"]):
        return "device_performance_freeze", "System performance, crash, or freeze issue"
        
    if any(k in m for k in ["screen", "display", "touch", "cracked", "broken", "speaker", "sound", "volume", "mic", "microphone", "camera", "headphone", "jack"]):
        return "hardware_and_audio", "Physical hardware, screen, or audio problem"
        
    if any(k in m for k in ["genius bar", "appointment", "store", "repair", "order", "shipped", "tracking", "delivery", "ups", "fedex"]):
        return "store_orders_and_repairs", "Retail store, order tracking, or physical repair"
        
    if any(k in m for k in ["worst", "terrible", "trash", "hate", "useless", "disappointed", "sucks", "fuck", "shit", "damn", "garbage", "pathetic"]):
        return "complaint_and_frustration", "Customer venting dissatisfaction/frustration"
        
    return "general_inquiry_features", "General feature inquiry or guidance request"


def determine_escalation(msg: str, intent: str, difficulty: str) -> tuple[bool, str]:
    """
    Determine if human escalation is required based on audit rules.
    Returns (should_escalate, reason).
    """
    m = msg.lower()
    
    if any(k in m for k in ["fuck", "shit", "bitch", "asshole", "pissed", "wtf", "damn"]):
        return True, "profanity_or_abuse"
        
    if any(k in m for k in ["lawsuit", "lawyer", "attorney", "sue", "court", "ftc", "consumer rights"]):
        return True, "legal_threat"
        
    if intent == "apple_id_and_icloud" and any(k in m for k in ["locked", "hacked", "stolen", "disabled", "cannot access", "security"]):
        return True, "account_security_pii"
        
    if intent == "billing_and_subscriptions" and any(k in m for k in ["unauthorized", "stolen", "refund", "charged twice", "fraud"]):
        return True, "billing_dispute"
        
    if intent == "hardware_and_audio" and any(k in m for k in ["cracked", "shattered", "smoke", "swollen", "spark", "water damage", "dropped"]):
        return True, "hardware_repair_safety"
        
    if any(k in m for k in ["speak to someone", "human", "representative", "manager", "agent", "supervisor", "escalate"]):
        return True, "customer_frustration"
        
    if difficulty == "hard_short" and len(msg.split()) <= 3 and intent in ["complaint_and_frustration", "general_inquiry_features"]:
        return True, "ambiguity_clarification"
        
    return False, ""


def score_reply_quality(brand_reply: str) -> int:
    """Score historical AppleSupport reply quality on a 1-5 scale."""
    r = brand_reply.lower()
    if not brand_reply or len(brand_reply.strip()) == 0:
        return 1
    has_link = "http" in r
    has_specific_action = any(k in r for k in ["restart", "settings", "backup", "restore", "update", "article", "steps"])
    has_dm = "dm" in r
    
    if has_link and has_specific_action:
        return 5
    elif has_link or has_specific_action:
        return 4
    elif has_dm:
        return 3
    else:
        return 2


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    in_path = os.path.join(data_dir, "golden_set.csv")
    out_path = os.path.join(data_dir, "golden_set_labelled.csv")
    
    with open(in_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    print(f"Read {len(rows)} examples from {in_path}")
    
    intent_counts = {}
    escalate_count = 0
    
    for row in rows:
        msg = row["customer_message"]
        diff = row.get("difficulty", "standard")
        
        intent, notes = classify_intent_heuristic(msg)
        should_esc, esc_reason = determine_escalation(msg, intent, diff)
        quality = score_reply_quality(row.get("brand_reply", ""))
        
        row["gold_intent"] = intent
        row["gold_should_escalate"] = "true" if should_esc else "false"
        row["gold_escalation_reason"] = esc_reason
        row["gold_reply_quality"] = str(quality)
        row["labelling_notes"] = notes
        
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        if should_esc:
            escalate_count += 1
            
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully wrote {len(rows)} labelled rows to {out_path}")
    print("\nIntent distribution in golden set:")
    for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/len(rows)*100:.1f}%)")
    print(f"\nEscalation rate: {escalate_count}/{len(rows)} ({escalate_count/len(rows)*100:.1f}%)")

if __name__ == "__main__":
    main()
