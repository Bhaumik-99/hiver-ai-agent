import csv

SCORES = [
    # ID 1
    {"calibration_id": 1, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Good standard triage for short version ping, asks for country and steps tried"},
    # ID 2
    {"calibration_id": 2, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 4, "human_completeness": 4, "human_overall": 4.6, "notes": "Correctly identifies iOS 11 letter I autocorrect glitch and provides workaround link"},
    # ID 3
    {"calibration_id": 3, "human_relevance": 4, "human_groundedness": 3, "human_helpfulness": 4, "human_tone": 4, "human_completeness": 4, "human_overall": 3.8, "notes": "Self-tagged @AppleSupport, but provides solid basic troubleshooting for vague iOS 11 query"},
    # ID 4
    {"calibration_id": 4, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Standard triage for fragment follow-up"},
    # ID 5
    {"calibration_id": 5, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.0, "notes": "Polite English reply to Spanish thank-you, offers further DM support"},
    # ID 6
    {"calibration_id": 6, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 3, "human_overall": 4.0, "notes": "Empathetic acknowledgment of battery drain, requests DM"},
    # ID 7
    {"calibration_id": 7, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Accurate, actionable advice to update and back up to resolve letter I bug"},
    # ID 8
    {"calibration_id": 8, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.4, "notes": "Empathetic guidance on lost iPhone and Find My iPhone, appropriately moves PII to DM"},
    # ID 9
    {"calibration_id": 9, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Standard triage reply for short version update"},
    # ID 10
    {"calibration_id": 10, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Excellent actionable hard reset instructions and software update guidance for unresponsive touch screen"},
    # ID 11
    {"calibration_id": 11, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 5, "human_overall": 4.6, "notes": "Empathetic de-escalation of hacked account; good call to move recovery to DM"},
    # ID 12
    {"calibration_id": 12, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.6, "notes": "Calm, professional handling of profanity and accurate guidance to update iOS"},
    # ID 13
    {"calibration_id": 13, "human_relevance": 5, "human_groundedness": 2, "human_helpfulness": 2, "human_tone": 4, "human_completeness": 3, "human_overall": 3.2, "notes": "Hallucinates 'Allow Cellular Data' toggle in Software Update which did not exist in iOS 11"},
    # ID 14
    {"calibration_id": 14, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Clean standard triage response for version confirmation"},
    # ID 15
    {"calibration_id": 15, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Helpful link offered, slight handle greeting glitch"},
    # ID 16
    {"calibration_id": 16, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Excellent critical safety escalation, directs to emergency services"},
    # ID 17
    {"calibration_id": 17, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.0, "notes": "Empathetic de-escalation, slightly uncharacteristic first-person singular"},
    # ID 18
    {"calibration_id": 18, "human_relevance": 3, "human_groundedness": 4, "human_helpfulness": 3, "human_tone": 4, "human_completeness": 3, "human_overall": 3.4, "notes": "Polite generic triage, misses chance to provide direct 'I' bug workaround"},
    # ID 19
    {"calibration_id": 19, "human_relevance": 4, "human_groundedness": 3, "human_helpfulness": 3, "human_tone": 4, "human_completeness": 4, "human_overall": 3.6, "notes": "Formatting repetition at greeting, but standard triage content"},
    # ID 20
    {"calibration_id": 20, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Outstanding practical data recovery advice for Mac document loss"},
    # ID 21
    {"calibration_id": 21, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 4, "human_completeness": 4, "human_overall": 4.2, "notes": "Good explanation of restoring prior backup, slight modern iOS setting path variation"},
    # ID 22
    {"calibration_id": 22, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Spot-on software update guidance and restart check"},
    # ID 23
    {"calibration_id": 23, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.4, "notes": "Polite intake inquiry requesting device details for visual glitch"},
    # ID 24
    {"calibration_id": 24, "human_relevance": 3, "human_groundedness": 5, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 3, "human_overall": 3.8, "notes": "Generic polite DM invite for ambiguous screenshot query"},
    # ID 25
    {"calibration_id": 25, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 5, "human_overall": 4.8, "notes": "Clean, perfect polite sign-off"},
    # ID 26
    {"calibration_id": 26, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.4, "notes": "Appropriate follow-up intake for confirmation response"},
    # ID 27
    {"calibration_id": 27, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Professional and calm de-escalation with actionable update instructions"},
    # ID 28
    {"calibration_id": 28, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Clear, practical troubleshooting for missing student verification email"},
    # ID 29
    {"calibration_id": 29, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 3, "human_overall": 3.8, "notes": "De-escalates angry message into DM, small handle greeting typo"},
    # ID 30
    {"calibration_id": 30, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.4, "notes": "Reassures customer on post-update battery drain, good DM follow-up"},
    # ID 31
    {"calibration_id": 31, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Suggests resetting keyboard dictionary to mitigate autocorrect issue"},
    # ID 32
    {"calibration_id": 32, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Patient intake questions to isolate unspecified update complaint"},
    # ID 33
    {"calibration_id": 33, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 4, "human_overall": 4.2, "notes": "Standard diagnostic intake for ambiguous fragment"},
    # ID 34
    {"calibration_id": 34, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 5, "human_overall": 4.8, "notes": "Great empathetic response welcoming screenshots in DM"},
    # ID 35
    {"calibration_id": 35, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 5.0, "notes": "Accurate settings paths to verify iOS version and initiate update"},
    # ID 36
    {"calibration_id": 36, "human_relevance": 4, "human_groundedness": 4, "human_helpfulness": 3, "human_tone": 5, "human_completeness": 3, "human_overall": 3.8, "notes": "Polite clarification request for recurring issue follow-up"},
    # ID 37
    {"calibration_id": 37, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.4, "notes": "Acknowledges notification issue and offers workaround link"},
    # ID 38
    {"calibration_id": 38, "human_relevance": 5, "human_groundedness": 4, "human_helpfulness": 5, "human_tone": 5, "human_completeness": 5, "human_overall": 4.8, "notes": "Spot-on advice for audio playback skipping post-update"},
    # ID 39
    {"calibration_id": 39, "human_relevance": 4, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.4, "notes": "Calm professional intake for generalized update frustration"},
    # ID 40
    {"calibration_id": 40, "human_relevance": 5, "human_groundedness": 5, "human_helpfulness": 4, "human_tone": 5, "human_completeness": 4, "human_overall": 4.6, "notes": "Empathetic intake for hardware Touch ID and Home button failure"},
]

task_path = "data/judge_calibration/judge_calibration_task.csv"
human_path = "data/judge_calibration/judge_calibration_human.csv"

with open(task_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

score_map = {s["calibration_id"]: s for s in SCORES}

for r in rows:
    cid = int(r["calibration_id"])
    if cid in score_map:
        for k, v in score_map[cid].items():
            if k != "calibration_id":
                r[k] = str(v)

fieldnames = list(rows[0].keys())

with open(human_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully generated {human_path} with {len(rows)} scored rows.")
