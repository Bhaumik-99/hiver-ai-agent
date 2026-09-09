import csv

with open("data/judge_calibration/judge_calibration_task.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

with open("data/judge_calibration/inspect_dump.txt", "w", encoding="utf-8") as out:
    for r in rows:
        out.write(f"ID {r['calibration_id']} [{r['system']}]:\n")
        out.write(f"  CUSTOMER: {r['customer_message']}\n")
        out.write(f"  REPLY:    {r['generated_reply']}\n\n")

print(f"Dumped {len(rows)} rows to data/judge_calibration/inspect_dump.txt")
