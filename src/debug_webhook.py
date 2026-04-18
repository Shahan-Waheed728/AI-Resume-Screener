import requests
import json

webhook_url = "https://shahan-waheed728.app.n8n.cloud/webhook-test/e852e99d-9807-4559-81f7-3a919da2bd65"

test_payload = {
    "candidate_name" : "Fatima Khan",
    "rf_result"      : "Qualified",
    "ml_ann_score"      : 97.05,
    "ml_match_score"    : 61.38,
    "ml_status"         : "HIRE / SHORTLIST",
    "timestamp"      : "2026-04-18 07:16:00"
}

response = requests.post(
    webhook_url,
    json=test_payload,
    timeout=5
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")