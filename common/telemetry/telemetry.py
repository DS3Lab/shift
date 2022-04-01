import os
import json
import requests

headers = {
    "apikey": os.environ.get("SERVICE_ROLE_KEY"),
    "Authorization": "Bearer "+os.environ.get("SERVICE_ROLE_KEY"),
    "Content-Type": "application/json",
}

def add_event(event_type, event_data, elapsed):
    res = requests.post(
        "https://seal.yao.sh/rest/v1/telemetry",
        headers=headers,
        json={
            "event_type": event_type,
            "event_data": json.dumps(event_data),
            "elapsed_time": elapsed,
        },
    )
    return res

if __name__=="__main__":
    res = add_event("test", {"foo": "bar"}, 1)
    print(res.text)