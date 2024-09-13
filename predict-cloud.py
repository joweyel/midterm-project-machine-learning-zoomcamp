import os
import sys
import json
import requests

host = os.getenv("EB_ENDPOINT")
if not host:
    print("Environment variable EB_ENDPOINT not set!")
    sys.exit(1)
print("Sending data to host: ", host)

url = f"http://{host}/predict"

def load_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data

def send_data(data):
    result = requests.post(url, json=data).json()
    print(result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = str(sys.argv[1])
        if os.path.exists(path) and path.endswith(".json"):
            send_data(load_json(path))
        else:
            print("No json given!")
    else:
        print("Nothing specified!")
