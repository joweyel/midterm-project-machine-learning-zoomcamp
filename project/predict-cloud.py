import os
import sys
import json
import requests

host =  "cardio-predict-app-env.eba-virctjis.us-east-1.elasticbeanstalk.com"
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
