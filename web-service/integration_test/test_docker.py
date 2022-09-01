import json
from pathlib import Path

import requests
from deepdiff import DeepDiff

test_directory = Path(__file__).parent

with open(test_directory / 'house_data.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)

url = 'http://127.0.0.1:9696/predict'
actual_response = requests.post(url, json=event)
print(actual_response.json())

expected_response = {
    'price': float(171629)
}
print(expected_response)

diff = DeepDiff(actual_response.json(), expected_response, significant_digits=0)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff