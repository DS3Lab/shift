import requests
from schemas.requests.reader import AllReaderConfigsU

endpoint = "http://10.233.0.1:8001/"


def get_nn_result(hash: str, classifier: str):
    if classifier == "Linear":
        req = requests.get(endpoint + "lc_result?job_hash=" + hash)
    else:
        req = requests.get(endpoint + "nn_result?job_hash=" + hash)
    if req.text == "null":
        return None
    else:
        return req.text


def get_reader_size(reader: AllReaderConfigsU):
    req = requests.post(
        endpoint + "query_reader_size_by_json", json={"json_reader": reader}
    )
    print(reader)
    print(req.text)
    if req.text == "null":
        return None
    else:
        return req.text
