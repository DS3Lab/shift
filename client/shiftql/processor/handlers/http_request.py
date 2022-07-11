import json

import requests


def base_query(method, endpoint, data=None):
    if method == "GET":
        response = requests.get(url=endpoint)
    elif method == "POST":
        response = requests.post(url=endpoint, json=data)
    else:
        raise ValueError("Invalid method")
    if response.status_code < 200 or response.status_code > 300:
        raise ValueError(
            "Invalid {} response: {}".format(response.status_code, response.text)
        )
    return response.json()


def resolve_reader_by_name(name, server_url):
    reader = base_query(
        "POST", server_url + "/query_reader_by_name", data={"name": name}
    )
    if reader is None:
        raise ValueError("Cannot resolve reader {} from {}".format(name, server_url))
    return reader


def simplify_reader_by_json(server_url, reader_json):
    reader_json = json.loads(reader_json)
    return base_query(
        "POST", server_url + "/simplify_reader", data={"json_reader": reader_json}
    )
