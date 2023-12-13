from datetime import datetime


def load_secrets():
    res = {}
    with open('files/secrets.txt', 'r') as f:
        for line in f.readlines():
            line = line.split('=')
            assert len(line) == 2
            source, apikey = line
            apikey = apikey.replace('\n', '')
            res[source] = apikey
    return res

secrets = load_secrets()

left_wing_sources = [
    "the-huffington-post",
    "msnbc",
    "cnn",
    "the-washington-post",
    "buzzfeed",
    "the-guardian",
]

right_wing_sources = [
    "fox-news",
    "breitbart-news",
    "the-washington-times",
    "the-american-conservative",
    "newsmax",
    "the-federalist",
]
