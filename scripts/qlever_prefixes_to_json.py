import json
import sys


def run():
    # read stdin
    data = {}
    for line in sys.stdin:
        assert line.startswith("PREFIX ")
        _, rest = line.split(" ", 1)
        prefix, uri = rest.split(":", 1)
        data[prefix.strip()] = uri.strip()[:-1]

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    run()
