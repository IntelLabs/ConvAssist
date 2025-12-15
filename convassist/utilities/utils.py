from charset_normalizer import from_path


def smart_readlines(path):
    result = from_path(path).best()
    enc = result.encoding if result else "utf-8"
    with open(path, encoding=enc) as f:
        return f.readlines()
