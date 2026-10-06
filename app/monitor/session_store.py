_verified = set()


def mark_verified(email):
    _verified.add(email)


def is_verified(email):
    return email in _verified
