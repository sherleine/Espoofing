# app/monitor/session_store.py
"""
STAND-IN for running this prototype on its own.

The real application already has app/monitor/session_store.py - do NOT copy
this file over it. view.py only needs mark_verified(email).
"""

_verified = set()


def mark_verified(email):
    _verified.add(email)


def is_verified(email):
    return email in _verified
