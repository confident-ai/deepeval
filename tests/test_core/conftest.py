try:
    import sys, pysqlite3 as sqlite3  # type: ignore
    sys.modules["sqlite3"] = sqlite3
    sys.modules["sqlite3.dbapi2"] = sqlite3.dbapi2
except Exception:
    pass
