"""Local database based on SQLite
"""
import sqlite3
from typing import List, Tuple


class LocalDB:
    def __init__(self, database_name: str = "evals.db"):
        self.database_name = database_name
        self.con = sqlite3.connect(database_name)
        self.cur = self.con.cursor()

    def create_table(self, table_name: str, columns: List[str]):
        self.cur.execute(f"CREATE TABLE {table_name}({', '.join(columns)})")

    def fetch_one(self, table_name: str, column_name: str):
        res = self.cur.execute(f"SELECT {column_name} FROM {table_name}")
        return res.fetchone()

    def fetch_all(self, table_name: str, column_name: str):
        res = self.cur.execute(f"SELECT {column_name} from {table_name}")
        return res.fetchall()

    def insert_many(self, table_name: str, data: List[Tuple]):
        sql_string = f"""INSERT INTO {table_name} ("""
        for i, d in enumerate(data):
            if i != 0:
                sql_string += ", "
            sql_string += "?"
        sql_string += ")"
        self.cur.executemany(sql_string)
        return self.con.commit()
