"""
SQLite Database for persistent storage.
"""
import sqlite3


class Database:
    def __init__(self, db_path="trading_system.db"):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, symbol TEXT, strategy TEXT, signal TEXT, price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, strategy TEXT, entry_date TEXT, entry_price REAL,
                exit_date TEXT, exit_price REAL, quantity INTEGER,
                pnl REAL, pnl_pct REAL, charges REAL, exit_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, cash REAL, positions_value REAL, portfolio_value REAL,
                open_positions INTEGER, total_pnl REAL, return_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        """)
        conn.commit(); conn.close()

    def _conn(self): return sqlite3.connect(self.db_path)

    def log_signal(self, date, symbol, strategy, signal, price):
        c = self._conn()
        c.execute("INSERT INTO signals (date,symbol,strategy,signal,price) VALUES (?,?,?,?,?)",
                  (date,symbol,strategy,signal,price))
        c.commit(); c.close()

    def get_signals(self, date=None, symbol=None):
        c = self._conn(); c.row_factory = sqlite3.Row
        q = "SELECT * FROM signals WHERE 1=1"; p = []
        if date: q += " AND date=?"; p.append(date)
        if symbol: q += " AND symbol=?"; p.append(symbol)
        rows = c.execute(q+" ORDER BY created_at DESC LIMIT 100", p).fetchall()
        c.close(); return [dict(r) for r in rows]
