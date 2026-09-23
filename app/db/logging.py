import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "predictions.sqlite3"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            smiles TEXT NOT NULL,
            prediction REAL,
            device TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(smiles: str, prediction: float, device: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, smiles, prediction, device) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), smiles, prediction, device),
    )
    conn.commit()
    conn.close()
