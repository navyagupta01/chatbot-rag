# extract_kmsdb.py
# Extracts all rows from kmsdocstable and saves as JSON
import psycopg2
import json
from datetime import date, datetime

DB_CONFIG = {
    'host': 'localhost',   # instead of '127.0.0.1'
    'port': 5432,
    'database': 'kmsdb',
    'user': 'postgres',
    'password': '<YOUR_PASSWORD>'
}


def fetch_kmsdocstable():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('SELECT * FROM public.kmsdocstable;')
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    data = []
    for row in rows:
        row_dict = {}
        for col, val in zip(columns, row):
            if isinstance(val, (date, datetime)):
                row_dict[col] = val.isoformat()
            else:
                row_dict[col] = val
        data.append(row_dict)
    cur.close()
    conn.close()
    return data

if __name__ == "__main__":
    data = fetch_kmsdocstable()
    print(f"Extracted {len(data)} rows from kmsdocstable.")
    with open('kmsdocstable.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved to kmsdocstable.json.")
