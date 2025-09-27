import psycopg2
from psycopg2 import OperationalError

try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="kmsdb",
        user="postgres",
    password="<YOUR_PASSWORD>"
    )
    print("Connection successful!")
    conn.close()
except OperationalError as e:
    print("Connection failed:")
    print(e)
except Exception as e:
    print("An unexpected error occurred:")
    print(e)
