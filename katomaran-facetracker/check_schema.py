# import sqlite3

# conn = sqlite3.connect("data/visitor_log.db")
# cursor = conn.cursor()

# cursor.execute("PRAGMA table_info(visits);")
# columns = cursor.fetchall()

# for col in columns:
#     print(col)

# conn.close()

import sqlite3
import pandas as pd

conn = sqlite3.connect("data/visitor_log.db")
df = pd.read_sql_query("SELECT * FROM visits", conn)

print(df.to_string(index=False))  # Print all rows without index
conn.close()