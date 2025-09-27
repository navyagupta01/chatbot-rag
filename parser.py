# parser.py
# Reads kmsdb_dump.dump and extracts all links and relevant data
import re

def parse_dump(file_path):
    with open(file_path, 'rb') as f:
        raw = f.read()
    data = raw.decode('utf-8', errors='ignore')

    # Extract table rows from COPY section
    table_data = []
    columns = []
    copy_match = re.search(r"COPY public.kmsdocstable \(([^)]+)\) FROM stdin;(.*?)\\.\n", data, re.DOTALL)
    if copy_match:
        columns = [col.strip() for col in copy_match.group(1).split(",")]
        rows = copy_match.group(2).strip().split("\n")
        for row in rows:
            if row and not row.startswith("\\."):
                fields = row.split("\t")
                if len(fields) == len(columns):
                    table_data.append(dict(zip(columns, fields)))

    # Extract URLs (http/https links) from all data
    links = re.findall(r'https?://\S+', data)
    return links, data, table_data

if __name__ == "__main__":
    links, full_data, table_data = parse_dump("kmsdb_dump.dump")
    print(f"Extracted {len(links)} links:")
    for link in links:
        print(link)
    print(f"\nExtracted {len(table_data)} rows from kmsdocstable:")
    if table_data:
        print("Columns:", list(table_data[0].keys()))
        for row in table_data[:5]:
            print(row)
