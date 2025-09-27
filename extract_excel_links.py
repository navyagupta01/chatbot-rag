# extract_excel_links.py
# Downloads Excel files from links in kmsdocstable.json and extracts their content
import os
import json
import pandas as pd
from pptx import Presentation

EXCEL_DIR = 'downloaded_excels'
os.makedirs(EXCEL_DIR, exist_ok=True)

excel_contents = {}
for filename in os.listdir(EXCEL_DIR):
    filepath = os.path.join(EXCEL_DIR, filename)
    if filename.lower().endswith(('.xlsx', '.xls')):
        try:
            excel_text = []
            xls = pd.ExcelFile(filepath)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                excel_text.append(f"Sheet: {sheet}\n" + df.to_string())
            excel_contents[filename] = '\n\n'.join(excel_text)
            print(f"Extracted Excel: {filename}")
        except Exception as e:
            print(f"Failed to parse Excel {filename}: {e}")
    elif filename.lower().endswith('.pptx'):
        try:
            prs = Presentation(filepath)
            ppt_text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        ppt_text.append(shape.text)
            excel_contents[filename] = '\n'.join(ppt_text)
            print(f"Extracted PPTX: {filename}")
        except Exception as e:
            print(f"Failed to parse PPTX {filename}: {e}")

with open('excel_contents.json', 'w', encoding='utf-8') as f:
    json.dump(excel_contents, f, ensure_ascii=False, indent=2)
print("Saved extracted Excel and PPTX contents to excel_contents.json.")
