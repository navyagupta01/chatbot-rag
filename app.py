import streamlit as st
import json
import re

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
st.title("Agentic RAG Chatbot")

# Load data
with open("kmsdocstable.json", "r", encoding="utf-8") as f:
    table_data = json.load(f)
with open("excel_contents.json", "r", encoding="utf-8") as f:
    excel_contents = json.load(f)
def answer_question(question, table_data, excel_contents):
    q = question.lower().strip()
    keywords = [w for w in re.findall(r'\w+', q) if len(w) > 2]
    # Search in extracted Excel/PPTX content first
    excel_results = []
    for fname, content in excel_contents.items():
        content_lc = content.lower()
        if all(kw in content_lc for kw in keywords):
            excel_results.append((fname, content))
    if not excel_results:
        for fname, content in excel_contents.items():
            content_lc = content.lower()
            if any(kw in content_lc for kw in keywords):
                excel_results.append((fname, content))
    if excel_results:
        import pandas as pd
        for fname, content in excel_results:
            link = None
            for row in table_data:
                docname = row.get('docname','').replace(' ','_').lower()
                if fname.lower().startswith(docname) or fname.lower().startswith(row.get('docname','').lower()):
                    link = row.get('filesource','')
                    break
            # Try to parse the content as a DataFrame summary
            try:
                # Try to extract the most relevant table row(s)
                # Assume content is from pandas .to_string(), so split by 'Sheet:'
                sheets = content.split('Sheet:')
                best_rows = []
                for sheet in sheets:
                    lines = sheet.split('\n')
                    # Find header
                    header = None
                    for i, line in enumerate(lines):
                        if 'Unnamed:' not in line and line.strip():
                            header = i
                            break
                    # Find rows with all keywords
                    for line in lines:
                        if all(kw in line.lower() for kw in keywords):
                            best_rows.append(line)
                st.markdown(f"**Relevant content from `{fname}`:**")
                if best_rows:
                    for row in best_rows:
                        st.write(row)
                else:
                    st.info("No direct table row match, showing file snippet:")
                    st.code(content[:1000].replace('\n',' ') + ('...' if len(content) > 1000 else ''), language=None)
            except Exception as e:
                st.code(content[:1000].replace('\n',' ') + ('...' if len(content) > 1000 else ''), language=None)
            if link:
                st.markdown(f"[View original file]({link})")
        return
    # Fallback: search in metadata as before
    results = []
    for row in table_data:
        row_text = ' '.join(str(v).lower() for v in row.values())
        if all(kw in row_text for kw in keywords):
            results.append(row)
    if not results:
        for row in table_data:
            row_text = ' '.join(str(v).lower() for v in row.values())
            if any(kw in row_text for kw in keywords):
                results.append(row)
    if results:
        for row in results:
            st.markdown(f"**Docname:** {row.get('docname','')}  ")
            st.markdown(f"**Date:** {row.get('date','')}  ")
            st.markdown(f"**Description:** {row.get('discription','')}  ")
            st.markdown(f"[View original file]({row.get('filesource','')})")
        return
    st.warning("No relevant information found.")

# Streamlit UI
question = st.text_input("Ask a question about your data:")
if question:
    answer_question(question, table_data, excel_contents)
