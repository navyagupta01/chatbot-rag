# rag_pipeline.py
# RAG pipeline to answer questions using parsed data
import json
import re

def answer_question(question, table_data, excel_contents):
    q = question.lower().strip()
    keywords = [w for w in re.findall(r'\w+', q) if len(w) > 2]

    # Search in extracted Excel/PPTX content first
    excel_results = []
    for fname, content in excel_contents.items():
        content_lc = content.lower()
        if all(kw in content_lc for kw in keywords):
            excel_results.append((fname, content))
    # If no full match, try partial match
    if not excel_results:
        for fname, content in excel_contents.items():
            content_lc = content.lower()
            if any(kw in content_lc for kw in keywords):
                excel_results.append((fname, content))
    if excel_results:
        answers = []
        for fname, content in excel_results:
            # Try to find the original link from table_data
            link = None
            for row in table_data:
                docname = row.get('docname','').replace(' ','_').lower()
                if fname.lower().startswith(docname) or fname.lower().startswith(row.get('docname','').lower()):
                    link = row.get('filesource','')
                    break
            snippet = content[:500].replace('\n',' ') + ('...' if len(content) > 500 else '')
            answers.append(f"Relevant content from {fname}: {snippet}\nYou can view the original file here: {link if link else 'N/A'}")
        return '\n\n'.join(answers)

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
        formatted = []
        for row in results:
            formatted.append(
                f"Docname: {row.get('docname', '')}\nDate: {row.get('date', '')}\nDescription: {row.get('discription', '')}\nFor more details, see the document here: {row.get('filesource', '')}\n"
            )
        return '\n'.join(formatted)
    else:
        return "No relevant information found."

if __name__ == "__main__":
    with open("kmsdocstable.json", "r", encoding="utf-8") as f:
        table_data = json.load(f)
    with open("excel_contents.json", "r", encoding="utf-8") as f:
        excel_contents = json.load(f)
    user_question = input("Ask a question: ")
    answer = answer_question(user_question, table_data, excel_contents)
    print("\nAnswer:\n" + answer)
