import os
import json
import re
import docx

REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PMSG_Research_Report.docx"))
OUTPUT_CORPUS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "corpus", "corpus.jsonl"))

def is_heading(text, style_name):
    t = text.strip()
    if not t:
        return False
    if t.startswith(("Executive Summary", "Data Sources")):
        return True
    if re.match(r"^[1-8](\.[0-9]+)*\s+", t):
        return True
    if style_name and "Heading" in style_name:
        return True
    return False

def get_section_category(title):
    t = title.lower()
    if "executive" in t:
        return "Executive Summary"
    if t.startswith("1"):
        return "1. Introduction"
    if t.startswith("2"):
        return "2. Data & Methodology"
    if t.startswith("3"):
        return "3. Exploratory Findings"
    if t.startswith("4"):
        return "4. Predictive Modeling"
    if t.startswith("5"):
        return "5. Forecasting"
    if t.startswith("6"):
        return "6. Scenario Analysis"
    if t.startswith("7"):
        return "7. Limitations"
    if t.startswith("8"):
        return "8. Conclusion"
    return "Report Background"

def clean_text(text):
    if not text:
        return ""
    # Replace unicode replacement characters
    t = text.replace("\ufffd", "—")
    # Normalize extra whitespaces
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def export_report_chunks():
    doc = docx.Document(REPORT_PATH)
    subsections = []
    current_title = None
    current_paras = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        s = p.style.name if p.style else ""
        if is_heading(t, s):
            if current_title and current_paras:
                subsections.append((current_title, "\n\n".join(current_paras)))
            current_title = t
            current_paras = []
        else:
            current_paras.append(t)

    if current_title and current_paras:
        subsections.append((current_title, "\n\n".join(current_paras)))

    chunks = []
    for idx, (title, text) in enumerate(subsections):
        c_title = clean_text(title)
        c_text = clean_text(text)
        sec_cat = get_section_category(c_title)
        doc_id = f"report_sec_{idx+1:02d}"
        
        full_chunk = f"Section: {c_title}\n\n{c_text}"
        chunks.append({
            "id": doc_id,
            "source": "report",
            "section": sec_cat,
            "title": c_title,
            "text": full_chunk
        })
    return chunks

def main():
    os.makedirs(os.path.dirname(OUTPUT_CORPUS), exist_ok=True)
    report_chunks = export_report_chunks()
    print(f"Extracted {len(report_chunks)} subsections from {REPORT_PATH}")

    # Load existing litreview records
    existing_records = []
    if os.path.exists(OUTPUT_CORPUS):
        with open(OUTPUT_CORPUS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("source") != "report":
                        existing_records.append(item)

    combined = existing_records + report_chunks
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(combined)} records to {OUTPUT_CORPUS}")

if __name__ == "__main__":
    main()
