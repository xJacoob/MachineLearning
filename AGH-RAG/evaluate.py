from rag import search, build_context, chat_answer
from pathlib import Path
import json

K_TOP = 5
RUN_NAME = "c1500_k5_prompt_v2"
QUESTION_PATH = str(Path(__file__).parent / "eval" / "questions.json")
RESULTS_DIR = Path(__file__).parent / "eval" / "results"

with open(QUESTION_PATH, 'r', encoding='utf-8') as f:
    questions = json.load(f)

records = []

for q in questions:
    result_list = search(q['question'], K_TOP)

    if q['type'] == 'out_of_scope':
        hit = None
    else:
        hit = any(r['source'] == q['source'] and r['page'] in q['page'] for r in result_list)

    context = build_context(result_list)
    answer = chat_answer(q['question'], context)

    sources = [
        {"source": r['source'], "page": r['page'], "distance": r['distance']} for r in result_list
    ]

    records.append({
        "question": q['question'],
        "type": q['type'],
        "expected": q['expected'],
        "answer": answer,
        "hit": hit,
        "sources": sources,
        "grade": None,
    })

hits = sum(1 for rec in records if rec['hit'] is True)
total = sum(1 for rec in records if rec['hit'] is not None)

output = {
    "run_name": RUN_NAME,
    "k_top": K_TOP,
    "hits": f"{hits}/{total}",
    "results": records,
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / f"{RUN_NAME}.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)