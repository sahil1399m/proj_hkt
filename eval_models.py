# eval_models.py
import json
from crag import (
    _detect_intent,
    embed_query,
    retrieve_from_chroma,
    rerank_chunks,
    classify_confidence,
    build_context,
    search_web,
    _generate_groq,
    _generate_ibm,
)

test_queries = [
    "What are the fees for MBA in Maharashtra?",
    "VJNT scholarship eligibility criteria",
    "CAP round 2 cutoff 2025",
]

results = []

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)

    # Run the full retrieval (same as pipeline does)
    intent    = _detect_intent(q)
    embedding = embed_query(q)
    chunks    = retrieve_from_chroma(embedding, intent)
    ranked, best_logit = rerank_chunks(q, chunks)
    conf_label, conf_score = classify_confidence(best_logit)
    web       = search_web(q)
    context   = build_context(ranked, web)

    print(f"Intent: {intent} | Confidence: {conf_label} ({conf_score})")
    print(f"Chunks retrieved: {len(chunks)} | Web results: {len(web)}")

    # Generate from both models
    groq_ans, groq_model = _generate_groq(q, context, intent, conf_label)
    ibm_ans,  ibm_model  = _generate_ibm(q, context, intent, conf_label)

    print(f"\n--- GROQ ({groq_model}) ---")
    print(groq_ans or "[EMPTY RESPONSE]")

    print(f"\n--- IBM ({ibm_model}) ---")
    print(ibm_ans or "[EMPTY RESPONSE]")

    results.append({
        "query":      q,
        "intent":     intent,
        "confidence": conf_label,
        "groq":       groq_ans,
        "ibm":        ibm_ans,
    })

# Save to file so you can review later
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nResults saved to eval_results.json")