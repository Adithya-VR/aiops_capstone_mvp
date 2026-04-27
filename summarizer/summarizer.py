# # summarizer/summarizer.py
# from functools import lru_cache

# @lru_cache(maxsize=1)
# def get_summarizer():
#     """
#     Load flan-t5-base once and cache it.
#     Downloads ~250MB on first run, cached forever after.
#     Runs on CPU — no GPU needed.
#     """
#     from transformers import pipeline
#     print("Loading google/flan-t5-base...")
#     return pipeline(
#         "text2text-generation",
#         model="google/flan-t5-base",
#         device=-1        # -1 = CPU
#     )


# def summarize_window(window_logs, dataset="bgl") -> str:

#     if window_logs is None or len(window_logs) == 0:
#         return "No log data available."

#     # Priority: FATAL → SEVERE → ERROR → anomalous
#     for level_filter in [
#         window_logs["level"] == "FATAL",
#         window_logs["level"] == "SEVERE",
#         window_logs["level"] == "ERROR",
#         window_logs["is_anomaly"] == 1,
#     ]:
#         filtered = window_logs[level_filter]
#         if not filtered.empty:
#             source = filtered
#             break
#     else:
#         source = window_logs

#     # Get top 5 most frequent error lines
#     top_lines = (
#         source
#         .groupby(["level", "content"])
#         .size()
#         .reset_index(name="count")
#         .sort_values("count", ascending=False)
#         .head(5)
#     )

#     if top_lines.empty:
#         return "Insufficient data for analysis."

#     lines_text = "\n".join(
#         f"- {row['content']}"
#         for _, row in top_lines.iterrows()
#     )

#     # Better structured prompt gets better results
#     prompt = (
#         f"Task: Analyze these system error logs and "
#         f"identify the root cause.\n\n"
#         f"Error logs:\n{lines_text}\n\n"
#         f"Root cause in one sentence:"
#     )

#     prompt = prompt[:800]

#     summarizer = get_summarizer()
#     result = summarizer(
#         prompt,
#         max_new_tokens=60,
#         min_new_tokens=10,
#         do_sample=False,
#         num_beams=4,
#         early_stopping=True
#     )
#     return result[0]["generated_text"].strip()
# def classify_window(window_logs) -> dict:
#     """
#     Zero-shot classification of the anomaly type.
#     Returns category and confidence score.
#     """
#     CATEGORIES = [
#         "hardware failure",
#         "network communication error",
#         "memory error",
#         "process crash",
#         "storage failure",
#         "security incident",
#         "normal operation"
#     ]

#     if window_logs is None or len(window_logs) == 0:
#         return {"category": "unknown", "confidence": 0.0}

#     error_logs = window_logs[
#         window_logs["level"].isin(
#             ["FATAL", "SEVERE", "ERROR"]
#         )
#     ]
#     if error_logs.empty:
#         error_logs = window_logs

#     top_content = (
#         error_logs["content"]
#         .value_counts().index[0]
#         if not error_logs.empty
#         else "unknown error"
#     )

#     try:
#         from transformers import pipeline
#         classifier = pipeline(
#             "zero-shot-classification",
#             model="cross-encoder/nli-MiniLM2-L6-H768",
#             device=-1
#         )
#         result = classifier(top_content, CATEGORIES)
#         return {
#             "category":   result["labels"][0],
#             "confidence": round(result["scores"][0], 3)
#         }
#     except Exception:
#         return {"category": "unknown", "confidence": 0.0}