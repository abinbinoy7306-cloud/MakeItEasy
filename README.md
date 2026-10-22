---
title: MakeItEasy - PDF AI
sdk: gradio
app_file: app.py
python_version: '3.12'
sdk_version: 6.30.0
---

# MakeItEasy - PDF AI

Upload PDFs, ask questions about their contents, and view the source passages used to answer.

## Evaluation metrics

After asking a question, open **Evaluate this answer**. The panel reports:

- **Hit Rate@5**: whether an expected source appears in the five retrieved chunks.
- **MRR**: the reciprocal rank of the first expected source.
- **Faithfulness**: whether answer claims are supported by retrieved context.
- **Answer Correctness**: whether the answer addresses the question and agrees with an optional reference answer.
- **Citation Accuracy**: whether inline `[Source N]` citations support the claims they follow.

Enter expected source filenames separated by commas to enable Hit Rate@5 and MRR.
The three answer-quality metrics are scored by the configured Ollama model.
