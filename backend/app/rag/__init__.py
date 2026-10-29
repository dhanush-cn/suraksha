"""RAG (Retrieval-Augmented Generation) over the alert history.

Alert QA: an operator asks "what triggered the last critical event at
Grasberg?" or "which mines had rainfall above 40mm in the past week?"
and gets a natural-language answer grounded in the ``alert_logs``
table and mine metadata.

Layers, in the order a request touches them:

* :mod:`app.rag.client`     -- OpenAI-chat-compatible httpx client
                              (embeddings + streaming chat completions).
                              One code path for OpenAI, Anthropic-compat,
                              Groq, Together, and local Ollama.
* :mod:`app.rag.embeddings` -- Turns an alert into a "source_text" the
                              embedding model sees. Deterministic
                              format so re-embedding produces stable
                              vectors.
* :mod:`app.rag.retrieval`  -- pgvector cosine-distance top-K over
                              ``alert_embeddings``.
* :mod:`app.rag.service`    -- ``ChatService.stream`` = embed the
                              question, retrieve K, build the prompt,
                              stream the LLM response.
* :mod:`app.rag.models`     -- ``AlertEmbedding`` ORM row.

Non-goals for this step:

* Multi-turn conversation memory. Each ``/api/chat`` request is a
  single Q&A; the frontend can keep its own history and pass it in
  the prompt if needed later.
* Tool-use / function-calling. The LLM answers from retrieved
  context; it doesn't call APIs on the user's behalf.
* Streaming SSE reconnect. If the client drops mid-stream the
  response is lost; a partial-answer resume story would need a
  server-side buffer keyed on request id.
"""
