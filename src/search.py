import numpy as np
import logging
from .config import TASK_DESCRIPTION

logger = logging.getLogger(__name__)

def search_faiss(query, model, index, metadata, top_n=3):
    """Return top_n similar issues for query_text."""
    instr = f"Instruct: {TASK_DESCRIPTION}\nQuery: {query}"

    emb = model.encode(instr, normalize_embeddings=True, convert_to_tensor=False)
    q = np.array(emb, dtype='float32').reshape(1, -1)
    dists, idxs = index.search(q, top_n)

    results = []
    for sim, idx in zip(dists[0], idxs[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx]
            results.append({
                'issue_key': item['issue_key'],
                'similarity': float(sim),
                'text': item['text'],
                'issuetype': item.get('issuetype','')
            })
        else:
            logger.warning("Invalid FAISS index %d", idx)
    return results
