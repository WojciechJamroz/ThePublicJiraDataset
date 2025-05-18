import logging
import numpy as np
from typing import Tuple, List, Dict, Any
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

def load_embedding_model(device: str = 'cpu') -> Tuple[SentenceTransformer, int]:
    """Load and return a SentenceTransformer model on the specified device."""
    logger.info("Loading embedding model: %s on %s", EMBEDDING_MODEL, device)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    dim = model.get_sentence_embedding_dimension()
    logger.info("Loaded embedding model; dimension: %d", dim)
    return model, dim


def generate_embeddings(
    issues: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """Return a list of dicts with issue_key, text, embedding (np.ndarray), issuetype."""
    raw_texts = [f"{i.get('summary','')} {i.get('description','')}" for i in issues]
    keys = [i.get('key') for i in issues]
    issuetypes = [i.get('issuetype','') for i in issues]
    try:
        embeddings = model.encode(
            raw_texts, batch_size=256, convert_to_tensor=False, normalize_embeddings=True
        )
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        return []

    data = []
    for key, text, emb, itype in zip(keys, raw_texts, embeddings, issuetypes):
        arr = np.array(emb, dtype='float32')
        data.append({
            'issue_key': key,
            'text': text,
            'embedding': arr,
            'issuetype': itype
        })

    logger.info("Generated embeddings for %d issues.", len(data))
    return data
