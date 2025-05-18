import os
import json
import logging
import faiss
from .config import FAISS_INDEX_FILE, INDEX_METADATA_FILE, INDEX_PROGRESS_FILE

logger = logging.getLogger(__name__)

def init_faiss_index(dim, use_gpu=False):
    """Initialize CPU or GPU FAISS index for inner-product similarity."""
    cpu_index = faiss.IndexFlatIP(dim)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("Initialized FAISS index on GPU.")
            return gpu_index
        except Exception as e:
            logger.error("GPU FAISS init failed, using CPU: %s", e)
            return cpu_index
    logger.info("Initialized FAISS index on CPU.")
    return cpu_index


def save_index_and_metadata(index, metadata):
    """Persist FAISS index and metadata to disk."""
    # write FAISS (ensure CPU index)
    if hasattr(index, 'is_GpuIndex') and index.is_GpuIndex():
        index_to_save = faiss.index_gpu_to_cpu(index)
    else:
        index_to_save = index
    faiss.write_index(index_to_save, FAISS_INDEX_FILE)
    # metadata
    with open(INDEX_METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    logger.info("Saved FAISS index and %d metadata entries.", len(metadata))


def load_index_and_metadata(use_gpu=False):
    """Load index and metadata from disk; return (index or None, metadata list)."""
    if not os.path.exists(INDEX_PROGRESS_FILE):
        logger.info("No progress file; skipping load.")
        return None, []

    # load progress
    with open(INDEX_PROGRESS_FILE, 'r') as f:
        prog = json.load(f)
    num = prog.get('num_indexed', 0)
    if num <= 0:
        return None, []

    # load metadata
    if not os.path.exists(INDEX_METADATA_FILE):
        logger.warning("Metadata file missing; need reindex.")
        return None, []
    with open(INDEX_METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    # load index
    if not os.path.exists(FAISS_INDEX_FILE):
        logger.warning("FAISS file missing; need reindex.")
        return None, []
    idx_cpu = faiss.read_index(FAISS_INDEX_FILE)
    if idx_cpu.ntotal != len(metadata) or idx_cpu.ntotal != num:
        logger.warning("Index/metadata/progress mismatch; need reindex.")
        return None, []

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx_cpu)
            logger.info("Loaded FAISS index on GPU with %d entries.", idx.ntotal)
            return idx, metadata
        except Exception as e:
            logger.error("GPU load failed; using CPU: %s", e)
    logger.info("Loaded FAISS index on CPU with %d entries.", idx_cpu.ntotal)
    return idx_cpu, metadata


def update_progress(num_indexed):
    """Write progress JSON file."""
    with open(INDEX_PROGRESS_FILE, 'w') as f:
        json.dump({'num_indexed': num_indexed}, f)
    logger.debug("Updated progress: %d items.", num_indexed)
