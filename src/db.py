import logging
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME, MONGO_BATCH_SIZE

logger = logging.getLogger(__name__)

def connect_to_db():
    """Connect to MongoDB and return the database object."""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB at %s", MONGO_URI)
        return client[DB_NAME]
    except Exception as e:
        logger.error("Error connecting to MongoDB: %s", e)
        return None


def get_jira_issues(db, initial_skip=0, batch_size=None):
    """Yield batches of Jira issues starting from initial_skip."""
    if db is None:
        logger.error("No database connection available.")
        return

    batch_size = batch_size or MONGO_BATCH_SIZE
    collection = db[COLLECTION_NAME]
    skip = initial_skip
    total = 0
    logger.info("Fetching Jira issues from '%s', starting at offset %d", COLLECTION_NAME, initial_skip)

    while True:
        cursor = collection.find(
            {},
            {"fields.summary": 1, "fields.description": 1, "key": 1, "fields.issuetype.name": 1, "_id": 0}
        ).sort("_id", 1).skip(skip).limit(batch_size)
        raw_batch = list(cursor)
        if not raw_batch:
            logger.info("No more issues to fetch. Total fetched: %d", total)
            break

        issues = []
        for doc in raw_batch:
            fld = doc.get("fields", {})
            issues.append({
                "key": doc.get("key"),
                "summary": fld.get("summary", ""),
                "description": fld.get("description", ""),
                "issuetype": fld.get("issuetype", {}).get("name", "")
            })

        batch_count = len(issues)
        total += batch_count
        logger.info("Fetched batch: %d issues (offset now %d)", batch_count, skip)
        yield issues

        if batch_count < batch_size:
            logger.info("Last batch smaller than batch_size; ending iteration.")
            break

        skip += batch_size
