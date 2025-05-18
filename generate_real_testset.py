import json
import pymongo
from pymongo.errors import ConnectionFailure
from src.config import MONGO_URI, DB_NAME, COLLECTION_NAME
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_FILE = "real_data_rag_testset.jsonl"
NUM_RECORDS_TO_FETCH = 50
SKIP_RECORDS = 900000

def fetch_data_from_mongodb():
    """
    Fetches data from MongoDB, skipping a specified number of records.
    Returns a list of dictionaries, each containing summary, description, and issuetype.
    """
    records = []
    try:
        client = pymongo.MongoClient(MONGO_URI)
        client.admin.command('ping') # Verify connection
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logging.info(f"Connected to MongoDB: {MONGO_URI}, DB: {DB_NAME}, Collection: {COLLECTION_NAME}")

        # Fetch records, skip, limit, and project necessary fields
        # Sorting by _id to ensure consistent skip
        # Temporarily remove projection to inspect full document structure
        cursor = collection.find(
            filter={},
            # projection={"summary": 1, "description": 1, "issuetype": 1, "_id": 0}, # Temporarily removed
            sort=[("_id", pymongo.ASCENDING)]
        ).skip(SKIP_RECORDS).limit(NUM_RECORDS_TO_FETCH)

        for doc_num, document in enumerate(cursor):
            logging.info(f"Fetched raw document {doc_num + 1}: {document}") # Log the raw document
            if doc_num % 50 == 0 and doc_num > 0:
                logging.info(f"Fetched {doc_num} documents...")
            
            fields = document.get("fields", {})
            summary = fields.get("summary")
            description = fields.get("description", "") # Handle missing descriptions
            issuetype_field = fields.get("issuetype", {})
            issuetype = issuetype_field.get("name") if issuetype_field else None

            if summary and issuetype: # Ensure essential fields are present
                records.append({
                    "summary": summary,
                    "description": description,
                    "expected_issuetype": issuetype
                })
            else:
                logging.warning(f"Skipping document due to missing summary or issuetype: {document}")
        
        logging.info(f"Successfully fetched {len(records)} records from MongoDB after skipping {SKIP_RECORDS}.")

    except ConnectionFailure:
        logging.error("Failed to connect to MongoDB. Please check your MONGO_URI and ensure MongoDB is running.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while fetching data: {e}")
        return None
    finally:
        if 'client' in locals() and client:
            client.close()
            logging.info("MongoDB connection closed.")
            
    return records

def save_to_jsonl(data: list, filename: str):
    """
    Saves the list of records to a .jsonl file.
    """
    if not data:
        logging.warning("No data to save.")
        return

    try:
        with open(filename, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(f"Data successfully saved to {filename}")
    except IOError as e:
        logging.error(f"Failed to write to file {filename}: {e}")

if __name__ == "__main__":
    logging.info("Starting script to generate test dataset from MongoDB...")
    mongo_data = fetch_data_from_mongodb()
    if mongo_data:
        save_to_jsonl(mongo_data, OUTPUT_FILE)
    else:
        logging.error("Failed to fetch data from MongoDB. Test dataset not generated.")
    logging.info("Script finished.")

