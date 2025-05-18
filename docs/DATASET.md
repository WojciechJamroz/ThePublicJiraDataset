# Building the MongoDB Dataset

This guide describes how to assemble and load a Jira dataset into MongoDB, using Apache as an example.

## 1. Prerequisites

- **MongoDB** (v4+)
- **Python 3.8+**, with a virtual environment
- Internet access to Jira REST APIs

## 2. Define Data Sources

1. Open `dataset/0. DataDefinition/jira_data_sources.json`.
2. Ensure the entry for **Apache** looks like:

   ```json
   {
     "Apache": {
       "name": "Apache",
       "jira_url": "https://issues.apache.org/jira"
     }
   }
   ```
3. You can customize other projects here as needed.

## 3. Install Notebook Requirements

```powershell
cd dataset/1. DataDownload
pip install -r requirements-manual.txt
```  
This installs `aiohttp`, `requests`, `pymongo`, etc., for the download notebook.

## 4. Run the Download Notebook

1. Launch Jupyter in that folder:
   ```powershell
   jupyter notebook
   ```
2. Open **DownloadData.ipynb**.
3. **Cell 1 (Imports)**: Verify all imports succeed.
4. **Cell 2 (Load Program Data)**: Verify `jira_data_sources` and `db` connect properly.
5. **Helper Functions** and **Issue Type/Link/Field** cells will fetch metadata into JSON files under `dataset/1. DataDownload`.
6. **Download Commands** cell: run:
   ```python
   await download_and_write_data_mongo(jira_data_sources['Apache'])
   ```
   This will stream all Apache issues into the **`Apache`** collection in your MongoDB `JiraRepos` database.

> Note: For large projects like Apache, the download may take several hours. You can adjust `iteration_max` and `num_desired_results`.

## 5. Verify in MongoDB

In a new PowerShell:
```powershell
mongo
> use JiraRepos
> db.getCollectionNames()
["Apache", ...]
> db.Apache.count()
# Should match the total issue count reported by the notebook.
```  

## 6. Next Steps

- Download comments (optional) via the **Download Jira Issue Comments** section in the notebook.
- Use the CLI (`python main.py index`) to build your FAISS index from the newly populated collection.

---
&nbsp;  
For more analysis, see the `dataset/2. OverviewAnalysis/README.md` and corresponding notebook.  
