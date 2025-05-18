
### Install Python Virtual Environment and Run JupyterLab Notebook

```bash
# Navigate to the correct directory (here)
cd ./ThePublicJiraDataset/1.\ DataDownload/

# Install, create, and load Python virtual environment
pip install virtualenv  # Ensure the Python venv package is installed
virtualenv .venv  # Create a virtual environment
. .venv/bin/activate  # Activate (enter) the virtual environment
pip install --upgrade pip  # Upgrade pip if necessary

# Install all pip packages into the virutal environment
pip install -r requirements-manual.txt

# Start JupyterLab
./.venv/bin/jupyter lab  # Reference this specific "jupyter" for virtualisation
```
