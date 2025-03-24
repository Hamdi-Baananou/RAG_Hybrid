# PowerShell script to create the graph_rag_app directory structure

# Create the root directory and navigate to it
New-Item -ItemType Directory -Path "graph_rag_app" -Force
Set-Location -Path "graph_rag_app"

# Create main files
New-Item -ItemType File -Path "app.py" -Force
New-Item -ItemType File -Path "requirements.txt" -Force
New-Item -ItemType File -Path "README.md" -Force
New-Item -ItemType File -Path ".env.example" -Force

# Create utils directory and its files
New-Item -ItemType Directory -Path "utils" -Force
New-Item -ItemType File -Path "utils/__init__.py" -Force
New-Item -ItemType File -Path "utils/graph_db.py" -Force
New-Item -ItemType File -Path "utils/pdf_processor.py" -Force
New-Item -ItemType File -Path "utils/query_engine.py" -Force
New-Item -ItemType File -Path "utils/embeddings.py" -Force
New-Item -ItemType File -Path "utils/logging_config.py" -Force

# Create prompts directory and its files
New-Item -ItemType Directory -Path "prompts" -Force
New-Item -ItemType File -Path "prompts/__init__.py" -Force
New-Item -ItemType File -Path "prompts/extraction_prompts.py" -Force

# Create components directory and its files
New-Item -ItemType Directory -Path "components" -Force
New-Item -ItemType File -Path "components/__init__.py" -Force
New-Item -ItemType File -Path "components/sidebar.py" -Force
New-Item -ItemType File -Path "components/results.py" -Force

# Output success message
Write-Host "Graph RAG Application directory structure created successfully at: $(Get-Location)\graph_rag_app"

# Return to the original directory
Set-Location -Path ".."