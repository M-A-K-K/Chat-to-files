from flask import request, jsonify, render_template_string
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize OpenAI Embeddings
openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Define the directory where Chroma DB will persist data
persist_directory = "./chroma_db"
embeddings_created_flag = "./embeddings_created.flag"

# HTML Template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload and Search</title>
</head>
<body>
    <h1>File Upload and Search</h1>
    
    <h2>Upload Files</h2>
    <form method="post" enctype="multipart/form-data" action="/upload">
        <input type="file" name="files" accept=".pdf,.txt,.docx" multiple>
        <button type="submit">Upload and Process</button>
    </form>
    
    {% if upload_message %}
    <p>{{ upload_message }}</p>
    {% endif %}
    
    <h2>Search</h2>
    <form method="get" action="/search">
        <input type="text" name="query" placeholder="Enter your query">
        <button type="submit">Search</button>
    </form>
    
    {% if search_result %}
    <h3>Search Results for "{{ query }}"</h3>
    <ul>
        {% for result, score in search_result %}
        <li>[SIM={{ score }}] {{ result }}</li>
        {% endfor %}
    </ul>
    {% endif %}
</body>
</html>
"""

def get_chroma_instance():
    """Initialize and return a Chroma instance."""
    return Chroma(
        collection_name="example_collection",
        embedding_function=openai_embeddings,
        persist_directory=persist_directory
    )

def embeddings_created():
    """Check if embeddings have been created."""
    return os.path.exists(embeddings_created_flag)

def set_embeddings_created():
    """Create a flag indicating that embeddings have been created."""
    with open(embeddings_created_flag, 'w') as f:
        f.write("Embeddings created")

def clear_embeddings_created():
    """Clear the flag indicating that embeddings have been created."""
    if os.path.exists(embeddings_created_flag):
        os.remove(embeddings_created_flag)

def upload_files():
    if 'files' not in request.files:
        return jsonify({"message": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No selected files"}), 400

    if not any(file.filename.endswith(('.pdf', '.txt', '.docx')) for file in files):
        return jsonify({"message": "Invalid file format. Only PDFs, text files, and DOCX are allowed."}), 400

    # Initialize Chroma
    db = get_chroma_instance()

    for file in files:
        try:
            # Use tempfile to handle temporary file creation
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file_path = temp_file.name
                file.save(file_path)

            # Process file
            if file.filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file.filename.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file.filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)

            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
            docs = text_splitter.split_documents(documents)
            
            # Add documents to Chroma
            db.add_documents(docs)
        finally:
            # Ensure the temporary file is removed after processing
            if os.path.exists(file_path):
                os.remove(file_path)

    # Set flag indicating embeddings have been created
    set_embeddings_created()

    return jsonify({"message": "Files processed and embeddings created"}), 200

def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"message": "No query provided"}), 400

    if not embeddings_created():
        return jsonify({"message": "No embeddings have been created. Please upload files first."}), 400

    # Initialize Chroma
    db = get_chroma_instance()
    
    # Perform similarity search with scores
    results = db.similarity_search_with_score(
        query,
        k=5  # Fixed number of top results to fetch
    )
    
    # Process results
    search_results = [(res.page_content, score) for res, score in results] if results else [("No results found", 0)]
    
    return render_template_string(html_template, query=query, search_result=search_results, upload_message=None)

def index():
    return render_template_string(html_template, upload_message=None, search_result=None, query=None)
