# # from flask import Flask, request, jsonify, render_template_string
# # import os
# # import tempfile
# # from langchain.document_loaders import PyPDFLoader, TextLoader
# # from langchain_community.document_loaders import Docx2txtLoader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain_community.embeddings import OpenAIEmbeddings
# # from langchain_chroma import Chroma

# # app = Flask(__name__)

# # # Initialize OpenAI Embeddings
# # openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# # # Define the directory where Chroma DB will persist data
# # persist_directory = "./chroma_db"
# # embeddings_created_flag = "./embeddings_created.flag"

# # # HTML Template
# # html_template = """
# # <!DOCTYPE html>
# # <html lang="en">
# # <head>
# #     <meta charset="UTF-8">
# #     <meta name="viewport" content="width=device-width, initial-scale=1.0">
# #     <title>File Upload and Search</title>
# # </head>
# # <body>
# #     <h1>File Upload and Search</h1>
    
# #     <h2>Upload Files</h2>
# #     <form method="post" enctype="multipart/form-data" action="/upload">
# #         <input type="file" name="files" accept=".pdf,.txt,.docx" multiple>
# #         <button type="submit">Upload and Process</button>
# #     </form>
    
# #     {% if upload_message %}
# #     <p>{{ upload_message }}</p>
# #     {% endif %}
    
# #     <h2>Search</h2>
# #     <form method="get" action="/search">
# #         <input type="text" name="query" placeholder="Enter your query">
# #         <button type="submit">Search</button>
# #     </form>
    
# #     {% if search_result %}
# #     <h3>Search Results for "{{ query }}"</h3>
# #     <ul>
# #         {% for result, score in search_result %}
# #         <li>[SIM={{ score }}] {{ result }}</li>
# #         {% endfor %}
# #     </ul>
# #     {% endif %}
# # </body>
# # </html>
# # """

# # def get_chroma_instance():
# #     """Initialize and return a Chroma instance."""
# #     return Chroma(
# #         collection_name="example_collection",
# #         embedding_function=openai_embeddings,
# #         persist_directory=persist_directory
# #     )

# # def embeddings_created():
# #     """Check if embeddings have been created."""
# #     return os.path.exists(embeddings_created_flag)

# # def set_embeddings_created():
# #     """Create a flag indicating that embeddings have been created."""
# #     with open(embeddings_created_flag, 'w') as f:
# #         f.write("Embeddings created")

# # def clear_embeddings_created():
# #     """Clear the flag indicating that embeddings have been created."""
# #     if os.path.exists(embeddings_created_flag):
# #         os.remove(embeddings_created_flag)

# # @app.route('/upload', methods=['POST'])
# # def upload_files():
# #     if 'files' not in request.files:
# #         return jsonify({"message": "No files part"}), 400
    
# #     files = request.files.getlist('files')
# #     if not files:
# #         return jsonify({"message": "No selected files"}), 400

# #     if not any(file.filename.endswith(('.pdf', '.txt', '.docx')) for file in files):
# #         return jsonify({"message": "Invalid file format. Only PDFs, text files, and DOCX are allowed."}), 400

# #     # Initialize Chroma
# #     db = get_chroma_instance()

# #     for file in files:
# #         try:
# #             # Use tempfile to handle temporary file creation
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
# #                 file_path = temp_file.name
# #                 file.save(file_path)

# #             # Process file
# #             if file.filename.endswith('.pdf'):
# #                 loader = PyPDFLoader(file_path)
# #             elif file.filename.endswith('.txt'):
# #                 loader = TextLoader(file_path)
# #             elif file.filename.endswith('.docx'):
# #                 loader = Docx2txtLoader(file_path)

# #             documents = loader.load()
# #             text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# #             docs = text_splitter.split_documents(documents)
            
# #             # Add documents to Chroma
# #             db.add_documents(docs)
# #         finally:
# #             # Ensure the temporary file is removed after processing
# #             if os.path.exists(file_path):
# #                 os.remove(file_path)

# #     # Set flag indicating embeddings have been created
# #     set_embeddings_created()

# #     return jsonify({"message": "Files processed and embeddings created"}), 200

# # @app.route('/search', methods=['GET'])
# # def search():
# #     query = request.args.get('query')
# #     if not query:
# #         return jsonify({"message": "No query provided"}), 400

# #     if not embeddings_created():
# #         return jsonify({"message": "No embeddings have been created. Please upload files first."}), 400

# #     # Initialize Chroma
# #     db = get_chroma_instance()
    
# #     # Perform similarity search with scores
# #     results = db.similarity_search_with_score(
# #         query,
# #         k=5  # Fixed number of top results to fetch
# #     )
    
# #     # Process results
# #     search_results = [(res.page_content, score) for res, score in results] if results else [("No results found", 0)]
    
# #     return render_template_string(html_template, query=query, search_result=search_results, upload_message=None)

# # @app.route('/', methods=['GET'])
# # def index():
# #     return render_template_string(html_template, upload_message=None, search_result=None, query=None)

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)

# # Import routes
# from routes import upload_files, search, index

# # Register routes with Flask
# app.add_url_rule('/upload', view_func=upload_files, methods=['POST'])
# app.add_url_rule('/search', view_func=search, methods=['GET'])
# app.add_url_rule('/', view_func=index, methods=['GET'])

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, jsonify, render_template_string
# import os
# import tempfile
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize OpenAI Embeddings
# openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# # Define the directory where Chroma DB will persist data
# persist_directory = "./chroma_db"
# embeddings_created_flag = "./embeddings_created.flag"

# # HTML Template
# html_template = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>File Upload and Search</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             margin: 20px;
#         }
#         h1 {
#             color: #333;
#         }
#         form {
#             margin-bottom: 20px;
#         }
#         .result-item {
#             border: 1px solid #ddd;
#             padding: 10px;
#             margin-bottom: 10px;
#             border-radius: 5px;
#         }
#         .result-item p {
#             margin: 0;
#             padding: 0;
#         }
#         .no-results {
#             color: #d9534f;
#         }
#         .message {
#             color: #5bc0de;
#         }
#     </style>
# </head>
# <body>
#     <h1>File Upload and Search</h1>
    
#     <h2>Upload Files</h2>
#     <form method="post" enctype="multipart/form-data" action="/upload">
#         <input type="file" name="files" accept=".pdf,.txt,.docx" multiple>
#         <button type="submit">Upload and Process</button>
#     </form>
    
#     {% if upload_message %}
#     <p class="message">{{ upload_message }}</p>
#     {% endif %}
    
#     <h2>Search</h2>
#     <form method="get" action="/search">
#         <input type="text" name="query" placeholder="Enter your query">
#         <button type="submit">Search</button>
#     </form>
    
#     {% if no_db_message %}
#     <p class="no-results">{{ no_db_message }}</p>
#     {% elif search_result %}
#     <h3>Search Results for "{{ query }}"</h3>
#     <div>
#         {% for result in search_result %}
#         <div class="result-item">
#             <p><strong>Result:</strong> {{ result['text'] }}</p>
#             <p><strong>Similarity Score:</strong> {{ result['score'] }}</p>
#         </div>
#         {% endfor %}
#     </div>
#     {% endif %}
# </body>
# </html>
# """

# def get_chroma_instance():
#     """Initialize and return a Chroma instance."""
#     return Chroma(
#         collection_name="example_collection",
#         embedding_function=openai_embeddings,
#         persist_directory=persist_directory
#     )

# def embeddings_created():
#     """Check if embeddings have been created."""
#     return os.path.exists(embeddings_created_flag)

# def set_embeddings_created():
#     """Create a flag indicating that embeddings have been created."""
#     with open(embeddings_created_flag, 'w') as f:
#         f.write("Embeddings created")

# def clear_embeddings_created():
#     """Clear the flag indicating that embeddings have been created."""
#     if os.path.exists(embeddings_created_flag):
#         os.remove(embeddings_created_flag)

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     if 'files' not in request.files:
#         return jsonify({"message": "No files part"}), 400
    
#     files = request.files.getlist('files')
#     if not files:
#         return jsonify({"message": "No selected files"}), 400

#     if not any(file.filename.endswith(('.pdf', '.txt', '.docx')) for file in files):
#         return jsonify({"message": "Invalid file format. Only PDFs, text files, and DOCX are allowed."}), 400

#     # Load the existing Chroma instance if it exists
#     db = get_chroma_instance()
#     all_docs = []  # Initialize an empty list to collect documents

#     for file in files:
#         try:
#             # Use tempfile to handle temporary file creation
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
#                 file_path = temp_file.name
#                 file.save(file_path)

#             # Process file
#             if file.filename.endswith('.pdf'):
#                 loader = PyPDFLoader(file_path)
#             elif file.filename.endswith('.txt'):
#                 loader = TextLoader(file_path)
#             elif file.filename.endswith('.docx'):
#                 loader = Docx2txtLoader(file_path)

#             documents = loader.load()
#             text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             docs = text_splitter.split_documents(documents)
            
#             # Collect all documents to embed
#             all_docs.extend(docs)
#         finally:
#             # Ensure the temporary file is removed after processing
#             if os.path.exists(file_path):
#                 os.remove(file_path)

#     # Add new documents to the Chroma instance
#     if all_docs:
#         # Add documents to the Chroma database
#         db.add_documents(all_docs)  # Ensure this method exists or use equivalent

#     # Set flag indicating embeddings have been created
#     set_embeddings_created()

#     return jsonify({"message": "Files processed and added to the database."}), 200

# @app.route('/search', methods=['GET'])
# def search():
#     query = request.args.get('query', '')
#     k = int(request.args.get('k', 5))  # Default value of k is 5 if not provided

#     if not query:
#         return render_template_string(html_template, upload_message=None, search_result=None, query=query, no_db_message="No query provided."), 400

#     try:
#         # Load the existing Chroma instance if it exists
#         db = get_chroma_instance()

#         if not embeddings_created() or db is None:
#             no_db_message = "No vector database is present. Please upload files to create the database."
#             return render_template_string(html_template, search_result=None, query=query, no_db_message=no_db_message)

#         # Create a RetrievalQA chain
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#         qa_chain = RetrievalQA.from_chain_type(
#                   llm, retriever=db.as_retriever())
        
#         # Include the user's query in the context
#         search_results = qa_chain({"query": query})

#         # Check if results are present
#         if 'result' not in search_results:
#             no_db_message = "No results found for the query."
#             return render_template_string(html_template, search_result=None, query=query, no_db_message=no_db_message)
        
#         # Format results for display
#         formatted_results = [
#             {"text": search_results["result"], "score": "N/A"}  # Adjust as needed
#         ]

#         return render_template_string(html_template, search_result=formatted_results, query=query, upload_message=None)

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")  # Log the error
#         return render_template_string(html_template, search_result=None, query=query, no_db_message="An unexpected error occurred during search."), 500

# @app.route('/', methods=['GET'])
# def index():
#     return render_template_string(html_template, upload_message=None, search_result=None, query=None)

# if __name__ == '__main__':
#     app.run(debug=True)


# above is working code with chroma db 


# import os
# from pinecone import Pinecone

# # Set your API key
# api_key = '3e336c49-39a9-44e2-8700-bfbfe191f0b3'

# # Initialize Pinecone client
# try:
#     pc = Pinecone(api_key=api_key)
#     print("Pinecone client initialized successfully.")
# except Exception as e:
#     print(f"Failed to initialize Pinecone client: {e}")

# # Define your index name
# index_name = "upload"

# # Check if the index exists
# try:
#     index_list = pc.list_indexes()
#     print(index_list)
#     if index_list[0]["name"] == "upload":
#         print(f"Index {index_name} exists.")
#         index_description = pc.describe_index(index_name)
#         print(index_description)
#     else:
#         print(f"Index {index_name} not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")

# print("we")
# print(pc.describe_index(index_name))
# from flask import Flask, request, render_template_string
# import os
# import tempfile
# import time
# from uuid import uuid4

# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv

# # Load environment variables from the .env file
# load_dotenv()
# app = Flask(__name__)

# # Set your API keys here
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = "3e336c49-39a9-44e2-8700-bfbfe191f0b3"  # Replace with your Pinecone API key

# # Initialize Pinecone client
# pc = Pinecone(api_key=pinecone_api_key)

# # Create a Pinecone index if it doesn't exist
# index_name = "pdf-file-index"  # Change if needed
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if index_name not in existing_indexes:
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     while not pc.describe_index(index_name).status["ready"]:
#         time.sleep(1)

# index = pc.Index(index_name)

# # Initialize embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # Initialize vector store
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# # HTML template for file upload
# html_template = '''
# <!doctype html>
# <html lang="en">
# <head>
#     <meta charset="utf-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <title>Upload PDF File</title>
# </head>
# <body>
#     <h1>Upload PDF File</h1>
#     <form action="/upload" method="post" enctype="multipart/form-data">
#         <input type="file" name="file" accept=".pdf">
#         <input type="submit" value="Upload">
#     </form>
#     {% if message %}
#         <p>{{ message }}</p>
#     {% endif %}
# </body>
# </html>
# '''

# @app.route('/')
# def index():
#     return render_template_string(html_template)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return render_template_string(html_template, message="No file part")
    
#     file = request.files['file']
#     if file.filename == '':
#         return render_template_string(html_template, message="No selected file")
    
#     if file and file.filename.endswith('.pdf'):
#         # Use tempfile to create a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             file_path = temp_file.name
#             file.save(file_path)
        
#         # Use PyPDFLoader to read the PDF file
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
        
#         # Create unique IDs for the documents
#         uuids = [str(uuid4()) for _ in range(len(documents))]
        
#         # Add the documents to the vector store
#         vector_store.add_documents(documents=documents, ids=uuids)
        
#         # Clean up
#         os.remove(file_path)
        
#         return render_template_string(html_template, message="PDF processed and documents added.")
#     else:
#         return render_template_string(html_template, message="Invalid file type. Please upload a .pdf file.")

# if __name__ == '__main__':
#     app.run(debug=True)


#working code for upload functionlatity 

# from flask import Flask, request, render_template_string
# import os
# import tempfile
# import time
# from uuid import uuid4

# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv

# # Load environment variables from the .env file
# load_dotenv()
# app = Flask(__name__)

# # Set your API keys here
# openai_api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = "3e336c49-39a9-44e2-8700-bfbfe191f0b3"  # Replace with your Pinecone API key

# # Initialize Pinecone client
# pc = Pinecone(api_key=pinecone_api_key)

# # Create a Pinecone index if it doesn't exist
# index_name = "pdf-file-index"  # Change if needed
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if index_name not in existing_indexes:
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     while not pc.describe_index(index_name).status["ready"]:
#         time.sleep(1)

# index = pc.Index(index_name)

# # Initialize embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # Initialize vector store
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# # HTML template for file upload and search
# html_template = '''
# <!doctype html>
# <html lang="en">
# <head>
#     <meta charset="utf-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <title>Upload PDF File and Search</title>
# </head>
# <body>
#     <h1>Upload PDF File</h1>
#     <form action="/upload" method="post" enctype="multipart/form-data">
#         <input type="file" name="file" accept=".pdf">
#         <input type="submit" value="Upload">
#     </form>
#     {% if upload_message %}
#         <p>{{ upload_message }}</p>
#     {% endif %}

#     <h1>Search</h1>
#     <form action="/search" method="get">
#         <input type="text" name="query" placeholder="Enter your search query">
#         <input type="number" name="k" placeholder="Number of results" value="5" min="1">
#         <input type="submit" value="Search">
#     </form>
#     {% if no_db_message %}
#         <p>{{ no_db_message }}</p>
#     {% endif %}
#     {% if search_result %}
#         <h2>Search Results for "{{ query }}":</h2>
#         <ul>
#         {% for result in search_result %}
#             <li>{{ result.text }} (Score: {{ result.score }})</li>
#         {% endfor %}
#         </ul>
#     {% endif %}
# </body>
# </html>
# '''

# @app.route('/')
# def index():
#     return render_template_string(html_template)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return render_template_string(html_template, upload_message="No file part")
    
#     file = request.files['file']
#     if file.filename == '':
#         return render_template_string(html_template, upload_message="No selected file")
    
#     if file and file.filename.endswith('.pdf'):
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             file_path = temp_file.name
#             file.save(file_path)
        
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
        
#         uuids = [str(uuid4()) for _ in range(len(documents))]
#         vector_store.add_documents(documents=documents, ids=uuids)
        
#         os.remove(file_path)
        
#         return render_template_string(html_template, upload_message="PDF processed and documents added.")
#     else:
#         return render_template_string(html_template, upload_message="Invalid file type. Please upload a .pdf file.")

# @app.route('/search', methods=['GET'])
# def search():
#     query = request.args.get('query', '')
#     k = int(request.args.get('k', 5))  # Default value of k is 5 if not provided

#     if not query:
#         return render_template_string(html_template, upload_message=None, search_result=None, query=query, no_db_message="No query provided."), 400

#     try:
#         # Check if the index exists and has embeddings
#         if not index or not vector_store:
#             no_db_message = "No vector database is present. Please upload files to create the database."
#             return render_template_string(html_template, search_result=None, query=query, no_db_message=no_db_message)

#         # Create the RetrievalQA chain
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#         qa_chain = RetrievalQA.from_chain_type(
#             llm, retriever=vector_store.as_retriever()
#         )
        
#         # Perform the search using the query
#         search_results = qa_chain({"query": query})

#         # Check if results are present
#         if 'result' not in search_results:
#             no_db_message = "No results found for the query."
#             return render_template_string(html_template, search_result=None, query=query, no_db_message=no_db_message)
        
#         # Format results for display
#         formatted_results = [
#             {"text": search_results["result"], "score": "N/A"}  # Adjust as needed
#         ]

#         return render_template_string(html_template, search_result=formatted_results, query=query, upload_message=None)

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")  # Log the error
#         return render_template_string(html_template, search_result=None, query=query, no_db_message="An unexpected error occurred during search."), 500

# if __name__ == '__main__':
#     app.run(debug=True)









































































# from flask import Flask, request, render_template_string
# import os
# from langchain_core.prompts import MessagesPlaceholder
# import tempfile
# import time
# from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrievalQA
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# # Load environment variables
# embedding_model = 'text-embedding-3-small'
# embedd = OpenAIEmbeddings(
#     api_key=os.getenv("OPENAI_API_KEY"), 
#     model=embedding_model
# )

# app = Flask(__name__)

# # Initialize Pinecone
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=pinecone_api_key)

# index_nam = "file"

# # Check if the index exists
# def check_index_exists(index_name):
#     existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
#     return index_name in existing_indexes

# # Create index if it doesn't exist
# if not check_index_exists(index_nam):
#     pc.create_index(
#         name=index_nam,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     while not pc.describe_index(index_nam).status["ready"]:
#         time.sleep(1)

# index = pc.Index(index_nam)
# vectorstore = PineconeVectorStore(index_name=index_nam, embedding=embedd)

# # HTML template for file upload and search
# html_template = '''
# <!doctype html>
# <html lang="en">
# <head>
#     <meta charset="utf-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <title>Upload File and Search</title>
# </head>
# <body>
#     <h1>Upload File</h1>
#     <form action="/upload" method="post" enctype="multipart/form-data">
#         <input type="file" name="files" accept=".pdf,.docx,.txt" multiple>
#         <input type="submit" value="Upload">
#     </form>
#     {% if upload_message %}
#         <p>{{ upload_message }}</p>
#     {% endif %}

#     <h1>Search</h1>
#     <form action="/search" method="get">
#         <input type="text" name="query" placeholder="Enter your search query">
#         <input type="submit" value="Search">
#     </form>
#     {% if no_db_message %}
#         <p>{{ no_db_message }}</p>
#     {% endif %}
#     {% if search_result %}
#         <h2>Search Results for "{{ query }}":</h2>
#         <ul>
#         {% for result in search_result %}
#             <li>{{ result.text }}</li>
#         {% endfor %}
#         </ul>
#     {% endif %}

#     {% if conversation_history %}
#         <h2>Conversation History:</h2>
#         <ul>
#         {% for message in conversation_history %}
#             <li><strong>{{ message.role }}:</strong> {{ message.content }}</li>
#         {% endfor %}
#         </ul>
#     {% endif %}
# </body>
# </html>
# '''

# # Initialize the conversation history list
# conversation_history = []

# @app.route('/')
# def index():
#     return render_template_string(html_template, conversation_history=conversation_history)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'files' not in request.files:
#         return render_template_string(html_template, upload_message="No file part", conversation_history=conversation_history)

#     files = request.files.getlist('files')
#     if not files:
#         return render_template_string(html_template, upload_message="No files selected", conversation_history=conversation_history)

#     all_documents = []
#     for file in files:
#         if file.filename == '':
#             continue
        
#         file_type = file.filename.split('.')[-1].lower()
#         if file_type not in ['pdf', 'docx', 'txt']:
#             continue
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
#             file_path = temp_file.name
#             file.save(file_path)
        
#         documents = []
#         if file_type == 'pdf':
#             loader = PyPDFLoader(file_path)
#             documents = loader.load()
#         elif file_type == 'txt':
#             loader = TextLoader(file_path)
#             documents = loader.load()
#         elif file_type == 'docx':
#             loader = Docx2txtLoader(file_path)
#             documents = loader.load()
        
#         all_documents.extend(documents)
        
#         os.remove(file_path)
    
#     if all_documents:
#         vectorstore.add_documents(documents=all_documents)
#         return render_template_string(html_template, upload_message="Files processed and documents added.", conversation_history=conversation_history)
#     else:
#         return render_template_string(html_template, upload_message="No valid files uploaded.", conversation_history=conversation_history)

# @app.route('/search', methods=['GET'])
# def search():
#     query = request.args.get('query', '')

#     if not query:
#         return render_template_string(
#             html_template, 
#             search_result=None, 
#             query=query, 
#             no_db_message="No query provided.",
#             conversation_history=conversation_history
#         ), 400

#     try:
#         # Check if the index exists
#         if not check_index_exists(index_nam):
#             return render_template_string(
#                 html_template, 
#                 search_result=None, 
#                 query=query, 
#                 no_db_message="No documents found. Please upload documents first.",
#                 conversation_history=conversation_history
#             ), 400

#         # Check if the index contains documents
#         index = pc.Index(index_nam)
#         # Use Pinecone's query to see if there are any vectors in the index
#         num_vectors = index.describe_index_stats().get('total_vector_count', 0)
        
#         if num_vectors == 0:
#             return render_template_string(
#                 html_template, 
#                 search_result=None, 
#                 query=query, 
#                 no_db_message="No documents found. Please upload documents first.",
#                 conversation_history=conversation_history
#             ), 400

#         # Initialize the retriever and chain
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#         qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

#         # Append user input to conversation history
#         conversation_history.append({"role": "user", "content": query})

#         # Perform the search using the query
#         ai_response = rag_chain.invoke({"input": query, "chat_history": conversation_history})
#         response_content = ai_response['answer']

#         # Append AI response to conversation history
#         conversation_history.append({"role": "assistant", "content": response_content})

#         # Format results for display
#         formatted_results = [{"text": response_content}]

#         return render_template_string(
#             html_template, 
#             search_result=formatted_results, 
#             query=query, 
#             upload_message=None,
#             conversation_history=conversation_history
#         )

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")  # Log the error
#         return render_template_string(
#             html_template, 
#             search_result=None, 
#             query=query, 
#             no_db_message="An unexpected error occurred during search.",
#             conversation_history=conversation_history
#         ), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# working code of pine cone
from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from langchain_core.prompts import MessagesPlaceholder
import tempfile
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
embedding_model = 'text-embedding-3-small'
embedd = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model=embedding_model
)

app = Flask(__name__)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_nam = "file"

# Check if the index exists
def check_index_exists(index_name):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    return index_name in existing_indexes

# Create index if it doesn't exist
if not check_index_exists(index_nam):
    pc.create_index(
        name=index_nam,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_nam).status["ready"]:
        time.sleep(1)

index = pc.Index(index_nam)
vectorstore = PineconeVectorStore(index_name=index_nam, embedding=embedd)

# Initialize the conversation history list with a default AI message
conversation_history = [{"role": "assistant", "content": "Hello! You need to upload documents first to start your conversation"}]

@app.route('/')
def index():
    return render_template('index.html', conversation_history=conversation_history, upload_success=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'message': 'No files selected'}), 400

    all_documents = []
    unsupported_files = []

    for file in files:
        if file.filename == '':
            continue

        file_type = file.filename.split('.')[-1].lower()
        if file_type not in ['pdf', 'docx', 'txt']:
            unsupported_files.append(file.filename)
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            file_path = temp_file.name
            file.save(file_path)

        documents = []
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_type == 'txt':
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

        all_documents.extend(documents)
        os.remove(file_path)

    if unsupported_files:
        return jsonify({'success': False, 'message': f'Unsupported file types: {", ".join(unsupported_files)}. Please upload files of type: PDF, DOCX, or TXT.'}), 400

    if all_documents:
        # Add documents to Pinecone vectorstore
        vectorstore.add_documents(documents=all_documents)
        return jsonify({'success': True, 'message': 'Files uploaded and processed successfully'})
    else:
        return jsonify({'success': False, 'message': 'No valid files uploaded'}), 400




@app.route('/query_response', methods=['POST'])
def query_response():
    data = request.get_json()
    query = data.get('query')

    if not query or not isinstance(query, list):
        return jsonify({'error': 'Invalid query format.'}), 400

    query_text = ""
    for q in query:
        if isinstance(q, dict) and q.get('role') == 'user' and 'content' in q:
            query_text = q['content']
        else:
            return jsonify({'error': 'Invalid query content.'}), 400

    try:
        # Initialize the retriever and chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

        # Append user input to conversation history
        conversation_history.append({"role": "user", "content": query_text})

        # Perform the search using the query
        ai_response = rag_chain.invoke({"input": query_text, "chat_history": conversation_history})
        response_content = ai_response['answer']

        # Append AI response to conversation history
        conversation_history.append({"role": "assistant", "content": response_content})

        return jsonify({'response': response_content})

    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Log the error
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == '__main__':
    app.run(debug=True)






    
# import os
# import tempfile
# import time
# from uuid import uuid4

# from flask import Flask, request, render_template_string
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv

# load_dotenv()

# embedding_model = 'text-embedding-3-small'
# embedd = OpenAIEmbeddings(
#     api_key=os.getenv("OPENAI_API_KEY"), 
#     model=embedding_model
# )

# app = Flask(__name__)

# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# print(pinecone_api_key)

# pc = Pinecone(api_key=pinecone_api_key)

# index_nam = "file"
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if index_nam not in existing_indexes:
#     pc.create_index(
#         name=index_nam,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     while not pc.describe_index(index_nam).status["ready"]:
#         time.sleep(1)

# index = pc.Index(index_nam)
# print(index)

# html_template = '''
# <!doctype html>
# <html lang="en">
# <head>
#     <meta charset="utf-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <title>Upload File and Search</title>
# </head>
# <body>
#     <h1>Upload File</h1>
#     <form action="/upload" method="post" enctype="multipart/form-data">
#         <input type="file" name="files" accept=".pdf,.docx,.txt" multiple>
#         <input type="submit" value="Upload">
#     </form>

#     {% if upload_message %}
#         <p>{{ upload_message }}</p>
#     {% endif %}

#     <h1>Search</h1>
#     <form action="/search" method="get">
#         <input type="text" name="query" placeholder="Enter your search query">
#         <input type="submit" value="Search">
#     </form>

#     {% if no_db_message %}
#         <p>{{ no_db_message }}</p>
#     {% endif %}

#     {% if search_result %}
#         <h2>Search Results for "{{ query }}":</h2>
#         <ul>
#         {% for result in search_result %}
#             <li>{{ result.text }}</li>
#         {% endfor %}
#         </ul>
#     {% endif %}

#     {% if metadata_list %}
#         <h2>Existing Files Metadata:</h2>
#         <ul>
#         {% for metadata in metadata_list %}
#             <li>{{ metadata }}</li>
#         {% endfor %}
#         </ul>
#     {% endif %}
# </body>
# </html>
# '''

# vectorstore = PineconeVectorStore(
#     index_name=index_nam,
#     embedding=embedd
# )



# @app.route('/')
# def index():
#     metadata_list = []
#     try:
#         # Fetch all vector IDs from the index
#         all_ids = []
#         index = pc.Index(index_nam)
        
#         # Retrieve all vector IDs in batches (assuming batch size here, adjust as needed)
#         ids_batches = index.list()
#         merged_list = [item for sublist in ids_batches for item in sublist]
        
#         for vec_id in merged_list:
#             try:
#                 # Fetch vector and metadata for each ID
#                 vector_info = index.fetch([vec_id])
#                 if vec_id in vector_info['vectors']:
#                     metadata = vector_info['vectors'][vec_id].get('metadata', {})
#                     file_name = metadata.get('file_name', 'No file name')
#                     metadata_list.append({
#                         'id': vec_id,
#                         'file_name': file_name
#                     })
#             except Exception as e:
#                 print(f"Error retrieving metadata for ID {vec_id}: {str(e)}")
#                 continue
#     except Exception as e:
#         print(f"Error retrieving all IDs: {str(e)}")

#     return render_template_string(html_template, metadata_list=metadata_list)


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'files' not in request.files:
#         return render_template_string(html_template, upload_message="No file part")

#     files = request.files.getlist('files')
#     if not files:
#         return render_template_string(html_template, upload_message="No files selected")

#     all_documents = []
#     existing_file_names = set()
    
#     try:
#         # Fetch all existing document metadata
#         index = pc.Index(index_nam)
#         all_ids = index.list()
#         for batch_ids in all_ids:
#             if batch_ids:
#                 vector_info = index.fetch(batch_ids)
#                 for vec_id in batch_ids:
#                     if vec_id in vector_info['vectors']:
#                         metadata = vector_info['vectors'][vec_id].get('metadata', {})
#                         file_name = metadata.get('file_name', None)
#                         if file_name:
#                             existing_file_names.add(file_name)
#     except Exception as e:
#         print(f"Error retrieving existing files: {str(e)}")

#     for file in files:
#         if file.filename == '':
#             continue
        
#         file_name = file.filename
#         file_type = file_name.split('.')[-1].lower()
        
#         if file_type not in ['pdf', 'docx', 'txt']:
#             continue

#         # Check if the file_name already exists
#         if file_name in existing_file_names:
#             return render_template_string(
#                 html_template, 
#                 upload_message=f"File '{file_name}' already uploaded. Please choose a different file."
#             )

#         # Process and upload the file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
#             file_path = temp_file.name
#             file.save(file_path)
        
#         documents = []
#         if file_type == 'pdf':
#             loader = PyPDFLoader(file_path)
#             documents = loader.load()
#         elif file_type == 'txt':
#             loader = TextLoader(file_path)
#             documents = loader.load()
#         elif file_type == 'docx':
#             loader = Docx2txtLoader(file_path)
#             documents = loader.load()
        
#         ids = []
#         metadatas = []
#         for doc in documents:
#             doc_id = str(uuid4())
#             ids.append(doc_id)
#             metadatas.append({"file_name": file_name})
#             all_documents.append((doc_id, doc))

#         os.remove(file_path)
    
#     if all_documents:
#         # Add documents to Pinecone with IDs and metadata
#         try:
#             vectorstore.add_texts(
#                 texts=[doc for _, doc in all_documents],
#                 ids=[id for id, _ in all_documents],
#                 metadatas=metadatas
#             )
#         except Exception as e:
#             print(f"Error adding documents to Pinecone: {str(e)}")
        
#         return render_template_string(
#             html_template, 
#             upload_message="Files processed and documents added."
#         )
#     else:
#         return render_template_string(
#             html_template, 
#             upload_message="No valid files uploaded."
#         )
      


# @app.route('/search', methods=['GET'])
# def search():
#     query = request.args.get('query', '')
    
#     if not query:
#         return render_template_string(html_template, search_result=None, query=query, no_db_message="No query provided."), 400

#     try:
#         vector_store = PineconeVectorStore(
#             index_name=index_nam,
#             embedding=embedd
#         )
#         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#         qa = RetrievalQA.from_chain_type(
#             llm=llm, 
#             chain_type="stuff",
#             retriever=vector_store.as_retriever()
#         )
        
#         search_results = qa.invoke({"query": query})

#         if not search_results or 'result' not in search_results:
#             no_db_message = "No results found for the query."
#             return render_template_string(html_template, search_result=None, query=query, no_db_message=no_db_message)
        
#         formatted_results = [
#             {"text": search_results["result"]}
#         ]

#         return render_template_string(html_template, search_result=formatted_results, query=query)

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")  
#         return render_template_string(html_template, search_result=None, query=query, no_db_message="An unexpected error occurred during search."), 500

# if __name__ == '__main__':
#     app.run(debug=True)
