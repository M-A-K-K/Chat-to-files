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
