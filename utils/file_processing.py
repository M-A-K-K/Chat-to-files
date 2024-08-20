import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from werkzeug.utils import secure_filename

load_dotenv()  

def allowed_file(filename):
    allowed_extensions = os.getenv('ALLOWED_EXTENSIONS', 'pdf,txt,docx').split(',')
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def process_file(file):
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    file_path = os.path.join(os.getenv('UPLOAD_FOLDER', 'uploads'), filename)
    file.save(file_path)

    if file_ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_ext == 'txt':
        loader = TextLoader(file_path)
    elif file_ext == 'docx':
        loader = Docx2txtLoader(file_path)

    documents = loader.load()
    text_content = ' '.join([doc.page_content for doc in documents])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text_content)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    embedded_texts = embeddings.embed_documents(texts)

    return [(f"{filename}-{i}", vector) for i, vector in enumerate(embedded_texts)]
