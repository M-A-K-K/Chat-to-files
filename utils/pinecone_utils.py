import pinecone
import os
from dotenv import load_dotenv

load_dotenv()  

def get_index():
    pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    index_name = 'document-embeddings'
    dimension = 1536

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=pinecone.ServerlessSpec(
                cloud='aws',
                region=os.getenv('PINECONE_ENVIRONMENT')
            )
        )
    
    return pc.Index(index_name)

def upsert_embeddings(index, embeddings):
    index.upsert(vectors=embeddings)
