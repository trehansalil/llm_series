from pymongo import MongoClient
from pymilvus import Milvus, DataType, CollectionSchema, FieldSchema, Collection
from src.chat_service import LLMManager

def fetch_news_data():
    client = MongoClient('mongodb+srv://thanos_inshorts:thanos@cluster0.hlbuku7.mongodb.net/?retryWrites=true&w=majority')
    filter = {}
    sort = list({
        'updated_at': -1
    }.items())

    result = client['inshorts_db']['news_data'].find(
        filter=filter,
        sort=sort
    )
    return result

def embed_text(text):
    llm_manager = LLMManager()
    embedding = llm_manager.generate_embedding(text, verbose=1)
    return embedding

def store_embeddings(embeddings):
    milvus = Milvus(host='localhost', port='19530')
    collection_name = 'news_embeddings'

    fields = [
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1024)
    ]
    schema = CollectionSchema(fields, description="News Embeddings Collection")
    collection = Collection(name=collection_name, schema=schema)

    collection.insert([embeddings])
    collection.load()

def main():
    news_data = fetch_news_data()
    for news in news_data:
        text = news['text']
        embedding = embed_text(text)
        store_embeddings(embedding)

if __name__ == "__main__":
    main()
