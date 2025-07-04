# tabular_document_processor.py
import os
import uuid
import json
import numpy as np
import asyncio
import aiofiles
import pandas as pd
from io import StringIO, BytesIO
from sklearn.preprocessing import normalize
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

from openai import AsyncOpenAI  
  


class TabularDocumentProcessor:
    @staticmethod
    async def read_csv(csv_path):
        async with aiofiles.open(csv_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        return pd.read_csv(StringIO(content))

    @staticmethod
    async def read_xlsx(xlsx_path):
        async with aiofiles.open(xlsx_path, mode='rb') as f:
            content = await f.read()
        return pd.read_excel(BytesIO(content))

    # imbedding with OpenAi
    
    @staticmethod  
    async def generate_embeddings(texts, client: AsyncOpenAI):
        responses = await asyncio.gather(*[
        client.embeddings.create(model="text-embedding-3-small", input=text)
        for text in texts
    ])
        embeddings = [res.data[0].embedding for res in responses]
        return normalize(np.array(embeddings)).tolist()
        
    
              
    @staticmethod
    async def store_embeddings(embeddings, rows, doc_name, qdrant_client, collection_name):
        points = [
            {
                'id': str(uuid.uuid4()),
                'vector': emb,
                'payload': {**row, 'doc_name': doc_name}
            }
            for emb, row in zip(embeddings, rows)
        ]
        try:
            await qdrant_client.get_collection(collection_name)
        except:
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )
        await qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Stored {len(points)} rows from {doc_name} into Qdrant.")


    """async def process_tabular_files(folder_path, qdrant_client, collection_name, client):""" # if use openAI embedding
    @staticmethod
    async def process_tabular_files(folder_path, qdrant_client, collection_name, client):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.csv', '.xlsx')):
                full_path = os.path.join(folder_path, file_name)
                if file_name.endswith('.csv'):
                    df = await TabularDocumentProcessor.read_csv(full_path)
                else:
                    df = await TabularDocumentProcessor.read_xlsx(full_path)

                rows = [{str(k): v for k, v in row.items()} for row in df.to_dict(orient='records')]
                texts = [json.dumps(row, ensure_ascii=False) for row in rows]
                
                embeddings = await TabularDocumentProcessor.generate_embeddings(texts, client)    # imbedding with OpenAi
                await TabularDocumentProcessor.store_embeddings(
                    embeddings, rows, os.path.splitext(file_name)[0], qdrant_client, collection_name
                )


# 🏁 main entry point
if __name__ == '__main__':
    async def main():
        val_uuid = "uuid"  
        folder_path = "docs"
        collection_name = f"{val_uuid}_tabular"
        client = AsyncOpenAI(api_key='')
        # Qdrant و الموديل
        qdrant_client = AsyncQdrantClient(host="localhost", port=6333)



        # تنفيذ المعالجة
        await TabularDocumentProcessor.process_tabular_files(
            folder_path=folder_path,
            qdrant_client=qdrant_client,
            collection_name=collection_name,

            client=client
        )

    asyncio.run(main())
