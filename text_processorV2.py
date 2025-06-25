import os
import uuid
import json
import numpy as np
import asyncio
import PyPDF2
import aiofiles
from typing import Any
from sklearn.preprocessing import normalize
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['HF_TOKEN'] = ''  # توكن هاغينغ فيس

class TextDocumentProcessor:

    @staticmethod
    async def read_txt(txt_path):
        async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as file:
            return await file.read()

    @staticmethod
    async def read_pdf(pdf_path):
        text = ''
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    

    @staticmethod
    async def chunk_text(text, client: AsyncOpenAI):
        prompt = f"""Split the text below into logical chunks based on section titles.
        Requirements:
        - Do NOT rewrite or remove any part of the text.
        - Each chunk must include:
            - "title": actual title or brief inferred label.
            - "txt": full, unmodified text under that title/topic.
        - Output must be deterministic and consistent for the same input.
        For "Summary": List general topics  — no names or details.
        format:
        [
  {{ "title": "Summary", "txt": "Topics: [Cluster 1, Cluster 2, ...]" }},
  {{ "title": "Title 1", "txt": "Text 1" }},
  ...
]
        Text:
        \"\"\"{text}\"\"\""""

        response = await client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        return [f"Title: {c['title']}\n{c['txt']}" for c in json.loads(content)]

    # imbedding with Transformer
    @staticmethod  
    async def generate_embeddings(text_chunks, model: SentenceTransformer):
        embeddings = model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)
        return normalize(embeddings).tolist()
    
    # imbedding with OpenAi
    """
    @staticmethod 
    async def generate_embeddings(text_chunks, client: AsyncOpenAI):
        responses = await asyncio.gather(*[
        client.embeddings.create(model="text-embedding-3-small", input=chunk)
        for chunk in text_chunks
    ])
        embeddings = [res.data[0].embedding for res in responses]
        return normalize(np.array(embeddings)).tolist()
    """

    @staticmethod
    async def store_embeddings(embeddings, chunks, doc_name, qdrant_client: AsyncQdrantClient, collection_name: str):
        points = []
        for emb, chunk in zip(embeddings, chunks):
            title, text = "", chunk
            if chunk.startswith("Title:"):
                parts = chunk.split("\n", 1)
                title = parts[0].replace("Title:", "").strip()
                text = parts[1] if len(parts) > 1 else ""

            points.append({
                'id': str(uuid.uuid4()),
                'vector': emb,
                'payload': {'title': title, 'text': text, 'doc_name': doc_name}
            })

        try:
            await qdrant_client.get_collection(collection_name)
        except:
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )

        await qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Stored {len(points)} items from {doc_name} into Qdrant.")

    @staticmethod
    async def process_text_files(folder_path: str, qdrant_client: AsyncQdrantClient, collection_name: str, model: SentenceTransformer, client: AsyncOpenAI):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.txt', '.pdf')):
                full_path = os.path.join(folder_path, file_name)
                doc_name = os.path.splitext(file_name)[0]
                text = await (
                    TextDocumentProcessor.read_txt(full_path)
                    if file_name.endswith('.txt')
                    else TextDocumentProcessor.read_pdf(full_path)
                )
                chunks = await TextDocumentProcessor.chunk_text(text, client)
                embeddings = await TextDocumentProcessor.generate_embeddings(chunks, model)         # imbedding with Transformer
                """embeddings = await TextDocumentProcessor.generate_embeddings(chunks, client)"""  # imbedding with OpenAi
                await TextDocumentProcessor.store_embeddings(embeddings, chunks, doc_name, qdrant_client, collection_name)

# ✅ طريقة الاستخدام
if __name__ == "__main__":
    async def main():
        val_uuid='uuid'
        qdrant = AsyncQdrantClient(host="localhost", port=6333)
        model = SentenceTransformer("HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1", token=os.environ['HF_TOKEN'])
        client = AsyncOpenAI(api_key='')
        await TextDocumentProcessor.process_text_files(
            folder_path="docs",
            qdrant_client=qdrant,
            collection_name=f"{val_uuid}_txt_pdf",
            model=model,
            client=client
        )

    asyncio.run(main())
