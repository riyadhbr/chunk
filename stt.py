import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=""
)

async def transcribe_audio_file(filepath: str) -> str:
    with open(filepath, "rb") as f:

        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    
    print(response.text)
    return response.text

# مثال على الاستخدام
if __name__ == "__main__":
    async def main():
        path = "output.wav"  # ضع مسار الملف هنا
        await transcribe_audio_file(path)
        

    asyncio.run(main())
