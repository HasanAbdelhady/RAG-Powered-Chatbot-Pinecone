from groq import Groq
import os
from index import setup_index, check_index_exists
from dotenv import load_dotenv
from prompt import prompt, context_retrieval
import asyncio

import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder_attention_mask.*")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

async def main():
    if not await check_index_exists():
        await setup_index()
        print("Setting up index...")
        print("Done ✅")

    client = Groq()

    system_prompt = "You're a personal assistant that can answer questions about Hasan Abdelhady's resume."

    while True:
        user_prompt = input("User:")
        final_prompt = prompt(context_retrieval(query=user_prompt, index_name="hasan-data", embed_model="all-MiniLM-L6-v2"), user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":final_prompt}
        ]


        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        print("Assistant: ", response.choices[0].message.content)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})


if __name__ == "__main__":
    asyncio.run(main()) # Run the main function