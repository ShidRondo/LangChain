from dotenv import load_dotenv
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY. Check your .env file.")

# 1. Create the Claude LLM
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 2. Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a friendly assistant. Answer the user question clearly.\n\nUser: {question}"
)

# 3. Combine prompt + Claude
chain = prompt | llm

if __name__ == "__main__":
    user_question = input("Ask Claude something: ")
    response = chain.invoke({"question": user_question})
    print("\nClaude:", response.content)