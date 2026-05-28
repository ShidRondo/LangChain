from dotenv import load_dotenv
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load API key
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY. Check your .env file.")

# Claude LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=500
)

# Load document
with open("doc.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector store
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Q&A prompt
qa_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions ONLY from the document.

If the answer is not found in the context, reply:
"I don't know based on this document."

End every answer with:
"(derived from the provided text)"

Context:
{context}

Question: {question}

Answer:
"""
)

# Chain
chain = qa_prompt | llm


def answer_question(question: str) -> tuple[str, str]:
    docs = retriever.invoke(question)

    source_snippet = docs[0].page_content[:120].replace("\n", " ")
    context = "\n\n".join([doc.page_content for doc in docs])

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    answer = response.content

    if "(derived from the provided text)" not in answer:
        answer += " (derived from the provided text)"

    return source_snippet, answer


if __name__ == "__main__":
    print("Claude Document Q&A — type 'exit' to stop.\n")

    while True:
        question = input("Your question: ")

        if question.lower() in ["exit", "quit"]:
            break

        source_snippet, answer = answer_question(question)

        print("\nSource Snippet:", source_snippet, "...")
        print("\nAnswer:", answer, "\n")