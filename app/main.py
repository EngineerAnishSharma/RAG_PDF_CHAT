import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv
load_dotenv()
# Set your API key as an environment variable first
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google GenAI client
client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------- LLM Service ----------------
class LLMService:
    def __init__(self, model="gemini-1.5-flash"):
        self.client = client
        self.model = model

    def generate(self, context: str, question: str):
        prompt = f"""
Answer strictly from the context below.

Context:
{context}

Question:
{question}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    content = candidate.content
                    if hasattr(content, "parts") and content.parts:
                        return content.parts[0].text
            return "LLM returned empty response."
        except genai_errors.ClientError as e:
            if "429" in str(e) or "quota" in str(e).lower():
                return "LLM request failed: quota exhausted. Please try later."
            return f"LLM request failed: {e}"
        except Exception as e:
            return f"LLM request failed: {e}"


llm_service = LLMService()


# ---------------- PDF Handling ----------------
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ---------------- QA Chain ----------------
def get_qa_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context.
If the answer is not in the context, just say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return model, prompt


# ---------------- Main Test ----------------
if __name__ == "__main__":
    # 1️⃣ Provide local PDF paths
    pdf_files = ["example.pdf"]  # replace with your PDF paths
    raw_text = get_pdf_text(pdf_files)
    print("PDF loaded successfully!")

    # 2️⃣ Split into chunks
    chunks = get_text_chunks(raw_text)
    print(f"Text split into {len(chunks)} chunks.")

    # 3️⃣ Create vector store
    create_vector_store(chunks)
    print("Vector store created successfully!")

    # 4️⃣ Load vector store and ask questions
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    model, prompt = get_qa_chain()

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        docs = vector_store.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        full_prompt = prompt.format(context=context, question=question)
        response = model.invoke(full_prompt)
        print("Answer:", response.content)
