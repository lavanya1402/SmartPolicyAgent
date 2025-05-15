import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

class DocumentStore:
    def __init__(self, path_or_url: str, doc_id: str):
        self.doc_id = doc_id
        self.path_or_url = path_or_url
        self.docs = self.load_docs()
        self.chunks = self.split_docs()
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)

    def load_docs(self) -> List[Document]:
        if self.path_or_url.lower().startswith("http"):
            loader = WebBaseLoader(self.path_or_url)
            return loader.load()
        else:
            loader = PyPDFLoader(self.path_or_url)
            return loader.load()

    def split_docs(self) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(self.docs)

class RouterAgent:
    def __init__(self, doc_stores: Dict[str, DocumentStore]):
        self.doc_stores = doc_stores
        self.llm = ChatOpenAI(temperature=0)

    def route(self, query: str) -> List[str]:
        prompt = (
            f"You are a router deciding relevant documents for the query.\n"
            f"Available documents: {list(self.doc_stores.keys())}\n"
            f"Query: {query}\n"
            "Respond with a comma-separated list of relevant document IDs only."
        )
        response = self.llm.call_as_llm(prompt)  # returns string here
        text = response  # directly assign string
        relevant_docs = [doc.strip() for doc in text.split(",") if doc.strip() in self.doc_stores]
        return relevant_docs

class RetrieverAgent:
    def __init__(self, doc_stores: Dict[str, DocumentStore]):
        self.doc_stores = doc_stores

    def retrieve(self, query: str, selected_docs: List[str], top_k=5) -> List[Document]:
        chunks = []
        for doc_id in selected_docs:
            vectorstore = self.doc_stores[doc_id].vectorstore
            retrieved = vectorstore.similarity_search(query, k=top_k)
            chunks.extend(retrieved)
        return chunks

class ReasoningAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

    def reason(self, query: str, chunks: List[Document]) -> str:
        context = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])
        prompt = (
            f"You are an expert AI assistant. Use the following extracted document sections to answer the question:\n\n"
            f"{context}\n\n"
            f"Question: {query}\n"
            f"Answer, referencing the document sources (include page numbers if available)."
        )
        response = self.llm.call_as_llm(prompt)
        return response  # already string

class ComplianceAgent:
    def __init__(self):
        pass

    def review(self, answer: str) -> str:
        redacted = answer.replace("confidential", "[REDACTED]")
        redacted += "\n\n[Note: This response is generated based on provided documents and may not replace official policies.]"
        return redacted

def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    documents = {
        "HR_Handbook": r"C:\Users\lavan\Downloads\HR_Handbook.pdf",
        "Security_Protocol": r"C:\Users\lavan\Downloads\Security_Protocol.pdf",
        "Sales_Playbook": r"C:\Users\lavan\Downloads\Sales_Playbook.pdf",
        "Engineering_SOPs": "https://datasense78.github.io/engineeringsop/"
    }

    print("[INFO] Loading documents and building vector stores...")
    doc_stores = {doc_id: DocumentStore(path, doc_id) for doc_id, path in documents.items()}

    router_agent = RouterAgent(doc_stores)
    retriever_agent = RetrieverAgent(doc_stores)
    reasoning_agent = ReasoningAgent()
    compliance_agent = ComplianceAgent()

    print("[INFO] Agentic system ready. Type 'exit' to quit.")

    while True:
        query = input("\nUser: ")
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("[LOG] RouterAgent selecting documents...")
        relevant_docs = router_agent.route(query)
        print(f"[LOG] Relevant docs: {relevant_docs}")

        print("[LOG] RetrieverAgent fetching chunks...")
        retrieved_chunks = retriever_agent.retrieve(query, relevant_docs)

        if not retrieved_chunks:
            print("Final Answer:\nSorry, no relevant information found in documents.")
            continue

        print("[LOG] ReasoningAgent composing answer...")
        answer = reasoning_agent.reason(query, retrieved_chunks)

        print("[LOG] ComplianceAgent reviewing answer...")
        safe_answer = compliance_agent.review(answer)

        print("\nFinal Answer:\n", safe_answer)

if __name__ == "__main__":
    main()
