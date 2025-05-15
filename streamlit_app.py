# streamlit_app.py
import streamlit as st
from main import DocumentStore, RouterAgent, RetrieverAgent, ReasoningAgent, ComplianceAgent

def run_agentic_system(documents, query):
    # Initialize doc stores only once (ideally cache this)
    doc_stores = {doc_id: DocumentStore(path, doc_id) for doc_id, path in documents.items()}
    router_agent = RouterAgent(doc_stores)
    retriever_agent = RetrieverAgent(doc_stores)
    reasoning_agent = ReasoningAgent()
    compliance_agent = ComplianceAgent()

    relevant_docs = router_agent.route(query)
    retrieved_chunks = retriever_agent.retrieve(query, relevant_docs)
    if not retrieved_chunks:
        return "Sorry, no relevant information found in documents."
    answer = reasoning_agent.reason(query, retrieved_chunks)
    safe_answer = compliance_agent.review(answer)
    return safe_answer

def main():
    st.title("Agentic Document QA")

    documents = {
        "HR_Handbook": r"C:\Users\lavan\Downloads\HR_Handbook.pdf",
        "Security_Protocol": r"C:\Users\lavan\Downloads\Security_Protocol.pdf",
        "Sales_Playbook": r"C:\Users\lavan\Downloads\Sales_Playbook.pdf",
        "Engineering_SOPs": "https://datasense78.github.io/engineeringsop/"
    }

    query = st.text_input("Enter your question:")
    if query:
        answer = run_agentic_system(documents, query)
        st.write("### Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
