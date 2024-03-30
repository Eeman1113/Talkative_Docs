import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Function to load text data
def load_data(file_path):
    loader = TextLoader(file_path)
    return loader.load()

# Function to split text into chunks
def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
    return text_splitter.split_documents(data)

# Function to create and load embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Function to initialize the retrieval QA system
def initialize_qa(llm_path, text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    llm = LlamaCpp(
        streaming=True,
        model_path=llm_path,
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4050,
        n_gpu_layers=45,
        n_threads=7,
        max_tokens=32
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))
    return qa

# Function to run the query
def run_query(qa, query):
    return qa.run(query)

# Main function
def main():
    st.title("Talkative_Docs QA System")

    # File upload
    uploaded_file = st.file_uploader("Upload your text file", type=["txt"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        text_chunks = split_text(data)
        embeddings = create_embeddings()
        qa = initialize_qa("./mistral-7b-instruct-v0.1.Q4_K_M.gguf", text_chunks, embeddings)

        # Query input
        query = st.text_input("Enter your query:")
        if st.button("Submit"):
            if query:
                result = run_query(qa, query)
                st.write("Answer:", result)
            else:
                st.warning("Please enter a query.")

# Run the app
if __name__ == "__main__":
    main()
