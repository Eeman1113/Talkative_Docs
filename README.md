# Talkative_Docs

Talkative_Docs is a project designed to facilitate question answering (QA) over large textual documents using advanced language models and embeddings. This README will guide you through setting up and using the code provided in this repository.

## Installation

Before using the code, you need to install the required dependencies. You can install them using pip:

```bash
!pip -q install langchain
!pip -q install torch
!pip -q install sentence_transformers
!pip -q install faiss-cpu
!pip -q install huggingface-hub
!pip -q install pypdf
!pip -q install accelerate
!pip -q install git+https://github.com/huggingface/transformers
!wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Eeman1113/Talkative_Docs.git
```

2. Import the necessary modules and initialize the required components:

```python
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
```

3. Load your text data:

```python
loader = TextLoader("/path/to/your/text/file.txt")
data = loader.load()
```

4. Split the extracted data into text chunks:

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
```

5. Download the embeddings:

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

6. Create embeddings for each of the text chunks and save them:

```python
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
vector_store.save_local("faiss_index")
```

7. Load the saved embeddings:

```python
vector_store = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
```

8. Import the LLM model:

```python
llm = LlamaCpp(
    streaming=True,
    model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4050,
    n_gpu_layers=45,
    n_threads=7,
    max_tokens=32
)
```

9. Initialize the QA system:

```python
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))
```

10. Run a query:

```python
query = "Your question goes here"
qa.run(query)
```

## Future Aspects

Some potential future aspects for this project include:

- Improving the efficiency and accuracy of the retrieval and QA systems.
- Supporting more languages and diverse types of documents.
- Integrating with other QA systems and platforms.
- Developing a user-friendly interface for easier interaction.
- Enhancing the model's ability to handle longer documents and more complex queries.

Feel free to contribute to the project by implementing these or other features!

## Contributing

If you want to contribute to this project, please fork the repository, make your changes, and submit a pull request. We welcome any contributions, whether they are bug fixes, enhancements, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For any questions or concerns, please contact [Eeman1113](https://github.com/Eeman1113).

Enjoy using Talkative_Docs!
