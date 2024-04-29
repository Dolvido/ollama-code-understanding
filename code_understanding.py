from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import os

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

def load_code_files(directory):
    code_files = list(Path(directory).glob("**/*.py"))
    print(f"Looking for .py files in: {Path(directory).resolve()}")
    print(f"Found {len(code_files)} .py files in the directory.")
    loaders = [TextLoader(str(file)) for file in code_files]
    documents = []
    for loader in loaders:
        loaded_doc = loader.load()
        if loaded_doc:
            documents.extend(loaded_doc)  # Make sure to extend with the loaded document if it's not None
    print(f"Loaded {len(documents)} documents.")
    return documents  # Add this return statement


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(chunked_documents)} document chunks.")
    return chunked_documents

def create_vector_store(documents):
    if not documents:
        raise ValueError("No documents to process.")
    embeddings = GPT4AllEmbeddings()
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store

def answer_question(vector_store, question):
    docs = vector_store.similarity_search(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"Answer the following question based on the provided code context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    model = Ollama(model="mistral")
    response = model.predict(prompt)
    return response

def generate_code(vector_store, instruction):
    docs = vector_store.similarity_search(instruction)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"Generate code based on the following instruction and code context:\n\nContext:\n{context}\n\nInstruction: {instruction}\n\nGenerated Code:"
    model = Ollama(model="codellama")
    response = model.predict(prompt)
    return response

# Load code files from the ./data directory
print(os.getcwd())
code_directory = "./data"
documents = load_code_files(code_directory)

# Split the loaded documents into chunks
chunked_documents = split_documents(documents)

# Create a vector store using GPT4All embeddings
vector_store = create_vector_store(chunked_documents)

# Main loop to ask the user for prompts
while True:
    action = input("Enter 'ask' to ask a question or 'generate' to generate code. Type 'exit' to quit: ")
    if action.lower() == 'exit':
        break
    elif action.lower() == 'ask':
        question = input("Enter your question: ")
        answer = answer_question(vector_store, question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    elif action.lower() == 'generate':
        instruction = input("Enter your instruction: ")
        generated_code = generate_code(vector_store, instruction)
        print(f"Instruction: {instruction}")
        print(f"Generated Code:\n{generated_code}")
    else:
        print("Invalid input. Please try again.")