from langchain.vectorstores import FAISS
from langchain.huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.community.document_loaders import PyPDFDirectoryLoader

input_document_folder_path = sys.argv[1]
output_index_path = sys.argv[2]

loader = PyPDFDirectoryLoader(input_document_folder_path, glob = '**/*.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
document_chunks = text_splitter.split_document(docs)

hf_emb = HuggingFaceEmbeddings(model_name = '') #SETUP: path to all_mpnet_base_v2
vector_db = FAISS.from_documents(document_chunks, hf_emb)
vector_db.save_local(output_index_path)

