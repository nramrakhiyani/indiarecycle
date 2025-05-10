import os
import sys
import requests
import gradio as gr

from ollama import Client #SETUP: install ollama python package
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

os.environ["no_proxy"] = "localhost,127.0.0.1"
#proxy_servers = {'http':'', 'https:',''}

#For semantic similarity
hf_emb = HuggingFaceEmbeddings(model_name = '') #SETUP: path to all_mpnet_base_v2

type = sys.argv[1]
enable_history = sys.argv[2]
with_rag = sys.argv[3]
if(with_rag.lower() == 'true'):
	rag_index_path = sys.argv[4]

def ollama_chat_realtime(message, history):
	text = message
	vector_db = None
	if(with_rag.lower() == 'true'):
		vector_db = FAISS.load_local(rag_index_path, hf_emb, allow_dangerous_deserialization = True)
		retriever = vector_db.as_retriever(search_kwargs = {'k':2}, verbose = True)
		search_result = retriever.invoke(text)
		context = search_result[0].page_content + '\n' + search_result[1].page_content
		text = """Answer the following question based only on the provided context:\n\n<context>""" + context + """</context>\n\nQuestion: """ + text

	#chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
	request_messages = []
	if(enable_history.lower() == 'true'):
		if(len(history) > 0):
			request_messages = history[:]
		request_messages.append({'role':'user', 'context':text})
		history.append({'role':'user', 'context':message})
	else:
		request_messages = [{'role':'user', 'context':text}]

	client = Client(host = 'http://localhost:11434', headers = {})
	stream = client.chat(model = '', messages = request_messages, stream = True) #SETUP Model name as 'model' argument

	partial_text = ''
	for chunk in stream:
		new_text = chunk['message']['content']
		partial_text += new_text
		yield partial_text
	history.append({'role':'assistant', 'context':partial_text})

def generate_error_response():
	return 'Improper model selection'

if(type.find('internal_ollama')):
	gr.ChatInterface(ollama_chat_realtime, type = 'messages').launch(server_name = '0.0.0.0', server_port = 7860)
else:
	gr.ChatInterface(generate_error_response, type = 'messages').launch(server_name = '0.0.0.0', server_port = 7860)