import os
import sys
import requests
import gradio as gr
import onnxruntime_genai as og

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

os.environ["no_proxy"] = "localhost,127.0.0.1"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
api_key = '' #SETUP: HF API Key

#proxy_servers = {'http':'', 'https:',''}
hf_emb = HuggingFaceEmbeddings(model_name = '') #SETUP: path to all_mpnet_base_v2

type = sys.argv[1]
with_rag = sys.argv[2]
if(with_rag.lower() == 'true'):
	rag_index_path = sys.argv[3]

def generate_response_realtime_internal_onnx_llm(message, history):
	text = message
	chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

	vector_db = None
	if(with_rag.lower() == 'true'):
		vector_db = FAISS.load_local(rag_index_path, hf_emb, allow_dangerous_deserialization = True)
		retriever = vector_db.as_retriever(search_kwargs = {'k':2}, verbose = True)
		search_result = retriever.invoke(text)
		context = search_result[0].page_content + '\n' + search_result[1].page_content
		text = """Answer the following question based only on the provided context:\n\n<context>""" + context + """</context>\n\nQuestion: """ + text

	#If there is a chat template, use it
	prompt = f'{chat_template.format(input = text)}'
	input_tokens = tokenizer.encode(prompt)

	params = og.GeneratorParams(model)
	params.input_ids = input_tokens
	generator = og.Generator(model, params)

	reply = ''
	while not generator.is_done():
		generator.compute_logits()
		generator.generate_next_token()
		
		new_token = generator.get_next_tokens()[0]
		reply = reply + tokenizer_stream.decode(new_token)
		yield reply

	del generator

def prompt_HF_LLM_endpoint(payload):
	headers = {"Authorization": f"Bearer {api_key}"}
	#response = requests.post(API_URL, headers = headers, json = payload, proxies = proxy_servers)
	response = requests.post(API_URL, headers = headers, json = payload)

def generate_response_static_external_llm(message, history):
	text = message
	vector_db = None
	if(with_rag.lower() == 'true'):
		vector_db = FAISS.load_local(rag_index_path, hf_emb, allow_dangerous_deserialization = True)
		retriever = vector_db.as_retriever(search_kwargs = {'k':2}, verbose = True)
		search_result = retriever.invoke(text)
		context = search_result[0].page_content + '\n' + search_result[1].page_content
		text = """Answer the following question based only on the provided context:\n\n<context>""" + context + """</context>\n\nQuestion: """ + text

	curr_parameters = {"return_full_text": False, "temperature": 0.3}
	reply = prompt_HF_LLM_endpoint({'inputs': text, 'parameters': curr_parameters})
	return reply

if(type.find('internal_onnx')):
	print ('Loading model')
	model = og.Model('') #SETUP: enter local Phi-3.5 onnx model path

	print ('Loading tokenizer')
	tokenizer = og.Tokenizer(model)
	tokenizer_stream = tokenizer.create_stream()

	print ('Starting CB')
	gr.ChatInterface(generate_response_realtime_internal_onnx_llm, type = 'messages').launch(server_name = '0.0.0.0', server_port = 7860)
else:
	gr.ChatInterface(generate_response_static_external_llm, type = 'messages').launch(server_name = '0.0.0.0', server_port = 7860)