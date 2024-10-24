# RAG script to extract PDF file content
import os
import tempfile
import streamlit as st
from embedchain import App

def embedchain_bot(db_path, api_key):
	return App.from_config(
		config={
		    "llm": {"provider": "openai", "config": {"api_key": api_key}},
		    "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
		    "embedder": {"provider": "openai", "config": {"api_key": api_key}}
		}
	)

# Set the title
st.title("Chat with PDF")

# Get OpenAI API key as input
openai_access_token = st.text_input("OpenAI API Key", type="password")

if openai_access_token:
	# Create a temporary directory for the vector database
	db_path = tempfile.mkdtemp()

	# Initialize the Embedchain app with the temp directory and OpenAI API Key
	app = embedchain_bot(db_path, openai_access_token)

	# Allow user to upload a PDF doc
	pdf_file = st.file_uploader("Upload PDF file", type="pdf")

	if pdf_file:

		# Create a temporary file to store the uploaded PDF doc
		with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:

			# Write the contents of the uploaded PDF doc to the temp file
			f.write(pdf_file.getvalue())
			temp_pdf_path = f.name

		# Add the temp file to the knowledge base
		app.add(temp_pdf_path, data_type="pdf_file")

		# Remove the temp file
		os.remove(temp_pdf_path)

		# Display a success message
		st.success(f"Addded {pdf_file.name} to knowledge base")

		# Get user question as input
		prompt = st.text_input("Ask a question about the PDF document")

		if prompt:
			
			# Get the answer from the Embedchain app
			answer = app.chat(prompt)

			# Display answer
			st.write(answer)
