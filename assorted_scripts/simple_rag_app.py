import torch
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings

warnings.simplefilter("ignore")

# Sample documents (knowledge base)
documents = [
    "Humanitarian crises often arise from conflicts, natural disasters, and pandemics, leaving millions in need of urgent assistance.",
    "The United Nations organizes rapid response teams to deliver food, water, and medical aid to areas affected by humanitarian emergencies.",
    "Refugee camps are established by organizations like UNHCR to shelter and protect displaced populations during humanitarian crises.",
    "Humanitarian aid workers face numerous challenges, including dangerous environments, logistical constraints, and limited resources in crisis zones.",
    "In response to severe droughts, humanitarian organizations provide clean drinking water and agricultural support to affected communities.",
    "The International Red Cross plays a crucial role in providing medical assistance and emergency relief during armed conflicts and natural disasters.",
    "The United Nations World Food Programme (WFP) delivers food aid to millions of people suffering from hunger due to humanitarian crises.",
    "Humanitarian interventions are often coordinated by international agencies to ensure that aid reaches the most vulnerable populations quickly.",
    "During the COVID-19 pandemic, humanitarian efforts focused on distributing personal protective equipment, vaccines, and healthcare services to affected regions.",
    "Disaster response teams are deployed immediately after earthquakes and floods to rescue survivors and provide temporary shelter and medical care."
]

# Create a TF-IDF vectorizer and fit it on the documents
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents).toarray()

# Create a FAISS index for fast similarity search
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_vectors, dtype=np.float32))

def retrieve(query, k=1):
    # Vectorize the query
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    
    # Retrieve the top k documents
    distances, indices = index.search(query_vector, k)
    
    # Return the retrieved documents
    return [documents[i] for i in indices[0]]

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_text(prompt):
    # Encode the prompt and generate text
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    outputs = model.generate(inputs,
                             max_length=150,
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             attention_mask=attention_mask,
                             early_stopping=True)
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def rag_query(query):
    # Step 1: Retrieve relevant documents
    retrieved_docs = retrieve(query, k=2)
    context = " ".join(retrieved_docs)
    
    # Step 2: Generate a response based on the retrieved context
    prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    response = generate_text(prompt)
    
    return response

if __name__ == "__main__":
    print("\n\n")
    query = "What challenges do humanitarian organizations face during crisis response?"
    response = rag_query(query)
    print("\n", response, "\n")