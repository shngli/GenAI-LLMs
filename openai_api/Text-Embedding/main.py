"""
Provide text embedding about unicorn startups to generate context for OpenAI 
about startup company from the user input (eg. Tell me about company [blank] and who invest in it)
"""
import openai 
import numpy as np
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")

### Read CSV file containing pre-generated summary embeddings about some unicorn startups
df = pd.read_csv('./data/unicorns_with_embeddings.csv')

### Note this function assumes we already set our Open AI key
def get_embedding(text):
    
    result = openai.Embedding.create(
      model='text-embedding-ada-002',
      input=text
    )
    return result["data"][0]["embedding"]

### Note: There are vector search services for larger amount of vectors eg. Pinecone or Weaviate
def vector_similarity(vec1,vec2):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(vec1), np.array(vec2))

def embed_prompt_lookup():
    ### Input initial user question
    question = input("What question do you have about a Unicorn company? ")
    
    ### Create prompt embedding from the user input
    prompt_embedding = get_embedding(question)
    
    ### Get prompt similarity with embeddings from the startup summaries
    ### Note how this will overwrite the prompt similarity column each time!
    df["prompt_similarity"] = df['embedding'].apply(lambda vector: vector_similarity(vector, prompt_embedding))

    ### Get most similar summary to the prompt embedding
    summary = df.nlargest(1,'prompt_similarity').iloc[0]['summary'] 

    ### Submit context about startup company in user prompt to OpenAI Completion API
    prompt = f"""Only answer the question below if you have 100% certainty of the facts, use the context below to answer.
            Here is some context:
            {summary}
            Q: {question}
            A:"""

    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=500,
        model="text-davinci-003"
    )
    
    ### Print OpenAI response about the startup company that user asked about
    print(response["choices"][0]["text"].strip(" \n"))

if __name__ == "__main__":
    embed_prompt_lookup()
