'''
Agentic RAG system:
1. Use Tavily API as a search tool to search the web
2. Bind the search tool to OpenAI GPT-4o-mini
3. Create a RAG chain to answer question using the retrieved context, 
   or call the search tool
4. Create a function to first retrieve context from Vector DB and use the
   RAG chain to check if the answer can be generated from the context
5. If the above step fails, execute the search tool to get new context from the web
6. Generate answer using the updated context from the tool call

'''
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create search tool and bind to LLM
@tool
def search_web(query: str) -> list:
    """Search the web for a query."""
    tavily_tool = TavilySearchResults(max_results=3, search_depth="advanced",
    	                              max_tokens=10000)

    results = tavily_tool.invoke(query)
    return [doc["content"] for doc in results]

tools = [search_web]
chatgpt = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
chatgpt_with_tools = chatgpt.bind_tools(tools)

# RAG chain
prompt = """Answer the question using the following context.
            If the answer is not in context, use the tools available
            to get the answer.

            Question: {question}
            Context: {context}
            Answer:
        """

prompt_template = ChatPromptTemplate.from_template(prompt)
qa_rag_chain = prompt_template | chatgpt_with_tools

# Agentic RAG workflow
def agentic_rag(question, context):
    tool_call_map = {"search_web": search_web}
    # Use any vector database here
    context = vectordb_retriever.invoke(query)
    response = qa_rag_chain.invoke({"context": context, "question": question})

    # If response content is present, then we have the answer from the context
    if response.content:
        print("Answer is in retrieved context")
        answer = response.content

    # If no response content present, then call search tool to get new context
    elif response.tool_calls:
        print("Answer is not in context, trying to use tools")
        tool_call = response.tool_calls[0]
        selected_tool = tool_call_map[tool_call["name"].lower()]
        print(f"Calling tool: {tool_call["name"]}")
        tool_output = selected_tool.invoke(tool_call[args])
        context = "\n\n".join(tool_output)
        response = qa_rag_chain.invoke({"context": context, "question": question})
        answer = response.content
    # If answer is also not found from web search
    else:
        answer = "Answer is not found"
    print(answer)

