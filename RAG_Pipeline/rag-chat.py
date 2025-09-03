import os
import chromadb

from openai import AzureOpenAI

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

model_name = 'gpt-4o-mini' # deployment name
tesla_10k_collection = 'tesla-10k-2019-to-2023'
azure_api_key = os.getenv('azure_api_key')
# Modify the Azure Endpoint and the API Versions as needed
azure_base_url = os.getenv('azure_base_url')
azure_api_version = os.getenv('azure_api_version')

client = AzureOpenAI(
  azure_endpoint = azure_base_url,
  api_key = azure_api_key,
  api_version = azure_api_version
)

embedding_model = AzureOpenAIEmbeddings(
    api_key=azure_api_key,
    azure_endpoint=azure_base_url,
    api_version=azure_api_version,
    azure_deployment="text-embedding-3-small"
)

chromadb_client = chromadb.PersistentClient(
    path="./tesla_db"
)

vectorstore_persisted = Chroma(
    collection_name=tesla_10k_collection,
    collection_metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_model,
    client=chromadb_client,
    persist_directory="./tesla_db"
)

retriever = vectorstore_persisted.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

qna_system_message = """
You are an assistant to a financial services firm who answers user queries on annual reports.
User input will have the context required by you to answer user queries.
This context will be delimited by: <Context> and </Context>.
The context contains references to specific portions of a document relevant to the user query.

User queries will be delimited by: <Question> and </Question>.

Please answer user queries only using the context provided in the input.
Do not mention anything about the context in your final answer. Your response should only contain the answer to the question.

If the answer is not found in the context, respond "I don't know".
"""

qna_user_message_template = """
<Context>
Here are some documents that are relevant to the question mentioned below.
{context}
</Context>

<Question>
{question}
</Question>
"""

def respond(user_query):
    relevant_document_chunks = retriever.invoke(user_query)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = "\n---\n".join(context_list)

    prompt = [
        {'role': 'developer', 'content': qna_system_message},
        {
            'role': 'user', 'content': qna_user_message_template.format(
             context=context_for_query,
             question=user_query)
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=0
        )

        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f'Sorry, I encountered the following error: \n {e}'

    return answer

def main():
    """
    Runs the main interactive loop for the Q&A system.

    This function initializes the conversation history, continuously prompts
    the user for queries, processes the queries using the `respond` function,
    and displays the assistant's responses. It also maintains the
    conversation history for context.

    Args:
        None

    Returns:
        None
    """

    # 1. Initialize conversation history.
    # This list stores the conversation between the user and the assistant.
    # It starts with a system message introducing the assistant's role.
    conversation_history = [
        {'role': 'developer', 'content': 'You are a helpful assistant who answers queries on financial documents'}
    ]

    # 2. Enter the interactive loop.
    # The loop continues until the user enters 'q' to quit.
    while True:
        # 2.1 Get user input.
        # Prompt the user to enter a query and store it in `user_query`.
        user_query = input("User (type q to quit): ")

        # 2.2 Check for quit condition.
        # If the user enters 'q', break out of the loop.
        if user_query == 'q':
            break

        # 2.3 Process the query and get the answer.
        # Call the `respond` function to process the user query and get the answer.
        answer = respond(user_query)

        # 2.4 Update conversation history.
        # Add the user's query and the assistant's answer to the conversation history.
        conversation_history.append({'role': 'user', 'content': user_query})
        conversation_history.append({'role': 'assistant', 'content': answer})

        # 2.5 Display the assistant's answer.
        # Print the assistant's answer to the console.
        print(f"Assistant: {answer}")

if __name__ == "__main__":
    main()
