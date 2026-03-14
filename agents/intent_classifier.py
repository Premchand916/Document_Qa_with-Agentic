from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0
)

def classify_intent(state):

    query = state["query"]

    prompt = f"""
    Classify the user query into one of these intents:

    1. document_qa
    2. data_analysis
    3. summarization
    4. document_comparison

    Query: {query}

    Only return the intent name.
    """

    response = llm.invoke(prompt)

    state["intent"] = response.content.strip()

    return state