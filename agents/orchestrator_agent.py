from langchain_google_genai import ChatGoogleGenerativeAI

def create_orchestrator():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0
    )

    return llm