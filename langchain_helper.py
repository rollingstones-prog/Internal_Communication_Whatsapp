from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os


# Initialize AI model and memory
def get_ai_response(user_text, session_memory=None):
    """Generate AI reply using LangChain + OpenAI."""

    # Keep memory persistent across messages if provided
    memory = session_memory or ConversationBufferMemory()

    llm = OpenAI(
        temperature=0.7,
        model_name="gpt-4o-mini",  # Fast & affordable, good for chatbots
        openai_api_key=os.getenv("OPENAI_API_KEY"))

    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    reply = conversation.predict(input=user_text)
    return reply, memory
