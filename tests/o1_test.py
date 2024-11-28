import langchain_openai
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Cannot currently set settings such as max tokens, temperature, and top p
llm = langchain_openai.ChatOpenAI(api_key=api_key, model="o1-mini")

response = llm.invoke([HumanMessage(content="test")])

print(response)
