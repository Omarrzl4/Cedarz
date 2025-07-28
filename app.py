import os

 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY or TAVILY_API_KEY in Hugging Face Secrets.")

 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
import gradio as gr

 
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
 
search_tool = TavilySearchResults()

 
tools = [
    Tool(
        name="Web Search",
        func=search_tool.run,
        description=(
             
    "You are Cedarz, a smart, friendly Lebanese tourism assistant helping tourists explore Lebanon 🇱🇧. "
    "Always respond with warmth, excitement, and **use emojis frequently** to make replies lively and engaging 🌟😄🌍. "
    "Use this web search tool only if you cannot answer from your own knowledge. "
    "Keep responses concise, relevant, and focused on tourism-related topics such as attractions 🏞️, culture 🎭, events 🎉, transportation 🚗, and local recommendations 🥙🗺️."
)

        )
    
]
 
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
) 
conversation_history = []
 
def build_agent_input(history, user_input, max_history=4):
    history = history[-max_history:]
    history_text = ""
    for usr, bot in history:
        history_text += f"Human: {usr}\nChatbot: {bot}\n"
    return f"{history_text}Human: {user_input}\nChatbot:"

 
def chatbot_response(user_input):
    if user_input.lower() in ["exit", "quit"]:
        conversation_history.clear()
        return "Safe travels! 🌍✈️"

    prompt = build_agent_input(conversation_history, user_input)
    response = agent.run(prompt)
    conversation_history.append((user_input, response))
    return response, ""
 
lebanon_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="red",
    neutral_hue="stone"
)

 
banner_html = """
<div style="text-align: center; margin-bottom: 20px;">
  <img src="https://i.imgur.com/KWeUkm8.jpeg"
     alt="Cedars of Lebanon"
     style="width: 100%; max-height: 250px; object-fit: cover; border-radius: 12px;" />
</div>
"""

with gr.Blocks(theme=lebanon_theme) as iface:
    gr.HTML(banner_html)
    gr.Markdown("## Cedarz: Your Lebanon Travel Assistant\nAsk about places, food, culture, or travel tips.")

    
    gr.Markdown("""
    ⚠️ **Note:** Guests (not logged in to Hugging Face) may be limited to one request.  
    👉 [Log in here for unlimited access](https://huggingface.co/login).
    """)
    
    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="Ask about Lebanon...", lines=2)

    chatbot_output = gr.Textbox(label="Cedarz Says")
    ask_button = gr.Button("Ask Cedarz")

    ask_button.click(
    fn=chatbot_response,
    inputs=user_input,
    outputs=[chatbot_output, user_input]
)


iface.launch()
