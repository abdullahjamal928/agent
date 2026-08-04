from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES

load_dotenv(override=True)

# Ollama Client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"      # Any non-empty string
)

MODEL_NAME = "llama3.2:latest"

system = [
    {
        "role": "system",
        "content": TWIN_SYSTEM_PROMPT
    }
]


def chat(message, history):

    # history is already in "messages" format because of type="messages"
    messages = system + history + [
        {
            "role": "user",
            "content": message
        }
    ]

    while True:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
        )

        assistant_message = response.choices[0].message

        # No tool call → return answer
        if response.choices[0].finish_reason != "tool_calls":
            return assistant_message.content

        # Convert assistant message into a normal dictionary
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )

        tool_results = handle_tool_calls(assistant_message.tool_calls)

        messages.extend(tool_results)


if __name__ == "__main__":

    demo = gr.ChatInterface(
        fn=chat,
        type="messages",
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(
            show_label=False,
            type="messages"
        ),
        css=CSS,
        js=JS,
        theme=gr.themes.Base(),
    )

    demo.launch()