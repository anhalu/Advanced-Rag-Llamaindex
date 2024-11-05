import os
from openai import OpenAI
from termcolor import colored
from pprint import pprint


client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL')
)

def main():
    # Conversation history is needed to maintain context in a chat
    conversation_history = []

    print(colored("Chatbot: Hello! I am here to help you. Type 'exit' to end the chat.", 'cyan'))
    while True:
        # Read user input
        user_input = input(colored("You: ", 'green'))

        # Exit condition
        if user_input.lower() == 'exit':
            print(colored("Chatbot: Goodbye!", 'cyan'))
            break

        # Add user message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # try:
        # Get response from the model
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",  # Replace this with your model name
            messages=conversation_history,
        )

        assistant_reply = response.choices[0].message.content
        
        print(colored(f"Chatbot: {assistant_reply}", 'cyan'))

        # Add assistant reply to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Print conversation history for debugging or logging purposes
        pprint(conversation_history)

if __name__ == "__main__":
    main()
