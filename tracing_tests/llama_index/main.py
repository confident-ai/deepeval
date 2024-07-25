from chatbot import query

while True:
    user_input = input("Enter your question: ")
    response = query(user_input)
    print("Bot response:", response)
