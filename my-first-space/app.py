import gradio as gr

def respond(message, history):
    response = f'You said: {message}\
    \nAnd I said: I love AI!'

    return response

gr.ChatInterface(fn = respond).launch()