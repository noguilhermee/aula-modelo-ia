import gradio as gr
from transformers import pipeline, Conversation

# Carregar o modelo do Hugging Face
modelo = pipeline("conversational", model="facebook/blenderbot-400M-distill")

# Criar um histórico de conversa
historico = Conversation()

def responder(mensagem):
    global historico
    historico.add_user_input(mensagem)
    resposta = modelo(historico)
    return resposta.generated_responses[-1]

# Criar a interface do chatbot
iface = gr.Interface(fn=responder, inputs="text", outputs="text")
iface.launch(share=True)  # O parâmetro "share=True" gera um link público