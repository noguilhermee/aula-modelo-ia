import json
import pickle
import numpy as np

# Caminho do modelo dentro do contêiner
modelo_path = "/var/task/modelo_fraude.pkl"

# Carregar o modelo treinado no contêiner
def carregar_modelo():
    with open(modelo_path, "rb") as f:
        return pickle.load(f)

modelo = carregar_modelo()

def lambda_handler(event, context):
    try:
        # Recebe os dados da requisição
        dados = json.loads(event["body"])
        entrada = np.array([[dados["valor_transacao"], dados["tempo_conta"], dados["num_transacoes_ult_30d"], dados["pais_origem"]]])

        # Faz a predição
        resultado = modelo.predict(entrada)

        return {
            "statusCode": 200,
            "body": json.dumps({"fraude": int(resultado[0])})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }