import torch

# Testa se CUDA está disponível
if torch.cuda.is_available():
    print("CUDA disponível. Treinamento será realizado na GPU.")
else:
    print("CUDA não disponível. Treinamento será realizado na CPU.")

# Teste simples de operação na GPU
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print(f"Tenso convertido para CUDA: {x}")


# Carregando modelo básico da EleutherAi
from transformers import GPTNeoForCausalLM, AutoTokenizer

# Configuração básica do logger
import logging

logging.basicConfig(
    filename='project.log',  # Nome do arquivo de log
    level=logging.INFO,      # Nível do log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato da mensagem
)

def main():
    try:
        logging.info("Iniciando o carregamento do modelo EleutherAI")
        model_name = "EleutherAI/gpt-neo-1.3B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name).to("cuda")
        logging.info(f"Modelo {model_name} carregado com sucesso")

        # Exemplo de inferência
        input_text = "Hello, world!"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=50)
        generated_text = tokenizer.decode(outputs[0])
        logging.info(f"Texto gerado: {generated_text}")

    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
