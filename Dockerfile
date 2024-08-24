# Use a imagem oficial do PyTorch com CUDA
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Instale Python e dependências do sistema
RUN apt-get update && apt-get install -y python3 python3-pip

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo de dependências e instale-as
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copie o código para o diretório de trabalho
COPY . .

# Comando para rodar o script Python
CMD ["python3", "main.py"]
