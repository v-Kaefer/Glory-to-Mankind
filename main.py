import torch

# Testa se CUDA está disponível
if torch.cuda.is_available():
    print("CUDA disponível. Treinamento será realizado na GPU.")
else:
    print("CUDA não disponível. Treinamento será realizado na CPU.")

# Teste simples de operação na GPU
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print(f"Tenso convertido para CUDA: {x}")
