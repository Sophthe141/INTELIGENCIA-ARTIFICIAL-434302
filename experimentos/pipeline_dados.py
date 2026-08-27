import re

# 1. Definimos um texto de teste
#texto_bruto = "Olá, mundo! Este é o nosso primeiro teste de tokenização."
texto_bruto = "token1 token2 token3 teste"
# 2. Usamos uma expressão regular (regex) para separar palavras e pontuações
# Essa regra diz: "corte o texto sempre que encontrar um espaço OU uma pontuação"
resultado_bruto = re.split(r'([,.:;?_!"()\']|--|\s)', texto_bruto)

# 3. Limpamos a lista removendo espaços vazios indesejados
tokens = [item for item in resultado_bruto if item.strip()]

print("--- RESULTADO DA TOKENIZAÇÃO ---")
print(f"Texto original: {texto_bruto}")
print(f"Quantidade de tokens: {len(tokens)}")
print(f"Lista de tokens: {tokens}")