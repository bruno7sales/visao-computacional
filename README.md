# Classificador de Imagens com CNN

Aplicação de visão computacional para criação de dataset próprio, treinamento de uma rede neural convolucional (CNN), avaliação do modelo e reconhecimento de objetos pela webcam.

O projeto foi desenvolvido em Python com TensorFlow, OpenCV e Streamlit. Ele permite coletar imagens de diferentes classes, treinar um modelo supervisionado e usar o modelo salvo para classificar novas imagens em tempo real ou por captura no navegador.

## Funcionalidades

- Coleta de imagens pela webcam para montar um dataset local.
- Organização automática das imagens por classe.
- Treinamento de uma CNN com aumento de dados.
- Separação automática entre treino, validação e teste.
- Avaliação do modelo com métricas e matriz de confusão.
- Reconhecimento por foto capturada no navegador.
- Reconhecimento ao vivo com webcam usando Streamlit WebRTC.
- Interface web em Streamlit para executar o fluxo completo.

## Tecnologias

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Streamlit WebRTC

## Estrutura do projeto

```text
src/
  app.py                    # interface web em Streamlit
  collect_dataset.py         # coleta imagens pela webcam local
  train_cnn.py               # treina a CNN
  evaluate_model.py          # avalia o modelo treinado
  live_recognition.py        # reconhecimento ao vivo via OpenCV
  image_classifier/
    camera.py                # captura, conversão e salvamento de frames
    datasets.py              # leitura e preparação do dataset
    inference.py             # carregamento do modelo e predição
    metrics.py               # cálculo de métricas
    model.py                 # arquitetura da CNN
    plots.py                 # geração de gráficos
    settings.py              # caminhos e configurações globais
```

Durante a execução, o projeto cria diretórios locais para armazenar dados, modelos e resultados:

```text
dataset/      # imagens organizadas por classe
modelos/      # modelo treinado e arquivo de classes
resultados/   # gráficos de treino e avaliação
```

## Como funciona

O fluxo principal é dividido em quatro etapas:

1. Criar um dataset com imagens de pelo menos duas classes.
2. Treinar a CNN com as imagens coletadas.
3. Avaliar o desempenho do modelo.
4. Usar o modelo treinado para reconhecer novas imagens pela webcam.

As imagens do dataset são lidas com TensorFlow, redimensionadas para `128x128` pixels e processadas em lotes. As imagens capturadas pela webcam são lidas com OpenCV, convertidas de BGR para RGB e redimensionadas antes da predição.

## Requisitos

- Python 3.10 ou superior
- Webcam local ou webcam disponível no navegador
- Ambiente virtual recomendado

## Instalação

Clone o repositório e acesse a pasta do projeto:

```powershell
git clone <url-do-repositorio>
cd visao-computacional
```

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Executando pela interface web

Para abrir a aplicação Streamlit:

```powershell
python -m streamlit run src\app.py
```

Também é possível executar diretamente:

```powershell
python src\app.py
```

A interface possui as seguintes telas:

- `Visão geral`: resumo do dataset e do estado do modelo.
- `Captura e dataset`: captura imagens pela webcam do navegador.
- `Treino`: treina a CNN com o dataset atual.
- `Avaliação`: avalia o modelo e exibe resultados.
- `Reconhecimento`: classifica imagens capturadas pela webcam.

## Uso pelo terminal

### 1. Coletar imagens

Crie imagens para cada classe desejada:

```powershell
python src\collect_dataset.py --class-name caneca --count 50
python src\collect_dataset.py --class-name celular --count 50
python src\collect_dataset.py --class-name livro --count 50
```

Controles da janela de coleta:

- `Espaço`: salva uma foto.
- `A`: ativa ou desativa a captura automática.
- `Q`: encerra a coleta.

As imagens serão salvas em:

```text
dataset/
  caneca/
  celular/
  livro/
```

### 2. Treinar o modelo

```powershell
python src\train_cnn.py --epochs 15
```

Arquivos gerados:

```text
modelos/modelo_cnn.h5
modelos/classes.json
resultados/treinamento.png
```

### 3. Avaliar o modelo

```powershell
python src\evaluate_model.py
```

O comando exibe métricas no terminal e gera a matriz de confusão em:

```text
resultados/matriz_confusao.png
```

### 4. Reconhecer objetos ao vivo

```powershell
python src\live_recognition.py
```

Pressione `Q` para encerrar a janela da webcam.

## Arquitetura da CNN

O modelo usa uma arquitetura convolucional simples para classificação multiclasse:

- Entrada com imagens `128x128x3`.
- Aumento de dados com flip horizontal, rotação e zoom.
- Normalização dos pixels.
- Três blocos convolucionais com `Conv2D` e `MaxPooling2D`.
- Camadas densas com `Dropout`.
- Saída `softmax` com uma classe por objeto cadastrado.

O treinamento utiliza:

- Otimizador: `Adam`
- Função de perda: `sparse_categorical_crossentropy`
- Métrica principal: `accuracy`

## Dicas para melhores resultados

- Capture pelo menos 30 a 50 imagens por classe.
- Use boa iluminação.
- Varie ângulos, distância, posição e fundo.
- Evite classes visualmente muito parecidas em datasets pequenos.
- Refaça o treinamento sempre que adicionar novas classes ou muitas imagens.
- Aumente o número de épocas caso a acurácia ainda esteja baixa.

## Solução de problemas

### A webcam não abre

Verifique se outro aplicativo está usando a câmera, como navegador, Teams ou Zoom. Também confira as permissões de câmera do Windows.

### O OpenCV não encontra a câmera

Use a captura pela interface Streamlit, que acessa a webcam pelo navegador. Essa opção costuma funcionar melhor quando o acesso direto via OpenCV encontra conflitos com o Windows.

### O reconhecimento não está disponível

Treine o modelo antes de usar a tela de reconhecimento. O projeto precisa dos arquivos `modelos/modelo_cnn.h5` e `modelos/classes.json`.

### A acurácia ficou baixa

Colete mais imagens, varie melhor as condições de captura e garanta que cada classe tenha exemplos suficientes.

## Licença

Este projeto pode ser usado para fins acadêmicos e de estudo.
