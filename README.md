# Classificador de imagens com CNN e reconhecimento ao vivo

Solucao pratica para o exercicio: voce cria um dataset proprio com imagens da webcam, treina uma CNN, avalia o resultado e depois usa o modelo salvo para reconhecer objetos em tempo real.

## Estrutura do projeto

```text
src/
  collect_dataset.py       # comando para montar o dataset pela webcam
  train_cnn.py             # comando para treinar a CNN
  evaluate_model.py        # comando para avaliar o modelo
  live_recognition.py      # comando para reconhecer ao vivo
  image_classifier/        # pacote reutilizavel para uma interface futura
    datasets.py            # carregamento e divisao dos dados
    inference.py           # carregamento do modelo e predicao
    metrics.py             # calculo de metricas
    model.py               # arquitetura da CNN
    plots.py               # graficos de treino e avaliacao
    settings.py            # caminhos e constantes do projeto
```

Uma interface futura, como Streamlit ou Gradio, deve importar funcoes de `image_classifier/` e evitar duplicar codigo dos scripts.

## 1. Preparar o ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Montar o dataset

Escolha de 3 a 5 classes, por exemplo: `caneca`, `celular`, `livro`.

Para coletar imagens de uma classe:

```powershell
python src\collect_dataset.py --class-name caneca --count 50
python src\collect_dataset.py --class-name celular --count 50
python src\collect_dataset.py --class-name livro --count 50
```

Controles da janela:

- `espaco`: salva uma foto.
- `a`: liga/desliga captura automatica.
- `q`: sai.

As imagens ficam em:

```text
dataset/
  caneca/
  celular/
  livro/
```

## 3. Treinar a CNN

```powershell
python src\train_cnn.py --epochs 15
```

Saidas geradas:

- `modelos\modelo_cnn.h5`
- `modelos\classes.json`
- `resultados\treinamento.png`

## 4. Avaliar

```powershell
python src\evaluate_model.py
```

Saidas geradas:

- `resultados\matriz_confusao.png`
- metricas exibidas no terminal

## 5. Reconhecer ao vivo

```powershell
python src\live_recognition.py
```

Controles:

- `q`: fecha a webcam.

O nome da classe prevista e a confianca aparecem sobre a imagem da webcam.

## Dicas para melhores resultados

- Use boa iluminacao.
- Varie angulos, distancia e fundo.
- Tire pelo menos 30 a 50 fotos por classe.
- Evite classes muito parecidas se o dataset for pequeno.
- Se a acuracia ficar baixa, colete mais imagens e aumente `--epochs`.

## Interface

Inicie a interface com:

```powershell
python -m streamlit run src\app.py
```

Se executar diretamente com `python src\app.py`, o projeto redireciona para o Streamlit automaticamente.

Na barra lateral, use `Atualizar cameras` e escolha a `Camera ativa`.

O menu da interface tem:

- `Visao geral`: resumo do dataset e do modelo.
- `Captura e dataset`: mostra a previa da camera antes de salvar, permite salvar uma foto, salvar a foto da previa ou coletar automaticamente.
- `Treino`: treina a CNN usando o dataset atual.
- `Avaliacao`: calcula metricas e mostra a matriz de confusao.
- `Reconhecimento`: mostra a camera com ou sem classificacao ao vivo.



## Problemas com camera

Se a interface mostrar que nenhuma camera foi detectada, o problema esta no acesso do OpenCV/Windows a webcam, nao no modelo CNN. Tente:

- Fechar Teams, Zoom, navegador ou outro app que possa estar usando a camera.
- Conferir as permissoes em Configuracoes do Windows > Privacidade e seguranca > Camera.
- Testar outro indice em `Indice manual da camera` na barra lateral.
- Aumentar `Buscar ate o indice` e clicar em `Atualizar cameras`.
- Conectar uma webcam USB se o computador nao tiver camera local disponivel.

## Webcam na interface

Na barra lateral, em `Fonte da imagem`, escolha:

- `Webcam do navegador`: usa a permissao de camera do browser. E a melhor opcao quando o OpenCV nao detecta cameras por indice.
- `Camera local por indice`: usa OpenCV com indices como 0, 1, 2. Use quando quiser captura continua/local pelo Python.

Para captura de dataset pela webcam do navegador, preencha a classe e tire a foto no componente `Webcam`; a imagem sera salva automaticamente no dataset da classe. Para reconhecimento, tire uma foto em `Webcam para reconhecimento` e o modelo classifica a imagem capturada.
