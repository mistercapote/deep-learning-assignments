# AI_LOG.md — Registro de Uso de Inteligência Artificial

> **Programming Assignment 1 (PA1): Segmentação de Instâncias**  
> **Disciplina:** Aprendizado Profundo (Deep Learning) — FGV EMAp  
> **Professor:** Dario Oliveira | **Monitor:** Erick Brito  
> **Equipe:** Jaime Willian Carneiro da Silva e Luiz Eduardo Bravin  
> **Data:** Setembro de 2026

---

## 1. Política de Uso e Filosofia da Equipe
Conforme estipulado no item 5 das diretrizes do assignment (*"Uso de IA é permitido e esperado. O que não é permitido é entregar algo que vocês não entendem"*), a equipe utilizou ferramentas de IA generativa de ponta como aceleradores de produtividade de código, depuração de erros sintáticos, geração de scaffolds de arquiteturas e implementação de funções utilitárias repetitivas (como plotagens). 

Em nenhum momento as decisões de modelagem, as escolhas matemáticas (design de representações de saída, formulação de perdas e decodificação pós-processamento) ou a interpretação analítica dos experimentos foram delegadas à IA sem validação técnica minuciosa.

---

## 2. Ferramentas Utilizadas
Durante as fases de concepção, codificação, ablações e análise de resultados, recorremos a três grandes ecossistemas de LLMs:
- **Google Gemini (Gemini 1.5 Pro / Advanced):** Utilizado primordialmente para geração e refatoração das classes de redes neurais em PyTorch, planejamento de testes de estresse e suporte à infraestrutura de hardware.
- **Anthropic Claude (Claude 3.5 Sonnet):** Utilizado na construção das lógicas de pós-processamento (matching guloso de instâncias, watershed marcado e fusão de tiles em mosaico) e resolução de bugs lógicos de indexação de tensores.
- **DeepSeek (DeepSeek-V2 / Coder):** Empregado na depuração de rotinas de pré-processamento de dados de bioimagem, operações com NumPy/SciPy (`ndimage`) e otimização de I/O em disco.

---

## 3. Principais Casos de Uso e Episódios Técnicos

### 3.1. Geração das Classes e Módulos das Arquiteturas
Para a execução da Baseline (Parte 1) e das Ablações arquiteturais (Parte 3 — Eixos 1 e 3), solicitamos às IAs a estruturação inicial das classes de modelos modulares em PyTorch (`torch.nn.Module`). As arquiteturas geradas e ajustadas pela equipe foram:
- `UNetBinary`: U-Net binária com encoder backbone ResNet-34 para a baseline da Parte 1.
- `UNetTernary`: U-Net adaptada para a Trilha A (3 canais de saída para classificação ternária: fundo, interior e fronteira; e 1 canal de regressão contínua com ativação Sigmoid para o mapa de distância ao fundo).
- `DeepLabTernary`: Arquitetura do Eixo 1 baseada em convoluções atrosas e módulo **ASPP** (*Atrous Spatial Pyramid Pooling*), mantendo a mesma espinha dorsal ResNet-34 sem skip connections diretas.
- `SegNetTernary`: Arquitetura do Eixo 1 que recupera resolução espacial através do armazenamento e desempilhamento de índices de pooling (**Max Unpooling** / *pool indices*).
- `ParseNetTernary`: Arquitetura do Eixo 3 com módulo de contexto global por *Image Pooling* (`ParseModule`) acoplado ao gargalo (*bottleneck*).
- `PSPNetTernary`: Arquitetura do Eixo 3 com *Pyramid Pooling Module* (**PPM**) em múltiplas escalas (1x1, 2x2, 3x3, 6x6) interpoladas e concatenadas no decoder.
- `UNetTernaryDilated`: Variante implementada para a Galeria de Falhas (Parte 5), aplicando convoluções dilatadas nas camadas finais do encoder para expansão do campo receptivo teórico sem perda de resolução.

### 3.2. Funções de Visualização e Gráficos (Matplotlib)
A manipulação de layouts complexos no `matplotlib` (gráficos de dispersão com múltiplos eixos, matrizes de amostras comparando original, máscaras de instâncias coloridas e mapas de contorno/distância) é notoriamente verbosa e propensa a erros manuais de formatação.
- Usamos Gemini e Claude para gerar as funções de plotagem estruturadas (`plot_metrics`, `plot_samples_binary`, `plot_samples_ternary`, `plot_ternary_masks`, `plot_training_curves_binary`, `plot_training_curves_ternary`).
- Coube à equipe realizar o fine-tuning estético e funcional: definição do colormap categórico `ListedColormap` (cores padronizadas para fundo, interior e fronteira), aplicação de colormaps com máscara de fundo preto (`prism.set_bad('black')`) e dimensionamento dos subplots.

### 3.3. Otimização de Treinamento e Pipeline de Dados (.pt)
O carregamento repetido de centenas de imagens PNG de alta resolução e múltiplas máscaras por amostra do dataset DSB2018 gerava um gargalo severo de I/O por época de treino.
- Solicitamos à IA estratégias para acelerar a leitura dos dados. A IA recomendou e estruturou um mecanismo de cache duplo (memória RAM via dicionário e disco via arquivos serializados `.pt` por tamanho de imagem).
- Com a conversão antecipada dos arrays pré-processados e tensores para arquivos `cache_{img_size}.pt` e `cache_{img_size}_ternary.pt` via `torch.save`/`torch.load`, o tempo de iteração por época foi drasticamente reduzido.

### 3.4. Resolução de Hardware: Suporte à GPU Intel Arc Graphics (XPU)
Durante o setup do ambiente de execução, deparamos com o desafio de rodar o treinamento acelerado por hardware em uma GPU dedicada **Intel Arc Graphics**. Por padrão, rotinas típicas de PyTorch buscam exclusivamente `torch.cuda.is_available()`.
- Demoramos um tempo considerável para conseguir alocar os tensores e modelos na GPU Intel até descobrirmos, com auxílio do Gemini e pesquisas técnicas, que era necessário instalar a extensão de aceleração da Intel para PyTorch (`intel-extension-for-pytorch` ou suporte nativo `torch.xpu`).
- Ajustamos o código de inicialização global de dispositivos para alternar de forma transparente:
  ```python
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
  ```
  Isso permitiu treinar com aceleração de hardware nativa na Intel Arc com performance expressivamente superior à CPU.

### 3.5. Correção de Alucinação / Sugestão Inadequada da IA: Pesos das Classes na Loss
Um episódio crucial em que a intervenção analítica da equipe foi indispensável ocorreu na formulação da função de perda ponderada para a classificação ternária (Trilha A).
- A IA (especialmente Claude e DeepSeek) insistia repetidamente em implementar funções dinâmicas complexas para calcular pesos das classes inversamente proporcionais às frequências de pixel em cada batch (ou via raiz quadrada do inverso da frequência).
- **Problema encontrado:** Ao calcular pesos dinamicamente ou usar fórmulas automáticas desproporcionais, a classe de fronteira (minoritária) recebia penalizações instáveis ou o fundo dominava o gradiente de forma errática, fazendo com que o modelo colapsasse ou apresentasse métricas de IoU e mAP insatisfatórias.
- **Nossa solução:** Rejeitamos a sugestão da IA e testamos experimentalmente a substituição da função dinâmica por um tensor constante fixo e controlado:
  ```python
  class_weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
  ```
  Ao fixar o peso da fronteira em `2.0` de maneira constante em relação ao fundo e interior, o treinamento convergiu de maneira suave, estabilizando as probabilidades preditas e alcançando um desempenho de detecção de instâncias significativamente superior.

### 3.6. Correções Contínuas de Bugs e Refatorações
Ao longo do desenvolvimento, as IAs atuaram como parceiras de depuração imediata para:
- Ajustar incompatibilidades de dimensões em concatenações de tensores com canais diferentes (`torch.cat`).
- Corrigir o cálculo vetorizado da matriz de IoU entre instâncias previstas e ground-truth no algoritmo de matching guloso em `evaluating.py`.
- Lidar com casos de borda na decodificação de watershed quando não havia sementes de interior (`labeled_markers.max() == 0`).
- Implementar padding reflexivo e janelamento deslizante na inferência em mosaico para imagens cujas dimensões não eram múltiplos perfeitos do tile size.

---

## 4. Declaração de Autoria do Log
Em conformidade com a transparência exigida pelo Assignment e pela ética acadêmica, declaramos que **este arquivo `AI_LOG.md` também foi gerado com auxílio de Inteligência Artificial**, tendo sua estrutura, episódios técnicos e redação sintetizados a partir das solicitações e experiências vivenciadas pela dupla ao longo do projeto.
