# AI_LOG.md — Registro de Uso de Inteligência Artificial

> **Programming Assignment 3 (PA3): Classificação de Fauna em Câmeras Trap com Poucos Rótulos (Trilha A — SimCLR)**
> **Disciplina:** Aprendizado Profundo (Deep Learning) — FGV EMAp
> **Professor:** Dario Oliveira | **Monitor:** Erick Brito
> **Equipe:** Jaime Willian Carneiro da Silva e Luiz Eduardo Bravin
> **Data:** Novembro de 2026

---

## 1. Política de Uso e Filosofia da Equipe

Conforme estipulado nas diretrizes do assignment (*"Uso de IA é permitido e esperado. O que não é permitido é entregar algo que vocês não entendem"*), a equipe utilizou ferramentas de IA generativa como aceleradores de desenvolvimento de código, apoio na modularização de scripts em PyTorch, depuração de erros de ambiente e geração de visualizações padronizadas em `matplotlib` e `seaborn`.

Todas as decisões experimentais — incluindo a escolha da Trilha A (SimCLR), a formulação da hipótese de aumentação de *crop* para animais pequenos no quadro, o controle estrito de vazamento de dados (*data leakage*) entre treino/validação/teste, a avaliação multi-seed e a interpretação crítica dos resultados na curva de orçamento — foram concebidas, auditadas e validadas pela equipe.

---

## 2. Ferramentas Utilizadas

Durante a implementação, execução dos experimentos e preparação da apresentação, utilizamos ferramentas de IA generativa para tarefas específicas:

* **Google Gemini:** Auxílio na estruturação da pipeline de pré-treino auto-supervisionado (SimCLR com loss NT-Xent), suporte à execução local em GPU Intel Arc (`xpu`), validação dos entregáveis obrigatórios do notebook, análise comparativa dos resultados da Parte 2 e roteirização da apresentação oral.
* **Anthropic Claude:** Apoio na refatoração modular do código entre o notebook e os módulos `src/redimensionamento.py` e `src/autosup.py`, padronização das rotinas de avaliação linear e fine-tuning, e depuração de erros de indexação e reprodutibilidade.


* **DeepSeek:** Empregado em ajustes pontuais de pré-processamento de imagens com OpenCV/PyTorch, verificação de integridade de arquivos do dataset e otimização de DataLoaders.

---

## 3. Principais Casos de Uso e Episódios Técnicos

### 3.1. Estruturação Modular do Pipeline (Piso e SimCLR)

Para manter o notebook limpo e focado na orquestração dos experimentos, utilizamos a IA para estruturar e refatorar as funções auxiliares em dois módulos Python:

* `src/redimensionamento.py`: Implementação do `DatasetTrap`, transformações de treino e validação (`transform_treino`, `transform_val`), arquitetura baseline `CNN_pequena` e loops de treinamento e avaliação (`train_epoch`, `eval_model`).


* `src/autosup.py`: Implementação completa da Trilha A, contendo a criação de pares de *views* (`criar_augment_simclr`), construção de encoders e classificadores (`criar_encoder`, `criar_classificador`), loop de pré-treino contrastivo (`pretreino_simclr` com suporte a *Automatic Mixed Precision* — `AMP`), avaliação linear com busca de hiperparâmetro `C` na validação (`avaliacao_linear`) e controle determinístico de sementes (`fixar_seed`).



### 3.2. Geração de Gráficos e Visualizações Padronizadas

A criação de figuras com múltiplos eixos, escalas logarítmicas e formatação limpa para os slides foi acelerada com auxílio de IA:

* **Parte 0:** Subplots empilhados de *Train/Val Loss* e *Val F1 macro* com marcação vertical automática da melhor época de validação (época 7 na ResNet-18), além da matriz de confusão normalizada por linha das 21 classes.


* **Parte 1:** Curva comparativa da loss `NT-Xent` entre pré-treino partindo do ImageNet vs. pesos aleatórios e grade visual das *views* geradas pelos diferentes pisos de `RandomResizedCrop`.


* **Parte 2:** Gráfico obrigatório da **Curva de Orçamento** (`curva_orcamento.png`) com eixo X em escala logarítmica (`[1, 5, 20]`), barras de erro de desvio padrão entre as 3 *seeds* no orçamento 5 e linha de referência do teto supervisionado (`0.686`), além do gráfico divergente de **Ganho Médio de F1 por Classe** (`ganho_f1_por_classe.png`).

### 3.3. Infraestrutura, Portabilidade e Suporte à GPU Intel Arc (`xpu`)

Assim como nos trabalhos anteriores, o desenvolvimento alternou entre execução em nuvem (Kaggle) e execução local acelerada em GPU Intel Arc (`xpu`):

* A IA auxiliou na centralização de caminhos via `pathlib.Path` e na detecção automática de hardware:
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')

```


* Isso permitiu rodar o pré-treino do SimCLR nas 6.505 imagens não rotuladas por 20 épocas com `AMP = True` em tempo viável.



### 3.4. Intervenções Críticas e Correções da Equipe sobre Sugestões/Rascunhos Iniciais

Diversos problemas metodológicos e de código exigiram intervenção direta da equipe para garantir validade científica aos resultados:

1. **Descasamento de resolução e subtreinamento no SimCLR:**
Em versões iniciais do pipeline, o pré-treino rodava com imagens reduzidas (`LADO_SSL = 160`) e apenas `EPOCAS_SSL = 3` para testes rápidos, o que impedia a convergência da loss NT-Xent e criava um descasamento de resolução com os `224×224` usados no fine-tuning e na avaliação. A equipe corrigiu os parâmetros para `LADO_SSL = 224` e `EPOCAS_SSL = 20`, além de padronizar `EPOCAS_FT = 15` para piso e método.


2. **Vazamento de estado global entre a Parte 1 e a Parte 2:**
Durante a auditoria do notebook, identificamos que a avaliação linear e o fine-tuning da Parte 1 dependiam das variáveis globais `loader_train` e `pesos_tensor` — que eram sobrescritas pelo loop de orçamentos da Parte 2 (terminando no orçamento 20). Caso células da Parte 1 fossem reexecutadas após a Parte 2, haveria vazamento silencioso de rótulos. A equipe corrigiu o problema isolando explicitamente `df_105`, `loader_105`, `loader_105_feat` e `pesos_105`, acompanhados de `assert len(df_105) == 105`.


3. **Ajuste de aumentação específico para o domínio (Câmeras Trap):**
O padrão genérico do SimCLR sugerido por literatura e IAs usa `RandomResizedCrop(scale=(0.2, 1.0))`. Como os animais frequentemente ocupam ~2% da imagem nas câmeras trap, crops pequenos geravam pares positivos compostos apenas por fundo repetitivo. A equipe formulou e testou a hipótese de elevar o piso do crop para `scale=(0.5, 1.0)`, preservando o animal nas duas *views*.


4. **Correção de escopo em respostas da IA:**
Em momentos pontuais durante o uso conversacional para revisão do notebook, a IA misturou contexto de projetos anteriores (métricas de rastreamento de objetos como IDSW/Fragmentação) e analisou uma versão truncada do arquivo `.ipynb`. A equipe filtrou imediatamente a alucinação, redirecionou o contexto para os arquivos corretos do PA3 e validou manualmente cada requisito da Parte 2 (orçamentos, pareamento por *seed* e ganho por classe).

---

## 4. Declaração de Autoria do Log

Em conformidade com a transparência exigida pela disciplina e pela ética acadêmica, declaramos que **este arquivo `AI_LOG.md` também foi redigido com auxílio de Inteligência Artificial**, tendo seu conteúdo, episódios técnicos e correções revisados e validados pela equipe a partir da experiência real no desenvolvimento do PA3.