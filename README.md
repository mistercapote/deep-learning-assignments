
# deep-learning-assignments
Este repositório contém as implementações de modelos e projetos práticos desenvolvidos durante a disciplina de Aprendizado Profundo (Deep Learning) — FGV EMAp.

# PA3: Poucos Rótulos — Fauna em Câmeras Trap com Dados Não Rotulados (Trilha A — SimCLR)
**Equipe:** Jaime Willian Carneiro da Silva e Luiz Eduardo Bravin  
**Disciplina:** Aprendizado Profundo | **Professor:** Dario Oliveira | **Monitor:** Erick Brito

Este diretório (`assignment-03/`) contém o pipeline completo para classificação de 21 espécies de fauna em imagens de câmeras trap (ENA24-detection) sob escassez severa de rótulos (5 rótulos por classe), utilizando pré-treino auto-supervisionado **SimCLR** sobre 6.505 imagens não rotuladas seguido de *fine-tuning* com backbone **ResNet-18** (11.187.285 parâmetros, abaixo do teto de 12M).

---

## 1. Ambiente
O projeto utiliza Python 3 e PyTorch (com suporte transparente a CPU, CUDA e Intel Arc `xpu`). Para configurar o ambiente, instale as dependências a partir da raiz do repositório:

```bash
pip install -r requirements.txt

```

---

## 2. Download e Estrutura dos Dados

Os experimentos utilizam o pacote preparado do dataset **ENA24-detection** (LILA BC, licença CDLA-Permissive-1.0), contendo 21 classes de fauna da América do Norte.

1. Faça o download do pacote `pa3_dados.zip` (415 MB):
https://drive.google.com/file/d/1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui/view?usp=sharing
2. Extraia o conteúdo dentro de `assignment-03/` de forma que a estrutura fique:

```text
deep-learning-assignments/assignment-03/
├── pa3_dados/
│   ├── classes.csv
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test_public/
│   └── labels/
│       ├── rotulos_1.csv
│       ├── rotulos_5.csv
│       ├── rotulos_20.csv
│       ├── val.csv
│       └── test_public.csv
├── models/
│   └── modelo_final.pt
├── src/
│   ├── redimensionamento.py
│   └── autosup.py
├── classes.csv
├── predict.py
├── main_final.ipynb
├── curva_orcamento.csv
└── AI_LOG.md

```

---

## 3. Comandos de Execução

Certifique-se de estar na pasta `assignment-03/` para executar os comandos abaixo.

### Treinar (pipeline completa: Partes 0, 1, 2 e 3)

Executa o notebook completo (piso supervisionado, pré-treino SimCLR nas 6.505 imagens sem rótulo, experimento de aumentação `scale=(0.2, 1.0)` vs. `scale=(0.5, 1.0)`, curva de orçamento nos orçamentos 1, 5 e 20 e exportação dos pesos e CSVs):

```bash
jupyter nbconvert --to notebook --execute --inplace main_final.ipynb

```

### Avaliar / Inferência (CPU e offline)

Executa o modelo final (`models/modelo_final.pt`) em CPU, sem acesso à internet, imprime o número de parâmetros e gera o CSV no formato `image_id,label`:

```bash
python predict.py --imagens pa3_dados/images/test_public --saida preds.csv

```

Para conferir **F1 macro** e **acurácia** das predições geradas no `test_public`:

```bash
python -c "
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
y_true = pd.read_csv('pa3_dados/labels/test_public.csv').set_index('image_id')['label']
y_pred = pd.read_csv('preds.csv').set_index('image_id')['label'].reindex(y_true.index)
print('F1 macro:', round(f1_score(y_true, y_pred, average='macro'), 4))
print('Acurácia:', round(accuracy_score(y_true, y_pred), 4))
"

```

---

## 4. Resumo dos Resultados (`test_public`)

Teto supervisionado de referência (todos os rótulos do `train`): **F1 macro = 0.686**.

| Orçamento (rótulos/classe) | Piso (ResNet-18 ImageNet) | Trilha A (SimCLR + Fine-tuning) |
| --- | --- | --- |
| **1 rótulo** (1 seed) | `0.212` | **`0.217`** |
| **5 rótulos** (média ± desvio, 3 seeds) | `0.391 ± 0.018` | **`0.419 ± 0.014`** |
| **20 rótulos** (1 seed) | **`0.611`** | `0.601` |

* **Modelo entregue (`models/modelo_final.pt`):** ResNet-18 com pré-treino SimCLR (`scale=(0.5, 1.0)`) + *fine-tuning* em `rotulos_5.csv` (F1 macro `0.4282`, acurácia `0.4525` no `test_public`, **11.187.285 parâmetros**).

---

## 5. Entregáveis no Repositório

* `predict.py`: script CLI de inferência offline em CPU.
* `models/modelo_final.pt`: pesos do modelo entregue.
* `classes.csv`: mapeamento das 21 espécies do ENA24.
* `curva_orcamento.csv`: métricas completas da Parte 2 (`metodo, orcamento, seed, f1_macro, acuracia`).
* `AI_LOG.md`: registro transparente do uso de ferramentas de IA no projeto.

