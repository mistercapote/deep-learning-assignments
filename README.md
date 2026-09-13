# deep-learning-assignments
Este repositório contém as implementações de modelos e projetos práticos desenvolvidos durante a disciplina de Deep Learning.


# PA1: Segmentação de Instâncias com Arquiteturas da Aula
**Equipe:** Jaime Willian Carneiro da Silva e Luiz Eduardo Bravin  

Este repositório contém o código e os modelos para segmentação de instâncias baseada em Deep Learning, explorando abordagens binárias e ternárias (Watershed) com arquitetura ResUNet.

---

## 1. Ambiente
O repositório possui um arquivo `requirements.txt`. Para configurar o ambiente, instale as dependências:
```bash
pip install -r requirements.txt
```


## 2. Download dos Dados
Os modelos foram treinados utilizando o dataset DSB2018 / BBBC038v1.

Faça o download dos dados no link oficial: https://bbbc.broadinstitute.org/BBBC038

Extraia os dados de forma que a pasta stage1_train fique dentro do diretório data/ na raiz do projeto.

```
deep-learning-assignments/assignment-01/data/stage1_train
```

## 3. Comandos de Execução

Para rodar a pipeline pelo terminal, certifique-se de estar na raiz do projeto e utilize os comandos curtos abaixo:
- Para Treinar: 
```bash
python src/train.py --data_dir data/stage1_train
```

- Para avaliar:
```bash
python assignment-01/src/evaluate.py --data_dir assignment-01/data/stage1_train --weights assignment-01/models/best_model_ternary.pt
```

