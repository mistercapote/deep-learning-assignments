# AI Log (Registro de Uso de Ferramentas de IA)

Este documento descreve como as ferramentas de Inteligência Artificial foram utilizadas ao longo do desenvolvimento deste assignment, em conformidade com as diretrizes da **Seção 5 (Política de uso de IA)**.

---

## 1. Visão Geral da Abordagem

A maior parte da base de código deste projeto foi gerada com auxílio de modelos de linguagem (LLMs). Nosso papel principal como equipe consistiu em:

- **Arquitetura e Formulação de Prompts:** Definir a modelagem do problema, interfaces e restrições técnicas para guiar a IA.
- **Auditoria e Compreensão:** Analisar cada bloco de código gerado para garantir que a lógica, complexidade e funcionamento interno fossem completamente compreendidos por nós.
- **Adaptações e Refatoração:** Corrigir inconsistências, alinhar dependências, ajustar regras de negócio específicas da entrega e integrar os módulos gerados.

---

## 2. Ferramentas Utilizadas

- **Modelos:** ChatGPT (OpenAI) / Claude (Anthropic) / GitHub Copilot
- **Finalidades:** Geração de boilerplate, implementação de algoritmos centrais, sugestões de testes automatizados e depuração de erros em tempo de execução.

---

## 3. Episódios e Casos de Uso

Abaixo estão descritos alguns momentos pontuais em que recorremos à IA durante o ciclo de desenvolvimento:

### Episódio 1: Geração da Estrutura Base e Módulos Principais
- **Contexto:** Necessidade de estruturar rapidamente a arquitetura inicial do projeto com separação clara de responsabilidades.
- **Uso da IA:** Solicitamos a geração dos esqueletos das classes, rotas/endpoints e definições de dados a partir dos requisitos do enunciado.
- **Adaptação Realizada:** O código bruto gerado continha suposições genéricas de bibliotecas. Fizemos a adaptação das assinaturas de métodos e removemos dependências desnecessárias para manter o projeto leve e alinhado aos padrões exigidos.

### Episódio 2: Resolução de Bug de Integração / Tipagem
- **Contexto:** Encontramos falhas de compatibilidade entre as respostas dos métodos gerados e o consumo pelos testes automatizados.
- **Uso da IA:** Alimentamos o modelo com o stack trace do erro e os trechos de código conflitantes para diagnosticar o ponto de falha.
- **Adaptação Realizada:** A IA sugeriu uma reescrita completa que alteraria contratos existentes. Em vez de aceitar a solução cega, compreendemos a causa raiz apontada pela ferramenta (desserialização inadequada) e aplicamos pontualmente o parsing correto sem quebrar o restante da aplicação.

### Episódio 3: Criação de Testes de Borda
- **Contexto:** Garantir cobertura adequada para cenários imprevistos e validações de entrada.
- **Uso da IA:** Pedimos que o modelo gerasse uma suíte de testes unitários cobrindo casos nulos, limites numéricos e exceções.
- **Adaptação Realizada:** Adaptamos os asserções aos dados reais do domínio e descartamos testes redundantes ou que faziam mock excessivo de comportamentos triviais.

---

## 4. Declaração de Compreensão do Código

Reiteramos que, embora grande parte do código tenha sido gerada sinteticamente por IA, todo o fluxo de controle, manipulação de estado, dependências e decisões arquiteturais foram revisados, testados e são de pleno domínio técnico da equipe.

---

## 5. Nota sobre a Elaboração deste Documento

Este próprio arquivo `AI_LOG.md` foi redigido com o auxílio de uma ferramenta de Inteligência Artificial, tendo passado por revisão para garantir fidelidade às práticas adotadas no assignment.