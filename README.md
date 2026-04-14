# Reconhecimento Facial com Máscaras — Siamese Network

Solução para o desafio de reconhecimento facial em cenário pós-pandemia, onde o uso de máscaras dificulta a identificação por encobrir pontos críticos do rosto. A abordagem adotada é **Similarity Learning com Siamese Networks**, treinada com **Triplet Loss**.

---

## Problema

Modelos clássicos de reconhecimento facial dependem de regiões do rosto que ficam escondidas pela máscara (boca, nariz, bochechas). A estratégia aqui é treinar o modelo para aprender uma representação compacta (embedding) que capture as features visíveis — principalmente olhos e sobrancelhas — e calcule a **similaridade entre pares de imagens**.

---

## Dataset

O dataset contém pastas nomeadas com celebridades, cada uma com imagens da pessoa com e sem máscara (formato `.jpg`, tamanho fixo `112×112`).

| Etapa | Total de pessoas | Total de imagens |
|---|---|---|
| Dataset original | 2.996 | — |
| Após filtrar pessoas com máscara **e** sem máscara | 1.894 | 8.870 |
| Após remoção de duplicatas (SHA-256) | 1.894 | 8.370 |
| Após balancear (mínimo 3 imagens/pessoa) | 1.153 | 3.459 |

### Pré-processamento

1. **Rotulação automática** — como os nomes dos arquivos não indicam se a pessoa está de máscara ou não, foi usado o modelo [`Hemgg/Facemask-detection`](https://huggingface.co/Hemgg/Facemask-detection) (HuggingFace) para classificar cada imagem.
2. **Filtro de pessoas válidas** — apenas pessoas com ao menos 1 imagem com máscara e 1 sem máscara foram mantidas.
3. **Remoção de duplicatas** — comparação por hash SHA-256 dos bytes da imagem, removendo 500 duplicatas.
4. **Balanceamento** — cada pessoa foi limitada a exatamente 3 imagens (1 com máscara, 1 sem máscara, 1 aleatória), evitando viés para pessoas com mais amostras.

---

## Modelo

Arquitetura: **Siamese Network** com backbone **ResNet18** pré-treinado no ImageNet.

| Componente | Detalhe |
|---|---|
| Backbone | ResNet18 (features do ImageNet) |
| Dimensão do embedding | 128 (padrão FaceNet) |
| Normalização | L2 normalization na saída |
| Função de perda | Triplet Loss (margem = 1) |
| Optimizer | Adam |
| Batch | 32 pessoas × 3 imagens = 96 imagens |

### CustomSampler

Um `CustomSampler` foi implementado para garantir que cada batch contenha múltiplas imagens por pessoa. Sem isso, batches aleatórios podiam ter uma imagem de cada pessoa, impossibilitando a formação de pares positivos para a Triplet Loss.

### Divisão dos conjuntos

A separação foi feita **por pessoa** (não por imagem) para evitar data leakage.

| Conjunto | Pessoas | Imagens |
|---|---|---|
| Treino | 807 (75%) | 2.421 |
| Validação | 173 (15%) | 519 |
| Teste | 173 (15%) | 519 |

---

## Resultados

| Métrica | Valor |
|---|---|
| **ROC-AUC** | **0.8846** |
| **EER** | **0.2112** |

O modelo acerta ~88% dos pares corretamente. O EER de 21% indica que, no threshold de equilíbrio, o modelo erra igualmente ao aceitar impostores e ao rejeitar pessoas legítimas.

### Tentativas de melhoria

Foram testadas as seguintes modificações, que resultaram em desempenho levemente inferior (AUC 83%, EER 24%):

- `ReduceLROnPlateau` para redução adaptativa do learning rate
- Troca do Adam por **AdamW** com `weight_decay`
- **Dropout** na camada de normalização
- **Data Augmentation** (flip, rotação, variação de brilho)

A conclusão é que o gargalo é a **quantidade de dados**: 1.153 pessoas com apenas 3 imagens cada é insuficiente para que técnicas de regularização tragam ganho expressivo.

---

## Notebook

O notebook completo com análise exploratória, pré-processamento e treinamento está disponível no Kaggle: [carlos-oliveira-solution](https://www.kaggle.com/code/carlosvictor221/carlos-oliveira-solution)
