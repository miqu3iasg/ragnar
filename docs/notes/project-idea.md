# Ragnar

A ideia do projeto é ser um assistente de pesquisa autônomo, um backend que recebe uma pergunta complexa, quebra em subtarefas, usa ferramentas (busca na web, leitura de PDF, execução de código), e devolve um relatório, tudo orquestrado por você, não por uma lib pronta.

## Arquitetura sugerida (evolutiva, em fases)

### Fundamentos

- FastAPI com um endpoint `/ask` que recebe uma pergunta e chama a API da Anthropic direto (sem agente ainda)
- Aprende: FastAPI, Pydantic, requests assíncronas, variáveis de ambiente, estrutura de projeto Python

### Fase 2 — Ferramentas (tools)

- Adicione function calling: o modelo pode "decidir" usar uma ferramenta de busca na web (ex: Tavily ou SerpAPI) ou ler um arquivo
- Aprende: decorators, reflexão (inspect), JSON schema, tratamento de erros

### Fase 3 — Orquestração de agente

- Implemente um loop de agente (tipo ReAct: pensa → age → observa → repete) do zero, sem framework
- Aprende: máquinas de estado, recursão, generators/async generators para streaming de "pensamentos"

### Fase 4 — Persistência e memória

- PostgreSQL para salvar conversas, Redis para cache, embeddings + pgvector para memória de longo prazo
- Aprende: SQLAlchemy (ORM), migrations (Alembic), modelagem de dados

### Fase 5 — Produção

- Fila de tarefas (Celery) pra pesquisas longas rodarem em background, WebSocket pra mandar updates em tempo real pro frontend
- Docker Compose juntando tudo
- Aprende: concorrência real em Python (GIL, multiprocessing vs asyncio), deploy

## Stack final

```markdown
FastAPI + Pydantic + SQLAlchemy + Alembic
PostgreSQL + pgvector
Redis + Celery
Anthropic SDK (function calling)
Docker Compose
```
