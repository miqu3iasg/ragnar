Lista única, em ordem. Cada item pressupõe o anterior concluído.

- [x] Criar pasta do projeto e ambiente virtual com uv
- [x] Instalar FastAPI e Uvicorn, subir servidor local com endpoint simples
- [x] Acessar a documentação automática do FastAPI (/docs)
- [x] Criar conta no OpenRouter e gerar API key
- [x] Guardar a API key em variável de ambiente (.env + .gitignore)
- [x] Instalar lib openai e python-dotenv (ou equivalente com uv add)
- [x] Criar config.py lendo variáveis de ambiente (api key, base url, modelo)
- [x] Criar client.py com a função get_completion, usando o client OpenAI apontado pro OpenRouter
- [x] Testar get_completion isolado, fora do FastAPI
- [x] Criar __init__.py nas pastas necessárias para permitir imports
- [x] Instalar pytest como dev dependency
- [x] Escrever teste com mock para get_completion
- [x] Rodar e validar o teste com uv run pytest
- [x] Criar modelo Pydantic Answer em domain/research/answer.py
- [x] Criar domain/research/service.py com função que chama get_completion e devolve a resposta
- [x] Ligar a rota /ask ao service (não mais ao client diretamente)
- [x] Usar response_model na rota /ask apontando para Answer
- [ ] Testar manualmente o fluxo completo pelo /docs
- [ ] Escrever teste do endpoint /ask usando TestClient, com mock
- [x] Pesquisar e implementar retry com backoff (lib tenacity) em get_completion
- [x] Capturar especificamente a exceção de rate limit do SDK openai e diferenciar de erros que não valem retry
- [ ] Criar um exception handler no FastAPI para erros de rate limit, com status HTTP apropriado
- [x] Decidir o nicho do produto (ex: acadêmico, jurídico, médico, notícias) — já decidi, será acadêmico
- [ ] Pesquisar tool calling / function calling na API do modelo escolhido, entender formato de definição de ferramentas
- [ ] Definir a primeira ferramenta: busca na web (pesquisar APIs de busca com camada gratuita, ex: Tavily, SerpAPI, Brave Search API)
- [ ] Implementar a ferramenta de busca isolada (função Python simples, sem IA ainda), testada sozinha
- [ ] Conectar a ferramenta ao loop de function calling: o modelo decide quando chamar a busca
- [ ] Implementar a etapa de decisão "vale a pena buscar fonte externa para essa pergunta, ou o modelo responde direto?"
- [ ] Implementar fetch do conteúdo das páginas retornadas pela busca (não só o snippet)
- [ ] Extrair texto limpo de cada página buscada (pesquisar libs de extração de conteúdo de HTML, ex: trafilatura, readability-lxml)
- [ ] Pesquisar o conceito de embeddings e como gerar embeddings de texto (via API do provedor ou modelo local leve)
- [ ] Implementar geração de embeddings da pergunta do usuário e de cada resultado de busca
- [ ] Implementar re-ranking dos resultados por similaridade semântica (pesquisar similaridade de cosseno)
- [ ] Escrever testes para a função de re-ranking com dados fake (sem chamar API de verdade)
- [ ] Pesquisar o conceito de Natural Language Inference (NLI) e modelos pré-treinados para essa tarefa
- [ ] Escolher e testar isoladamente um modelo de NLI (pesquisar modelos disponíveis via HuggingFace, ex: modelos treinados em MNLI)
- [ ] Implementar comparação par a par entre trechos de fontes diferentes, classificando concordância/contradição/neutralidade
- [ ] Definir estrutura de dados para representar uma fonte (url, título, trecho usado, score de relevância, resultado de contradição)
- [ ] Implementar a montagem da resposta final citando explicitamente as fontes usadas, com seus respectivos trechos
- [ ] Atualizar o modelo Answer para incluir a lista de fontes citadas
- [ ] Escrever testes de ponta a ponta do fluxo completo (pergunta → busca → re-rank → contradição → resposta com fontes), tudo mockado
- [ ] Avaliar necessidade de persistência: banco de dados para histórico de perguntas e fontes usadas
- [ ] Instalar e configurar PostgreSQL, escolher ORM (SQLAlchemy) e configurar migrations (Alembic)
- [ ] Modelar tabelas para perguntas, respostas e fontes citadas
- [ ] Persistir cada pesquisa realizada no banco
- [ ] Avaliar se um classificador de credibilidade de domínio/fonte agrega valor real ao nicho escolhido
- [ ] Se sim, definir features do classificador (domínio, presença de autor, data, citações cruzadas, etc) e montar dataset inicial manualmente
- [ ] Treinar um classificador simples (scikit-learn) para pontuar credibilidade de fonte
- [ ] Integrar o score de credibilidade ao pipeline de re-ranking/exibição de fontes
- [ ] Avaliar se o caso de uso justifica processamento em background (pesquisas demoradas, múltiplas fontes)
- [ ] Se sim, configurar Redis, escolher fila (Celery ou RQ) e implementar worker separado
- [ ] Alterar endpoint para aceitar a pergunta, devolver um ID de tarefa e processar em background
- [ ] Criar endpoint para consultar status/resultado da pesquisa em andamento
- [ ] Implementar streaming de progresso da pesquisa (SSE ou WebSocket) — ex: "buscando fontes", "avaliando credibilidade", "montando resposta"
- [ ] Adicionar autenticação de usuários na API
- [ ] Adicionar rate limiting na própria API
- [ ] Adicionar logging estruturado em toda a aplicação
- [ ] Adicionar métricas básicas (ex: Prometheus) para monitorar uso, custo de tokens e erros
- [ ] Escrever Dockerfile da aplicação e docker-compose juntando API, worker, Postgres e Redis
- [ ] Documentar o projeto (README com setup, arquitetura, decisões de design e exemplos de uso)
- [ ] Revisar cobertura de testes e cobrir lacunas restantes

---

## Ferramentas e tecnologias por trecho do roadmap

### Busca e extração de conteúdo
- **API de busca**: Tavily (feita para agentes de IA, tem camada free), Brave Search API, ou SerpAPI — pesquisar "Tavily API python", tem SDK e é a mais usada em projetos de agente
- **Extração de texto de página**: `trafilatura` (mais moderna, boa pra artigos/notícias) ou `readability-lxml` — pesquisar "trafilatura python extract article text"

### Embeddings e re-ranking
- **Gerar embeddings via API**: modelos de embedding do próprio OpenRouter/OpenAI, ou `sentence-transformers` rodando local (mais leve que um LLM, não deve pesar tanto no SSD)
- **Similaridade de cosseno**: `numpy` (implementação simples) ou `scikit-learn` (`cosine_similarity` pronta) — pesquisar "cosine similarity python embeddings"

### NLI / detecção de contradição
- **HuggingFace Transformers**: pesquisar "huggingface pipeline zero-shot NLI" e "MNLI model huggingface" — existem modelos prontos pra rodar local, alguns pequenos o suficiente pra CPU
- Alternativa mais simples pra começar: usar o próprio LLM (via prompt estruturado) pra classificar contradição entre dois trechos, e migrar pra um modelo NLI dedicado depois — pesquisar essa troca antes de decidir qual caminho seguir primeiro

### Classificador de credibilidade
- **scikit-learn**: pesquisar "scikit-learn text classification tutorial" — ponto de partida clássico
- Pra features de texto: `TfidfVectorizer` do próprio scikit-learn é suficiente pra começar, não precisa de nada mais sofisticado nessa fase

### Function calling / tool calling
- Pesquisar na documentação do OpenRouter e/ou do modelo especificamente escolhido: "function calling" ou "tool calling" — o formato segue o padrão da OpenAI, então a doc da OpenAI sobre function calling também serve de referência: https://platform.openai.com/docs/guides/function-calling

### Persistência e infraestrutura (retomando o que já estava planejado)
- PostgreSQL, SQLAlchemy, Alembic, Redis, Celery/RQ, Docker — mesmas referências já passadas anteriormente

---

## Estrutura de pastas sugerida (evolução da atual)

```
ragnar/
├── domain/
│   └── research/
│       ├── question.py            # já existe
│       ├── answer.py              # inclui agora lista de fontes citadas
│       ├── source.py              # modelo Pydantic: url, título, trecho, score, credibilidade
│       └── service.py             # orquestra o fluxo: decide buscar, chama tools, monta resposta
│
├── infrastructure/
│   ├── llm/
│   │   ├── client.py              # já existe (get_completion)
│   │   ├── config.py              # já existe
│   │   └── tools.py               # definição das tools expostas ao modelo (function calling)
│   │
│   ├── search/
│   │   ├── client.py              # chamada à API de busca (Tavily/Brave/SerpAPI)
│   │   └── extractor.py           # extração de texto limpo das páginas (trafilatura)
│   │
│   ├── embeddings/
│   │   ├── client.py              # geração de embeddings
│   │   └── ranking.py             # similaridade de cosseno / re-ranking
│   │
│   ├── nli/
│   │   └── contradiction.py       # detecção de contradição entre trechos
│   │
│   └── credibility/
│       ├── model.py               # carregamento/uso do classificador treinado
│       └── train.py               # script separado para treinar o classificador (não roda em produção)
│
├── api/
│   └── routes/
│       └── research.py            # endpoint /ask (e futuramente /ask/status se for assíncrono)
│
├── tests/
│   ├── domain/
│   │   └── research/
│   │       └── test_service.py
│   └── infrastructure/
│       ├── llm/
│       │   └── test_client.py     # já existe
│       ├── search/
│       │   └── test_client.py
│       ├── embeddings/
│       │   └── test_ranking.py
│       └── nli/
│           └── test_contradiction.py
│
├── main.py
├── .env
├── .gitignore
├── pyproject.toml
└── uv.lock
```

Nota sobre `credibility/train.py`: scripts de treino de modelo geralmente não rodam como parte da API — são executados manualmente ou por um pipeline separado, gerando um arquivo de modelo salvo (ex: `.pkl` ou `.joblib`) que o `model.py` carrega em produção. Vale pesquisar como salvar/carregar modelos scikit-learn (`joblib.dump` / `joblib.load`) quando chegar nessa parte.
