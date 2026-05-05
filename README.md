# FIAP Tech challenge 04-2026
Projeto para o primeiro bimestre do curso da FIAP de ML Engineering de 05/03/2026 a 05/05/2026:
Criar um modelo de ML para prever churn de uma compania de telecom.

# Stakeholders
- Patrocinador / Executivo (budget, prioridades)
- Product Owner / Gerente de produto (requisitos, sucesso do produto)
- Equipe de ML / Data Science (modelagem, experimentos)
- Engenharia de Dados (ETL, pipelines, qualidade dos dados)
- Engenharia de Software / MLOps (deploy, infra, integração)
- Equipe de QA / Testes (validação, testes de regressão)
- Equipe de Segurança & Privacidade (compliance, controle de acesso)
- Compliance / Jurídico (regulação, requisitos legais)
- Operações / Negócio / Stakeholders de domínio (especificações funcionais, adoção)
- Usuários finais / Clientes (feedback, aceitação)
- Suporte / Atendimento ao cliente (incidentes, comunicação)
- Equipe de Monitoramento & Observabilidade (performance, dérivas)
- Parceiros externos / Fornecedores (dados, modelos ou serviços terceirizados)

# Métricas de negócio
Redução de churn em ≥ X% (ex: 3–8%) em grupos acionados

# Dados
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# SLO (Service Level Objectives)
A API de churn deve identificar clientes com risco de cancelamento com qualidade suficiente para permitir ações preventivas do time de retenção, mantendo desempenho consistente em produção.

## Objetivos do modelo
F1-score ≥ 0.80 em janela móvel de 7 dias de produção
Recall ≥ 0.75 e Precision ≥ 0.70 para clientes churn
AUC-ROC ≥ 0.85

## Objetivos da API
Latência p95 ≤ 200 ms e Disponibilidade ≥ 99.5%
Taxa de erro da API ≤ 1%
Timeout rate ≤ 0.5%

## Estabilidade
Data drift < 0.2 em features críticas
Queda de F1 não pode exceder -5% vs baseline de treinamento

# Setup

poetry install
poetry run .\src\baseline.py