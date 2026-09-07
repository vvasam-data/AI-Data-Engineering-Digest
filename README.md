# 🤖 Autonomous AI Data Engineering Digest Agent

An autonomous, serverless AI agent built with **LangChain**, **Google Gemini 2.5 Flash**, and **GitHub Actions**. The agent fetches daily updates from RSS feeds and YouTube channels, filters them specifically for **Data Engineering** relevance (ETL/ELT, vector databases, query engines, and LLMOps), and delivers a curated HTML digest straight to your inbox every morning.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green?style=flat-square)
![Google Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 🏗️ Architecture & Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│                      GitHub Actions                         │
│                  (Daily Cron Trigger)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agent Harness (LangChain)                 │
│                                                             │
│  1. Fetch Updates ──>  2. LLM Reasoning  ──>  3. Deliver    │
│     (RSS & YouTube)       (Gemini 2.5)         (Resend API) │
└─────────────────────────────────────────────────────────────┘