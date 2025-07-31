#!/bin/bash

# Child Monitor - Start Script (HTML Version)
# Inicia apenas o backend que serve a interface HTML

clear
echo "🚀 Iniciando Child Monitor (Versão HTML/Bootstrap)..."

# Navegar para o diretório do projeto
cd "$(dirname "$0")"

# Verificar se existe ambiente virtual
if [ -d ".venv" ]; then
    echo "🐍 Ativando ambiente virtual..."
    source .venv/bin/activate
else
    echo "⚠️ Ambiente virtual não encontrado. Criando..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Instalar dependências se necessário
echo "📦 Verificando dependências do backend..."
pip install -r backend/requirements.txt

# Navegar para o diretório do backend para iniciar o servidor
cd backend

# Iniciar o backend (que agora serve a interface HTML)
echo "🌐 Iniciando servidor backend na porta 8000..."
echo "📱 Interface disponível em:"
echo "   🏠 Local: http://localhost:8000"
echo "   🌍 Rede:  http://$(hostname -I | awk '{print $1}'):8000"
echo "🛑 Para parar, pressione Ctrl+C"
echo ""

python main.py
