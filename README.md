# SafeZone - Safety You Can See

**SafeZone** é um sistema avançado de monitoramento infantil que utiliza inteligência artificial (YOLOv5) para detectar pessoas e alertar quando uma criança sai de uma zona segura pré-definida. O sistema oferece uma interface web moderna, imagem da área e alertas sonoros.

<br>
<br>

### **Características Principais**

- **Detecção AI**: YOLOv5 otimizado para detecção de pessoas em tempo real
- **Zona Segura**: Definição visual de área de monitoramento
- **Alertas Inteligentes**: Sistema de alarme com controle de silenciamento
- **Interface Responsiva**: Design moderno com Bootstrap 5 e tema claro/escuro
- **Log de Eventos**: Histórico detalhado de todas as detecções (o histórico não é guardado)
- **Fallback Robusto**: Sistema de backup com OpenCV DNN
- **GPU Acceleration**: Suporte CUDA para melhor performance

<br>
<br>

## 🚀 **Instalação Rápida**

### **1. Clone o Repositório**
```bash
git clone git@github.com:pfviegas/SafeZone.git
cd SafeZone
```

### **2. Execução Automática**
```bash
chmod +x start-html.sh
./start-html.sh
```
### ** Ou Execução Manual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend
python main.py
```

### **3. Acesso ao Sistema**
- **HTTP**: `http://localhost:8000`
- **Rede**: `https://[SEU-IP]:8000`

<br>
<br>

## 📋 **Como Usar**

### **1. Seleção da Zona Segura**
1. Execute o sistema
2. Clique e arraste no vídeo para definir a zona
3. Confirme a seleção com Enter
4. Sistema inicia monitoramento automaticamente

### **2. Configuração de Audio**
1. Clique em **"Ativar Som"** para habilitar alertas
2. Navegador pode solicitar permissão de autoplay
3. Use **"Silenciar Alarme"** para pausar alertas específicos

### **Cores da Zona de Detecção**
- 🟢 **Verde**: Pessoa na zona segura
- 🟡 **Amarelo**: Pessoa fora da zona (aguardando)
- 🔴 **Vermelho**: Alarme ativo

<br>
<br>

**⚠️ Aviso**: Este sistema é destinado para monitoramento doméstico. Sempre supervisione crianças diretamente e use este sistema como ferramenta auxiliar. O sistema não substitui a supervisão humana responsável.

**🔒 Privacidade**: Todo processamento é feito localmente. Nenhum dado de vídeo é enviado para servidores externos.
