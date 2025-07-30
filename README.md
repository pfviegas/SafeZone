# SafeZone - Child Monitor

**Safety You Can See** - Sistema de monitoramento infantil com detecção de zona segura em tempo real.

## **Visão Geral**

O **SafeZone** é um sistema avançado de monitoramento infantil que utiliza inteligência artificial (YOLOv5) para detectar pessoas e alertar quando uma criança sai de uma zona segura pré-definida. O sistema oferece uma interface web moderna, alertas sonoros.

### **Características Principais**

- **Detecção AI**: YOLOv5 otimizado para detecção de pessoas em tempo real
- **Zona Segura**: Definição visual de área de monitoramento
- **Alertas Inteligentes**: Sistema de alarme com controle de silenciamento
- **Interface Responsiva**: Design moderno com Bootstrap 5 e tema claro/escuro
- **Log de Eventos**: Histórico detalhado de todas as detecções (o histórico não é guardado)
- **Fallback Robusto**: Sistema de backup com OpenCV DNN
- **GPU Acceleration**: Suporte CUDA para melhor performance


## 🚀 **Instalação Rápida**

### **1. Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/child-monitor.git
cd child-monitor
```

### **2. Execução Automática**
```bash
chmod +x start-html.sh
./start-html.sh

# Ou execução manual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend
python main.py
```

### **3. Acesso ao Sistema**
- **HTTPS**: `https://localhost:8000` (PWA habilitado)
- **HTTP**: `http://localhost:8000` (fallback)
- **Rede**: `https://[SEU-IP]:8000`

## 📋 **Configuração Inicial**

### **1. Seleção da Zona Segura**
1. Execute o sistema
2. Clique e arraste no vídeo para definir a zona
3. Confirme a seleção
4. Sistema inicia monitoramento automaticamente

### **2. Configuração de Audio**
1. Clique em **"Ativar Som"** para habilitar alertas
2. Navegador pode solicitar permissão de autoplay
3. Use **"Silenciar Alarme"** para pausar alertas específicos

### **3. Instalação PWA (Opcional)**
1. Acesse via HTTPS
2. Clique no botão **"📱 Instalar App"** quando aparecer
3. Confirme instalação no navegador
4. App ficará disponível na tela inicial

## ⚙️ **Configurações Avançadas**

### **Perfis de Performance**
```python
# backend/child_monitor/yolo_config.py
PERFORMANCE_PROFILES = {
    "fast": {          # Máxima velocidade, menor precisão
        "model_name": "yolov5n.pt",
        "img_size": 320,
        "confidence": 0.7
    },
    "balanced": {      # Equilibrio velocidade/precisão
        "img_size": 640,
        "confidence": 0.6
    },
    "accurate": {      # Máxima precisão, menor velocidade
        "model_name": "yolov5s.pt",
        "img_size": 832,
        "confidence": 0.5
    }
}
```

### **SSL/HTTPS Personalizado**
```bash
# Gerar certificados customizados
cd backend
openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=SEU-DOMINIO.COM"

# Iniciar com SSL
python main.py --ssl
```

## 🏗️ **Arquitetura do Sistema**

```
SafeZone/
├── backend/                    # Servidor principal
│   ├── main.py                # Ponto de entrada
│   ├── child_monitor/         # Módulos principais
│   │   ├── core.py           # FastAPI app & rotas
│   │   ├── detection.py      # YOLOv5 detector
│   │   ├── alerts.py         # Sistema de alertas
│   │   └── yolo_config.py    # Configurações AI
│   ├── static/               # Recursos estáticos
│   │   ├── styles.css        # CSS customizado
│   │   ├── sw.js            # Service Worker PWA
│   │   └── manifest.json     # PWA manifest
│   └── templates/
│       └── index.html        # Interface principal
├── start-html.sh             # Script inicialização
└── README.md                 # Esta documentação
```

## 🔌 **API e WebSocket**

### **Endpoints REST**
```
GET  /                    # Interface principal
GET  /video_feed         # Stream de vídeo
GET  /health            # Status do sistema
POST /shutdown          # Parar sistema
```

### **WebSocket (`/ws`)**
```javascript
// Mensagens recebidas:
{
  "type": "detection",
  "persons": [{"x": 100, "y": 200, "in_zone": true}],
  "zone": {"x1": 50, "y1": 50, "x2": 300, "y2": 400},
  "alert_active": false
}

{
  "type": "start",        // Alarme iniciado
  "timestamp": 1640995200
}

{
  "type": "stop",         // Alarme terminado
  "duration": 5.2
}
```

## 🎨 **Personalização da Interface**

### **Tema Escuro/Claro**
```css
/* Variáveis CSS personalizáveis */
:root {
    --zone-safe-color: #198754;      /* Verde da zona segura */
    --zone-warning-color: #ffc107;   /* Amarelo de aviso */
    --zone-danger-color: #dc3545;    /* Vermelho de perigo */
}

[data-bs-theme="dark"] {
    --bs-body-bg: #212529;           /* Fundo escuro */
    --bs-body-color: #ffffff;        /* Texto claro */
}
```

### **Cores da Zona de Detecção**
- 🟢 **Verde**: Pessoa na zona segura
- 🟡 **Amarelo**: Pessoa fora da zona (aguardando)
- 🔴 **Vermelho**: Alarme ativo

## 🔍 **Solução de Problemas**

### **Problemas Comuns**

#### **❌ Câmera Não Detectada**
```bash
# Verificar câmeras disponíveis
ls /dev/video*

# Testar câmera manualmente
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

#### **❌ YOLOv5 Lento/Não Funciona**
```bash
# Verificar CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Instalar CUDA toolkit se necessário
# Para Ubuntu:
sudo apt install nvidia-cuda-toolkit

# Testar com perfil rápido
# Editar yolo_config.py → performance_profile="fast"
```

#### **❌ SSL/HTTPS Não Funciona**
```bash
# Verificar certificados
ls -la backend/*.pem

# Regenerar certificados
cd backend
rm -f *.pem
../start-html.sh  # Irá regenerar automaticamente
```

#### **❌ Service Worker Falha**
```javascript
// Verificar console do navegador
// Deve mostrar: "✅ Service Worker registrado"

// Se falhar, verificar:
// 1. HTTPS habilitado
// 2. Arquivo sw.js existe em /static/
// 3. Navegador suporta SW
```

### **Logs e Debug**
```bash
# Logs detalhados
cd backend
python main.py --debug

# Verificar status em tempo real
curl http://localhost:8000/health

# Monitor WebSocket
# Use DevTools → Network → WS para ver mensagens
```

## 📊 **Performance e Benchmarks**

### **Resultados Típicos**
| Hardware | FPS | Latência | CPU | GPU |
|----------|-----|----------|-----|-----|
| i5 + GTX 1060 | 25-30 | <100ms | 40% | 60% |
| i7 + RTX 3070 | 45-60 | <50ms | 25% | 40% |
| i3 (sem GPU) | 8-12 | 200ms | 80% | - |
| Raspberry Pi 4 | 3-5 | 500ms | 95% | - |

### **Otimizações Disponíveis**
- ✅ **GPU CUDA**: 5-10x mais rápido
- ✅ **Modelo YOLOv5n**: Mais rápido que YOLOv5s
- ✅ **Resolução 416px**: Menor que 640px padrão
- ✅ **Confiança 0.6**: Filtra detecções desnecessárias

## 🤝 **Contribuição**

### **Como Contribuir**
1. Fork o projeto
2. Crie uma branch feature: `git checkout -b feature/nova-funcionalidade`
3. Commit suas mudanças: `git commit -m 'Adiciona nova funcionalidade'`
4. Push para a branch: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

### **Áreas para Contribuição**
- 🔧 **Melhorias de Performance**: TensorRT, ONNX, quantização
- 📱 **Mobile**: Apps nativos Android/iOS
- 🤖 **AI**: Modelos mais precisos, reconhecimento facial
- 🎨 **UI/UX**: Novas interfaces, acessibilidade
- 🔒 **Segurança**: Autenticação, criptografia de streams

## 📄 **Licença**

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 **Autor**

**Mr. V** - *Desenvolvimento Principal*

## 🙏 **Agradecimentos**

- **Ultralytics**: Por manter o YOLOv5 otimizado
- **FastAPI**: Framework web moderno e rápido
- **Bootstrap**: Interface responsiva e elegante
- **OpenCV**: Processamento de vídeo robusto

---

**⚠️ Aviso**: Este sistema é destinado para monitoramento doméstico. Sempre supervisione crianças diretamente e use este sistema como ferramenta auxiliar. O sistema não substitui a supervisão humana responsável.

**🔒 Privacidade**: Todo processamento é feito localmente. Nenhum dado de vídeo é enviado para servidores externos.
