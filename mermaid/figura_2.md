%%{init: {
  "theme": "neutral",
  "themeVariables": {
    "fontFamily": "Inter, ui-sans-serif, Segoe UI, Roboto, system-ui",
    "primaryColor": "#ffffff",
    "primaryTextColor": "#0f172a",
    "primaryBorderColor": "#1f2937",
    "lineColor": "#334155",
    "tertiaryColor": "#f8fafc"
  },
  "flowchart": { "curve": "basis", "htmlLabels": true, "nodeSpacing": 18, "rankSpacing": 35 }
}}%%
flowchart TD

%% ===== Clases de estilo =====
classDef io fill:#e0f2fe,stroke:#0369a1,stroke-width:1.2px,rx:10,ry:10,color:#0b3552;
classDef block fill:#eef2ff,stroke:#334155,stroke-width:1.2px,rx:10,ry:10,color:#0f172a;
classDef accent fill:#fde68a,stroke:#92400e,stroke-width:1.2px,rx:10,ry:10,color:#0f172a;
classDef step fill:#fff7ed,stroke:#9a3412,stroke-width:1px,rx:10,ry:10,color:#111827;
classDef data fill:#fef9c3,stroke:#ca8a04,stroke-width:1.2px,rx:12,ry:12,color:#111827;

%% ===== Entrada / UI =====
U([👤 <b>Usuario</b>]):::io
I[[🧾 <b>Interfaz Textual</b>]]:::io
U --> I

%% ===== Detección de idioma =====
DID{{🔎 <b>Detección de Idioma</b>}}:::accent
I -->|Texto en español o shipibo| DID

%% ===== Procesamiento del corpus =====
PC[[🧠 <b>Procesamiento del Corpus</b>]]:::block
DID --> PC

P[🧪 <b>Preprocesamiento</b>]:::block
LD[<b>Limpieza de datos</b>]:::step
NO[<b>Normalización</b>]:::step
ES[<b>Estructuración</b>]:::step

PC --> LD --> P
PC --> NO --> P
PC --> ES --> P

%% ===== Tokens / Embeddings =====
TOK[<b>Tokenización</b>]:::step
EMB[<b>Generación de embeddings</b>]:::step
P --> TOK
P --> EMB

MD[[💬 <b>Módulo de Intenciones y Diálogo</b>]]:::accent
TOK & EMB -->|Tokens y embeddings| MD

DI[<b>Detección de intención</b>]:::step
GD[<b>Gestión de diálogo</b>]:::step
PR[<b>Preparación de respuesta</b>]:::step
MD --> DI
MD --> GD
MD --> PR

%% ===== NLP y modelos de traducción =====
NLP[[🤖 <b>Modelo NLP</b>]]:::block
PR --> NLP

SHES[<b>Modelo de traducción SH→ES</b>]:::step
ESSH[<b>Modelo de traducción ES→SH</b>]:::step
NLP --> SHES
NLP --> ESSH

VAL[[🧭 <b>Validación lingüística</b>]]:::accent
SHES --> VAL
ESSH --> VAL

%% ===== Base de datos =====
DB[(📚 <b>Base de Datos</b>)]:::data
PR -.->|Traducción procesada| DB

%% ===== Salida / Entrega =====
GR[[✨ <b>Generación de Respuesta</b>]]:::block
VAL --> GR
DB --> GR

API[🌐 <b>APIs de integración</b>]:::io
PTX[✍️ <b>Prototipo textual</b>]:::io
GR --> API --> PTX

%% ===== Bucle de respuesta al usuario =====
GR -->|Respuesta traducida| I
