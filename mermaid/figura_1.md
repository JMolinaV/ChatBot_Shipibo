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
  "flowchart": { "curve": "basis", "htmlLabels": true, "nodeSpacing": 25, "rankSpacing": 36 }
}}%%
flowchart TD

%% ==== Estilos ====
classDef start fill:#dcfce7,stroke:#166534,stroke-width:1.2px,rx:10,ry:10,color:#052e16;
classDef process fill:#eef2ff,stroke:#334155,stroke-width:1.2px,rx:10,ry:10,color:#0f172a;
classDef decision fill:#fde68a,stroke:#92400e,stroke-width:1.2px,rx:14,ry:14,color:#111827;
classDef data fill:#fef9c3,stroke:#ca8a04,stroke-width:1.2px,rx:10,ry:10,color:#111827;
classDef io fill:#e0f2fe,stroke:#0369a1,stroke-width:1.2px,rx:10,ry:10,color:#0b3552;
classDef result fill:#e6fffa,stroke:#0f766e,stroke-width:1.4px,rx:12,ry:12,color:#065f46;
classDef code fill:#f5f5f5,stroke:#a3a3a3,stroke-width:1px,rx:10,ry:10,color:#111827;

%% ==== Flujo principal ====
S([▶️ <b>Inicio</b>]):::start

subgraph MOD["<b>1) Crear módulo traductor</b>"]
  direction TB
  C[[📦 <b>Clase</b> <code>Traductor</code>]]:::process
  M1["⚙️ <b>__init__()</b><br/>inicializa diccionario"]:::code
  D["(📚 <b>Diccionario ES→SH</b><br/><code>{hola: jaté, ...}</code>)"]:::data
  TR["🧩 <b>traducir(frase)</b>"]:::code
  C --> M1 --> D
  C --> TR
end

S --> MOD --> O["🆕 <b>Crear objeto</b><br/><code>traductor = Traductor()</code>"]:::process
IN[["✍️ <b>Frase de entrada</b><br/><code>hola amigo gracias</code>"]]:::io
O --> IN --> P["🧹 <b>Preprocesar</b><br/><code>frase.lower().split()</code>"]:::process

subgraph LOOP[<b>Bucle por palabras</b>]
  direction TB
  Q{{❓ <b>¿palabra ∈ diccionario?</b>}}:::decision
  T1["➕ <b>Agregar</b><br/><code>diccionario[palabra]</code>"]:::process
  T2["➕ <b>Agregar</b><br/><code>[palabra]</code>"]:::process
  Q -- Sí --> T1
  Q -- No --> T2
end

P --> Q
T1 --> ACC[🗂️ <b>Acumular traducciones</b>]:::process
T2 --> ACC
ACC --> J["🔗 <b>Unir con espacios</b><br/><code>join(lista)</code>"]:::process
OUT[[📝 <b>Traducción</b><br/><code>jaté nokon joikon</code>]]:::result
J --> OUT

PRINT["🖨️ <b>Imprimir</b><br/><code>print(...)</code>"]:::process
OUT --> PRINT --> END([🏁 <b>Fin</b>]):::start

%% Bucle de realimentación (opcional)
OUT -. Respuesta traducida .-> IN
