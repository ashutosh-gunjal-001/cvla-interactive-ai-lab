# 🧠 CVLA Interactive AI Lab – AI-Powered Visualizer for Complex Variables & Linear Algebra 🤖🧮

**CVLA Interactive AI Lab** is a free, AI-powered, interactive web app that allows users to explore *Complex Variables* and *Linear Algebra* topics visually. Designed for education and self-learning, it uses lightweight AI models and math engines to visualize functions, solve equations, and simulate transformations — right in your browser.

---

## 🌐 Documentation:
https://drive.google.com/file/d/1ZDmShr00OD6v9GDcbs3c0Q-g8XAda4w1/view?usp=drive_link

---

## 🚀 Features

- 🔍 Visualize complex functions with domain coloring  
- 🧮 Animate 2D/3D matrix transformations and eigenvalue flow  
- ✍️ Solve equations step-by-step with symbolic reasoning  
- 🧠 AI-generated suggestions for subspaces, bases, solutions  
- 🎛️ Real-time sliders for input control and interactive graphing  
- 🌐 Fully hosted using free-tier platforms (no installation needed)

---

## 🛠️ Tech Stack

**Frontend:**  
- React.js (Next.js 14)  
- TailwindCSS  
- Plotly.js, Three.js, Math.js, KaTeX  
- Streamlit (for Python-based alternate UIs)

**Backend & AI Models:**  
- FastAPI (optional)  
- Hugging Face Spaces (Gradio-based demos)  
- ONNX for lightweight inference  
- Replicate / Ollama for on-demand LLM/AI tools

**Hosting Services:**  
- Vercel (Frontend Deployment)  
- Hugging Face Spaces (AI Models & Demos)  
- Render (Python backend)  
- GitHub Pages (Static Visual Assets)

---

## 🧪 System Architecture

[User Input: Matrix / Equation / Function]  
        ↓  
[Web Frontend (React / Streamlit)]  
        ↓  
[Math Logic & Visualizations (JS / Python)]  
        ↓  
[Optional AI Models (HF Spaces, Replicate)]  
        ↓  
[Interactive Output: Plotly / Canvas / 3D Renders]

---

## 📁 Project Structure

cvla-ai-lab/  
├── client/                   → Frontend (React/Next.js)  
│   ├── pages/                → Pages per basket (topics)  
│   ├── components/           → UI blocks (navbar, forms)  
│   ├── utils/                → Math & parser functions  
│   └── visualizations/       → Plotly / Three.js visual modules  
├── server/                   → FastAPI backend (optional)  
│   ├── api/                  → Solver endpoints  
│   └── models/               → Model wrappers (SVD, CR checker)  
├── huggingface_spaces/       → Gradio demos and AI configs  
│   ├── gradio_apps/  
│   └── ai_models/  
├── public/                   → Static files (images, logos)  
├── static/                   → Sample graphs / outputs  
├── README.md  
└── requirements.txt  

---

## 📦 Requirements

streamlit==1.32.0           # Optional Python-based interface  
numpy==1.26.4               # Matrix operations  
scipy==1.12.0               # Linear algebra, integration  
sympy==1.12                 # Symbolic math (step-by-step)  
matplotlib==3.8.2           # Static plots and graphs  
plotly==5.19.0              # Interactive charts and 3D plots  
pillow==10.2.0              # Image handling (e.g., domain coloring)

---

## 🙌 Contributing

Contributions welcome! Fork this repo, build new modules, or improve visual demos. Open issues or submit pull requests — let’s make complex math fun and visual! 🎨

---

## 📜 License

This project is open-source under the MIT License.
