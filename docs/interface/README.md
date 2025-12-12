# 🧬 Interface - Streamlit Control Deck

**Module**: `interface/app.py`
**Framework**: Streamlit
**Purpose**: Main entry point for the Alexandria web UI.

---

## 🎯 Overview

The Alexandria Control Deck is a full-stack Streamlit application providing a visual interface for interacting with the Alexandria cognitive system. It follows Streamlit's multipage app pattern with a main `app.py` and separate page modules in `pages/`.

---

## 📂 Structure

| File/Folder | Description |
|-------------|-------------|
| `app.py` | Main application entry point, renders landing page. |
| `pages/1_🧠_Dashboard.py` | Real-time monitoring (CPU, RAM, LanceDB status). |
| `pages/2_🍄_Mycelial_Brain.py` | Interactive reasoning chat + neural network visualization. |
| `pages/3_🕸️_Knowledge_Graph.py` | 3D knowledge graph explorer. |
| `pages/4_🔮_Abduction.py` | Hypothesis generation interface. |
| `pages/5_💥_Collider.py` | Concept collision playground. |
| `components/sidebar.py` | Shared sidebar navigation. |
| `utils/state_manager.py` | Session state initialization. |

---

## ▶️ Running the Interface

```bash
streamlit run interface/app.py
```
Or use the convenience script:
```bash
./run_interface.bat  # Windows
# or
python -m streamlit run interface/app.py
```

---

## 🎨 Styling

Custom CSS is embedded for a dark theme (GitHub-like):
- Background: `#0e1117`
- Primary accent: `#58a6ff`
- Card backgrounds: `#161b22`

---

**Last Updated**: 2025-12-11
**Version**: 1.0
**Status**: Development
