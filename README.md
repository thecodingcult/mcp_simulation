# 🧠 MCP Simulation – Interactive Visualization of Model Context Protocol

**Author:** thecodingcult  
**Repository:** `mcp_simulation`

---

## 🚀 Overview

This project visualizes the **Model Context Protocol (MCP)** using `matplotlib`. It provides an animated, interactive breakdown of how a language model request flows through key components such as memory retrieval, tool invocation, prompt assembly, and final LLM output.

It simulates a realistic agent pipeline, making it ideal for:
- Teaching AI architecture concepts
- Demonstrating prompt orchestration
- Visualizing real-time LLM data flows

---

## ✨ Key Features

- Modular component layout (UserInput → OutputRender)
- Glowing highlights and animations for active modules
- Eased packet motion tracing between components
- Narration bar with step-by-step guidance
- Clickable components with dynamic info panel
- Speed controls (0.5x / 1x / 2x), pause, and reset

---

## 🖼 Simulation Preview

> _The simulation runs in a dark-themed matplotlib window with animated packets and flowing narration._

---

## 🧱 Components Visualized

- **User Input** – accepts the user's question
- **MCP Server** – orchestrates the overall flow
- **Memory Store** – retrieves past context
- **Tool Suite** – calls external APIs/tools
- **Context Assembler** – prepares final LLM prompt
- **LLM Engine** – generates response
- **Output Renderer** – displays the result
- **Memory Updater** – updates long-term memory

---

## ▶️ How to Run

1. **Install dependencies**

```bash
pip install matplotlib numpy

Run the simulation
python mcp_sim.py

Controls inside the animation
⏸ Pause / ▶ Play / 🔁 Reset
🐢 0.5x / 🚀 1x / ⚡️ 2x speed options

🖱 Click any component to view details in the info panel

📁 Project Structure
mcp_simulation/
├── mcp_sim.py       # Main simulation code
└── README.md        

💡 Educational Purpose
This simulation was designed to help learners visualize AI agent workflows, especially how multi-step LLM systems orchestrate memory, tools, and final responses.

Use it for:
Lectures & teaching
AI workshops
Prompt engineering demonstrations




