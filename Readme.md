# home-cortex

**Experimental local home-cortex: stationary cognition layer for humanoid robots**

Personal sandbox exploring a dedicated, always-on local brain box for home humanoids.

### Core Vision

Humanoid robots in 2026 can dance on stages and fold laundry in controlled demos — but they still struggle to be genuinely useful in a chaotic, messy and highly personalized home.

Every home is its own chaotic universe, with tons of data that's both highly unstructured and local, that no companies (Tesla, Google) could pre-program for you.

To make a simple omelet in your kitchen, a humanoid robot must instantly know details that are unique to your home today:

Where the eggs, salt, and butter are stored right now
Which frying pan is best suited for the task (and where it lives)
Where the clean plates and bowls are kept
Which bin is for egg shells
… plus hundreds of other hyper-local, ever-changing facts and relationships

These aren't superficial trivia—they're essential for safe, efficient household help. Humanoids generate massive real-time sensor data (vision streams alone often exceed 10 Mbps), and turning that into reliable understanding requires computationally intensive machine learning for perception, mapping, and persistent memory.
But physics imposes hard limits: every additional watt of onboard compute adds weight, drains battery faster, and compromises balance and runtime. Offloading to the cloud sacrifices privacy (your home's intimate data leaves the premises) and introduces unacceptable latency for real-time decisions.
The result? A dedicated, local, always-on cognition layer becomes essential—not a luxury—to bridge the gap between impressive demos and everyday usefulness in real homes.

**home-cortex** is the missing piece:  
A stationary, local server that acts as the **cortex** — persistent home-specific memory, semantic understanding, long-horizon reasoning, and high-level planning.  

This allows the robot's onboard compute to stay lean and focused exclusively on **cerebellum**-layer functions: real-time balance, low-latency control loops, short-term sensor fusion, grasp servoing, and collision avoidance.

### Current Scope (6-week intellectual gym)

- **Kitchen wedge only** — focused on 18–25 everyday objects (eggs, frying pan, plates, bowls, knife, trash can, etc.)
- Primary task family:
  - Find and retrieve X (ie, find me the kitchen knife)
  - Update state of X (ie, trash can is full)
  - Update location of X (ie, Move bowls into the shelf)
- Simulated environment only (Gazebo / Isaac Sim)
- No hardware, no Matter protocol, no UI beyond terminal + minimal Streamlit labeler
- Black-box LLM usage only (Ollama + any 7B–13B model) — **zero model-layer work**
- Agent/system layer focus: memory store, tool definitions, ReAct-style loop, ROS2 boundary

In a home setting, responsiveness is non-negotiable: waiting 30–60 seconds for the robot to "figure out" where a bowl or knife is, feels broken and useless.  
Target: deliver high-level intents in **<3–4 seconds** on modest always-on hardware (mini-PC / NUC class, ~15–65W TDP) for simple tasks like "find me the knife" or "update trash can state".

### Goals of this sandbox

- Test whether a tiny persistent memory + agent loop actually feels magical on chaotic home-like scenarios
- Explore how much value comes from semantic locations, confidence decay, drift handling, and high-level intents
- Decide after 6 weeks whether this itch deserves more time without touching primary work (Relativity + Exploro)

### Non-goals (for now)

- Whole-home or multi-room modeling
- Vector databases / embeddings / neural memory
- Custom hardware design
- Open-source release or product framing
- Low-level robot control
- Multi-robot fusion
- Full natural-language interface

### Architecture Sketch
