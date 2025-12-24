# Gateway: Local Voice-to-Voice AI Assistant
#### A latency-optimized, distributed audio pipeline implementing VAD + STT + LLM + TTS with an OpenAI-compatible API.

**[Documentation](#Documentation)** | **[Architecture](#Architecture)** | **[Demo](#Demo)** | **[Usage](#Usage)** | **[Roadmap](#Roadmap)** | **[Motivation](#Motivation)**

### Features
- Async-rich latency-optimized audio-to-audio AI Assistant (VAD + STT + LLM + TTS)
- Exposing OpenAI-compatible endpoints for all running models (REST, Websockets)
- Launching models on-demand using YAML config file
- Distributed architecture (run models on different nodes)
- gRPC for communication between containers
- OpenWebUI on-demand


### Documentation
- **[README -- Development](README.dev.md)**
- **[README -- API Usage](README.api.md)**
- **[README -- Architecture](README.arch.md)**

### Architecture
###HERE BE DRAGONS


## Demo
#### HERE BE DRAGONS

## Usage

### Prerequisites
- Linux machine
- NVIDIA GPU, min 24GB VRAM, CUDA 12 or higher
- Installed docker, docker compose, Nvidia container toolkit (ctk). See [guide.md](assets/docs/docker-docker-compose-ctl.md) to install

1. Clone the repository
    ```sh
    git clone https://app.git.valerii.cc/valerii/gateway.git
    cd gateway
    ```

2. Use config.yaml to configure running models \
**Note:** default config should suffice
    ```sh
    cp config.example.yaml config.yaml
    ```

3. Build Images
    ```sh
    sh run.dev.sh
    ```    

4. Start Containers 
    ```sh
    docker compose up -d
    ```

5. Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to access API documentation
![scalar.png](assets/screenshots/scalar.png)

## Roadmap
### HERE BE DRAGONS

## Motivation

As many other kids, watching and re-watching Iron Man movies I was captivated with an idea of having my own Jarvis someday.

Inspired of it or maybe something else, I chose a route of AI Engineer. Having played a founding engineering role in 2 startups already (as of 24 Dec '25), I decided to give my idea a shot.

Then I wondered, what are the SOTA open-weights models as of today, fast enough for seamless real-time communication would enable this.
I collected them all: Silero VAD, Parakeet V3 (STT), GPT-OSS-20B (LLM), and Kokoro (TTS) under one backend to figure how would I put them together into one audio-to-audio AI assistant pipeline.
If it performed well enough, I could've been delegating some mundane simple tasks vocally. 

As of today I have 2 proxmox nodes as my homelab, and how cool would it be to see an AI assistant could interact with it, -- thought I, outlining architecture of the Gateway. \
It could access the books I own, see movies I binge-watched during the weekend, access my self-hosted email and calendar, and draft replies to emails. Truly an unlimited power.

As of this project, it is a prototype I built to see, realistically, to what degree could I've minimized latency. 
I was really interested to see how would I approach and an architectural problem of sticking this many different models. 

Nearly 1 month later I like what I see. The current version of assistant interrupts me, does not let interrupt itself, spins off into languishing monologues, but it's standing on a solid foundation.
