# Home Cortex GUI

A thin household chat client. Open WebUI is a behavioral reference only;
this app does not include its backend, RAG, workspace, or component tree.

The browser talks to Home Cortex through the compose reverse proxy
(`/session`, `/conversations`, `/v1`). Chat transcripts live in
Cortex. This process serves static files.
