# NeuroAgent Deployment Guide (100% Free, 1-Click Access)

This guide provides two simple, 100% free ways to deploy and share NeuroAgent so that anyone can access the system through a single link.

---

## Option 1: Instant 1-Click Sharing (Recommended - Fastest Performance)

This option uses Cloudflare Tunnel to expose your local running instance to a secure public HTTPS link.

### Why this is best:
- **Instant:** Generates a public HTTPS link in 3 seconds.
- **Full GPU Speed:** Scans process in ~1-2 seconds on your NVIDIA RTX 3050 Laptop GPU.
- **Zero Cost & Zero Registration:** No accounts, no sign-ups, no credit cards required.

### How to use:
1. Double-click **`share_online.bat`** in the project folder.
2. The script will automatically:
   - Verify the backend server is running on port 8000.
   - Launch Cloudflare Tunnel.
   - Display a public HTTPS link (e.g., `https://xxxx.trycloudflare.com`).
3. Copy that link and send it to anyone! They can open it on their phone, laptop, or tablet and use the full NeuroAgent application.

---

## Option 2: 24/7 Permanent Cloud Hosting on Hugging Face Spaces (100% Free Forever)

If you want the application to stay online even when your laptop is closed, deploy to Hugging Face Spaces.

### Free Resources:
- 16 GB RAM
- 2 vCPUs
- 50 GB persistent storage
- Permanent URL: `https://huggingface.co/spaces/<your-username>/neuro-agent`

### Quick 1-Click Cloud Deployment:
1. Double-click **`deploy_24_7_free.bat`**.
2. If prompted, paste your free Hugging Face token (from https://huggingface.co/settings/tokens).
3. The script automatically:
   - Creates the Space (`neuro-agent`) on Hugging Face using Docker SDK.
   - Synchronizes production models, backend, and compiled React UI.
   - Triggers the 24/7 cloud container build.
4. Your permanent 24/7 link is ready:
   `https://huggingface.co/spaces/<your-username>/neuro-agent`
   *(It stays online forever, completely independent of whether your laptop is on or off).*

---

## Architecture Summary
- **Single Port:** Backend FastAPI serves both the REST/SSE APIs and the compiled React SPA from `frontend_react/dist`.
- **Zero CORS Issues:** The frontend communicates relatively with the backend (`backendUrl = ""`), ensuring flawless operation over any domain or tunnel.
