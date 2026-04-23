# Blackjack Frontend

This frontend is a lightweight Vite + React UI inspired by the AlumniConnect frontend structure:
Since I co-authored that project, I thought it would be good to re-use its components.
Link to AlumniConnect repo: https://github.com/Chris-B33/AlumniConnect

- componentized UI (`components/`)
- API wrapper (`services/api.js`)
- env-driven backend URL (`.env`)

## Local run

```bash
cd frontend
npm install
cp .env.example .env
```

Set `VITE_API_BASE_URL` in `.env`:

```env
VITE_API_BASE_URL=https://<api-id>.execute-api.<region>.amazonaws.com
VITE_PREDICT_PATH=/predict
```

Start dev server:

```bash
npm run dev
```
