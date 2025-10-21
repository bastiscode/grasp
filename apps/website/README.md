# GRASP Website

HTML-first port of the GRASP conversational interface. This project mirrors the Flutter-based web app while relying on semantic HTML, accessible styling, and a lightweight Svelte component layer.

## Development

```bash
cd apps/website
npm install
npm run dev
```

The dev server runs at http://localhost:5173 by default and supports hot module reloading. The app code lives under `src/`, with shared configuration and components in `src/lib/`.

## Building for production

```bash
npm run build
```

The static site is emitted to `dist/`. You can copy that folder into an nginx (or any static) Docker image, similar to the existing Flutter build workflow.

### Docker

```bash
docker build -t grasp-website .
docker run -p 8080:80 grasp-website
```

The image stages the Vite build on Node 22 Alpine and serves the static assets via `nginx:alpine-slim`.
