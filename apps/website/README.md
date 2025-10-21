# GRASP Website (SvelteKit)

SvelteKit port of the GRASP conversational interface. The project mirrors the existing Flutter-based web app while relying on semantic HTML, accessible styling, and a lightweight Svelte component layer.

## Requirements

- Node.js 20.19+ (or 22.12+/24+) — the tooling enforces this.  
  We recommend using `nvm` to manage versions.
- npm (ships with Node).

Install project dependencies once:

```bash
cd apps/website_svelte_kit
npm install
```

## Development

```bash
npm run dev
```

The dev server runs at http://localhost:5173 with hot module replacement. Source code lives under `src/` with shared components in `src/lib/`. Static assets (favicons, robots.txt, etc.) reside in `static/`.

## Building for production

```bash
npm run build
```

With the static adapter, the production build is emitted to `build/`. You can preview the output locally with:

```bash
npm run preview
```

## Docker

```bash
docker build -t grasp-website .
docker run -p 8080:80 grasp-website
```

The multi-stage Dockerfile compiles the static build using Node 22 Alpine, then serves the exported site via `nginx:alpine-slim`.
