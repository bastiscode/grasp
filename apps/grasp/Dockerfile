FROM ghcr.io/cirruslabs/flutter AS build

WORKDIR /app
COPY . .
RUN flutter build web --release --no-tree-shake-icons

FROM nginx:alpine-slim
COPY --from=build /app/build/web /usr/share/nginx/html
