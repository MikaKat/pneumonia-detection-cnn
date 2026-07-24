# --- Stage 1: build the static bundle -------------------------------------
FROM node:20-alpine AS build
WORKDIR /app

# Install deps first (better layer caching)
COPY package.json package-lock.json* ./
RUN npm ci || npm install

# Build
COPY . .
# VITE_API_BASE can be baked at build time; defaults to /api (same-origin proxy).
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# --- Stage 2: serve with nginx --------------------------------------------
FROM nginx:1.27-alpine AS serve

# nginx template: envsubst fills ${API_UPSTREAM} at container start so you can
# point the /api proxy at your Python service without rebuilding the image.
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

# Where nginx forwards /api. Override in Coolify, e.g. http://backend:8000
ENV API_UPSTREAM=http://backend:8000

EXPOSE 80
# nginx's own entrypoint runs envsubst over /etc/nginx/templates/*.template
CMD ["nginx", "-g", "daemon off;"]
