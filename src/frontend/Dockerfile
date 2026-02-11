FROM node:22-slim AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --quiet

COPY . .
RUN npm run build

FROM nginx:alpine

ARG BACKEND_PORT
ARG FRONTEND_PORT

ENV BACKEND_PORT=${BACKEND_PORT:-8080}
ENV FRONTEND_PORT=${FRONTEND_PORT:-3030}

RUN addgroup -g 101 -S nginx || true && \
    adduser -S -D -H -u 101 -h /var/cache/nginx -s /sbin/nologin -G nginx -g nginx nginx || true && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid && \
    mkdir -p /usr/share/nginx/html && \
    chown -R nginx:nginx /usr/share/nginx/html

COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html
COPY --chown=nginx:nginx nginx.conf.template /etc/nginx/templates/default.conf.template

USER nginx

EXPOSE ${FRONTEND_PORT}

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider --prefer-family=IPv4 http://127.0.0.1:${FRONTEND_PORT}/ || exit 1


CMD ["nginx", "-g", "daemon off;"]