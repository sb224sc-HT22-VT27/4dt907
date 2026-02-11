FROM node:22-slim AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --quiet

COPY . .
RUN npm run build

FROM nginx:alpine

ARG BACKEND_PORT
ARG FRONTEND_PORT

ENV BACKEND_PORT=${BACKEND_PORT}
ENV FRONTEND_PORT=${FRONTEND_PORT}

COPY --from=builder /app/dist /usr/share/nginx/html

COPY nginx.conf.template /etc/nginx/templates/default.conf.template

RUN touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid /var/cache/nginx /var/log/nginx /etc/nginx/conf.d

USER nginx

EXPOSE ${FRONTEND_PORT}

CMD ["nginx", "-g", "daemon off;"]