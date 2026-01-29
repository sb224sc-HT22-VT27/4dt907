FROM node:22-slim AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

ARG BACKEND_PORT
ENV BACKEND_PORT=${BACKEND_PORT:-8080}
ARG FRONTEND_PORT
ENV FRONTEND_PORT=${FRONTEND_PORT:-3030}

COPY --from=builder /app/dist /usr/share/nginx/html

COPY nginx.conf.template /etc/nginx/templates/default.conf.template

EXPOSE ${FRONTEND_PORT}

CMD ["nginx", "-g", "daemon off;"]