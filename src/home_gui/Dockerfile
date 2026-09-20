FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-alpine
WORKDIR /app
RUN npm install -g serve@14.2.4
COPY --from=build /app/dist ./dist
COPY serve.json ./dist/serve.json
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]
