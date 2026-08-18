# syntax=docker/dockerfile:1

FROM node:22-alpine3.20 AS open-webui-build

# Pin the exact commit behind Open WebUI v0.9.5 so the patch cannot silently
# apply to a different upstream release.
ARG OPEN_WEBUI_COMMIT=3660bc00fd807deced3400a63bfa6db47811a3bb

RUN apk add --no-cache git
WORKDIR /src
RUN git init \
    && git remote add origin https://github.com/open-webui/open-webui.git \
    && git fetch --depth 1 origin "${OPEN_WEBUI_COMMIT}" \
    && git checkout --detach FETCH_HEAD

RUN npm ci --force

COPY docker/cortex/open-webui/home-cortex-greeting.patch /tmp/home-cortex-greeting.patch
RUN git apply --check /tmp/home-cortex-greeting.patch \
    && git apply /tmp/home-cortex-greeting.patch

ENV APP_BUILD_HASH=home-cortex-v0.9.5
RUN npm run build

FROM ghcr.io/open-webui/open-webui:v0.9.5

COPY --from=open-webui-build /src/build /app/build
COPY --from=open-webui-build /src/backend/open_webui/main.py /app/backend/open_webui/main.py
COPY --from=open-webui-build /src/backend/open_webui/routers/cortex.py /app/backend/open_webui/routers/cortex.py
