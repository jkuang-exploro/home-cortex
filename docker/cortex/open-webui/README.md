# Home Cortex Open WebUI customization

This image adds proactive greetings for the `老管家` virtual model to Open
WebUI v0.9.5.

The patch is applied to the exact upstream commit pinned in `Dockerfile`. It
adds:

- an authenticated Open WebUI backend proxy for Cortex conversation creation;
- a frontend hook that requests a greeting when `老管家` is selected in a
  blank chat; and
- persistence of that greeting as the root assistant message in normal Open
  WebUI chat history.

The browser sends its existing Open WebUI session token to the proxy. The
proxy, not the browser, adds `CORTEX_API_KEY` and forwards the verified Open
WebUI user ID and email to Cortex.

Build and deploy from `docker/cortex`:

```sh
docker compose build open-webui
docker compose up -d --force-recreate --no-deps open-webui
```

When upgrading Open WebUI, update the base image and pinned source commit
together, then rebase `home-cortex-greeting.patch` and run a complete frontend
production build before deployment.
