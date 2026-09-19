# Vision frontend boundary

Vision UI work is paused pending MicroDuck hardware. Home GUI currently exposes
no Vision page, and the Home Cortex API currently exposes no Vision routes.

## Ownership

`home_gui` owns future browser interaction: source selection, observation
history, evidence review, and human enrollment actions. It may consume stable
Home Cortex HTTP DTOs and browser-compatible live-stream endpoints. It must not
import `home_cortex_client`, detector libraries, tracker state, device files, or
backend persistence implementations.

`home_cortex` owns the future browser-facing API, authorization, normalized
observation/evidence DTOs, artifact access policy, and reconciliation actions.
It does not proxy or transcode device video.

`home_cortex_client` owns camera capture and media publication. It may advertise
a browser-compatible stream endpoint through a future backend discovery seam,
but the GUI never controls its Python runtime directly.

```text
home_cortex_client -- encoded stream --> browser or media gateway
home_cortex_client -- serialized evidence --> home_cortex -- API --> home_gui
```

## Stable constraints

- Browser DTOs are derived from backend domain records; they do not expose
  detector tensors, model-native class IDs, embeddings, filesystem paths, raw
  camera credentials, or direct database operations.
- The browser never receives a device RTSP URL. If MicroDuck uses RTSP, a media
  gateway may translate it to a browser-compatible authenticated transport.
- Live-stream availability is advisory. Historical observations and evidence
  remain useful when the device is offline.
- Enrollment remains an explicit authenticated human action over an existing
  household item; visual evidence alone never mutates household facts.
- Missing localization is represented as missing data, not a fabricated pose.

No route names, page components, gateway product, or polling mechanism are
committed while the feature is paused. Those choices should be made only after
the MicroDuck transport and backend ingestion adapter exist.
