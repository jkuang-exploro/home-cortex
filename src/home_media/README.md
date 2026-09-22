# home-media

`home-media` is the independent, read-only photo and video service used by the
Home Cortex web application. It recursively indexes configured source roots into
a rebuildable SQLite database and generates thumbnails or video posters lazily.
Its import package remains `home_media`, because Python identifiers cannot contain
hyphens. It has no Python dependency on `home_cortex`.

## Runtime contract

The container expects these paths by default:

- `/media/photo` — original photos, mounted read-only;
- `/media/video` — original videos, mounted read-only;
- `/data` — writable SQLite index and derivative cache.

Configuration is available through `HOME_MEDIA_PHOTO_ROOT`,
`HOME_MEDIA_VIDEO_ROOT`, `HOME_MEDIA_DATA_ROOT`,
`HOME_MEDIA_THUMBNAIL_SIZE`, `HOME_MEDIA_PAGE_SIZE`,
`HOME_MEDIA_FFPROBE_PATH`, and `HOME_MEDIA_FFMPEG_PATH`.

The index is not populated during startup. Call `POST /media-api/refresh` through
the deployment proxy. Refresh compares relative path, byte size, and nanosecond
mtime, probing only new or changed files. Deleting `/data/media.db` and the
thumbnail directory is safe; both are derived from the source roots.

Media IDs are `media_` plus the first 24 hexadecimal characters of SHA-256 over
`media type + NUL + POSIX relative path`. Browser-facing responses never contain
absolute paths. Each content lookup starts from an indexed ID and revalidates the
resolved path against its configured canonical root, preventing traversal and
symlink escape.

Photo timestamp priority is EXIF `DateTimeOriginal`, `DateTimeDigitized`, general
EXIF `DateTime`, then filesystem mtime. Video timestamp priority is ffprobe stream
creation time, container creation time, then filesystem mtime. Naive embedded
timestamps are interpreted in the service host's local timezone and serialized as
UTC. Photo thumbnails apply EXIF orientation. ffmpeg applies video display
rotation while extracting posters.

Authentication stays at nginx: `/media-api/` uses nginx `auth_request` against
Home Cortex `/session`. The service itself is exposed only on the private Compose
network and has no published host port.

## Local development

```bash
uv sync --project src/home_media --extra dev
uv run --project src/home_media --extra dev pytest
uv run --project src/home_media uvicorn home_media.app:app --reload --port 8002
```

All source-media operations are reads. V1 deliberately has no upload, delete,
rename, move, edit, album, favorite, sharing, AI, or semantic-search API.
