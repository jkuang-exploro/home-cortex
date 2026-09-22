"""Static regression guards for the independent media-service boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEDIA_SOURCE = ROOT / "home-media" / "src" / "home_media"
GUI_SOURCE = ROOT / "src" / "home_gui" / "src"


def test_media_and_cortex_python_packages_do_not_import_each_other() -> None:
    assert not (ROOT / "home_media").exists()
    assert MEDIA_SOURCE.is_dir()
    media_text = "\n".join(path.read_text() for path in MEDIA_SOURCE.glob("*.py"))
    cortex_text = "\n".join(
        path.read_text() for path in (ROOT / "src" / "home_cortex").rglob("*.py")
    )
    assert "import home_cortex" not in media_text
    assert "from home_cortex" not in media_text
    assert "import home_media" not in cortex_text
    assert "from home_media" not in cortex_text


def test_gui_has_native_media_route_and_incremental_controls() -> None:
    app = (GUI_SOURCE / "App.svelte").read_text()
    media = (GUI_SOURCE / "components" / "Media.svelte").read_text()
    media_api = (GUI_SOURCE / "lib" / "media.ts").read_text()
    sidebar = (GUI_SOURCE / "components" / "Sidebar.svelte").read_text()
    assert "window.location.pathname === '/media'" in app
    assert "<Media" in app
    assert ">Media</button>" in sidebar
    assert "Load more" in media
    assert "<video" in media
    assert "ArrowLeft" in media and "ArrowRight" in media and "Escape" in media
    assert "15_000" in media_api
    assert "600_000" in media_api
    assert "isMediaPage" in media_api
    assert "returned the web app instead of JSON" in media_api
