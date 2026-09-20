"""Static guards for the Compose/nginx public network boundary."""

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker" / "docker-compose.yml"
NGINX = ROOT / "docker" / "proxy" / "nginx.conf"
GUI_DOCKERFILE = ROOT / "src" / "home_gui" / "Dockerfile"


def test_nginx_is_the_only_lan_facing_web_entrypoint() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]

    assert services["proxy"]["ports"] == ["80:80"]
    assert "ports" not in services["home-gui"]
    assert services["home-gui"]["expose"] == ["3000"]
    assert services["cortex-api"]["ports"] == ["127.0.0.1:8001:8000"]


def test_nginx_routes_gui_and_api_over_the_compose_network() -> None:
    config = NGINX.read_text()

    assert "listen 80;" in config
    assert "listen 443" not in config
    assert "server_name home-cortex-0;" in config
    assert "resolver 127.0.0.11" in config
    assert "server home-gui:3000 resolve;" in config
    assert "server cortex-api:8000 resolve;" in config
    assert len(re.findall(r"^\s*location / \{", config, re.MULTILINE)) == 1
    for route in (
        "/session",
        "/conversations",
        "/agent/",
        "/v1/",
        "/health",
        "/admin/",
        "/docs",
        "/redoc",
        "/openapi.json",
    ):
        assert route in config


def test_production_gui_container_listens_on_internal_port_3000() -> None:
    dockerfile = GUI_DOCKERFILE.read_text()

    assert "EXPOSE 3000" in dockerfile
    assert '["serve", "-s", "dist", "-l", "3000"]' in dockerfile


def test_user_facing_docs_do_not_require_port_3000() -> None:
    for path in (ROOT / "Readme.md", ROOT / "src" / "README.md", ROOT / "setup_v0.2.md"):
        assert ":3000" not in path.read_text()
