# Gateway - For Developers

- Supported OS: Linux, macOS


### Setup for Development

1. Clone the repository
```shell
git clone https://app.git.valerii.cc/valerii/gateway.git
cd gateway
```

2. Install [uv](https://github.com/astral-sh/uv)
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

3. Install package & deps
```sh
uv venv
uv sync --dev
```

4. Generate protobuf files
```sh
uv run scripts/gen_proto.py
```

### Adding/ Removing packages
```sh
uv add requests --optional core
uv remove request --optional core
```
or adding to a group e.g., development
```sh
uv add requests --dev
uv remove requests --dev
```

### Upgrading a version
1. Bump up version in `pyproject.toml`
2. Execute
```sh
git tag v0.1.4
git push origin v0.1.4
```

### Tools for Development

#### Static Type Checker
```sh
uv run pyright
```

#### Testing
Run all the tests
```sh
uv run pytest
```

Show all testing markers 
```sh
uv run pytest --markers | head -1
```

Run tests assigned to a marker
```sh
uv run pytest -m "marker"
```
