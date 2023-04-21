from typer_builder import build_app_from_module

if __name__ == "__main__":
    app = build_app_from_module("commands")
    app()
