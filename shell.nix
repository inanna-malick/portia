{ pkgs ? import <nixpkgs> {} }:

let
  libPath = pkgs.lib.makeLibraryPath (
    [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]
    ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
      pkgs.glib
    ]
  );
in
pkgs.mkShell {
  name = "portia-dev-shell";

  buildInputs = [
    pkgs.rustc
    pkgs.cargo
    pkgs.clippy
    pkgs.rustfmt
    pkgs.pkg-config
    pkgs.rustPlatform.bindgenHook
    pkgs.libclang
    pkgs.python3
    pkgs.uv
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];

  LIBCLANG_PATH = "${pkgs.lib.getLib pkgs.libclang}/lib";

  shellHook = ''
    ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
      export LD_LIBRARY_PATH="/run/opengl-driver/lib:${libPath}:''${LD_LIBRARY_PATH:-}"
    ''}
    export TRANSFORMERS_NO_TORCHVISION=1
    export PYTORCH_ENABLE_MPS_FALLBACK=1

    if [ -f pyproject.toml ] && [ -f uv.lock ]; then
      LOCK_HASH="$(cksum uv.lock | awk '{print $1 "-" $2}')"
      STAMP_FILE=".venv/.portia_uv_lock_stamp"
      if [ ! -x ".venv/bin/python" ] || [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE" 2>/dev/null)" != "$LOCK_HASH" ]; then
        echo "[portia nix-shell] Syncing pinned Python environment (.venv) from uv.lock..."
        uv sync --locked --python "${pkgs.python3}/bin/python3"
        echo "$LOCK_HASH" > "$STAMP_FILE"
      fi
      export PORTIA_PYTHON="$PWD/.venv/bin/python"
      export VIRTUAL_ENV="$PWD/.venv"
      export PATH="$PWD/target/release:$PWD/target/debug:$PWD/.venv/bin:$PATH"
    fi

    echo "[portia nix-shell] Ready (PORTIA_PYTHON=''${PORTIA_PYTHON:-python3})."
    echo "  Build CLI : cargo build --release -p portia-cli"
    echo "  Open TUI  : cargo run --release -p portia-cli -- tui --db examples/bgp_incidents.sqlite"
  '';
}
