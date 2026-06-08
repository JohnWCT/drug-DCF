#!/usr/bin/env bash
# Install RTK (Rust Token Killer) inside the DAPL Docker container.
# Usage (from host):
#   docker exec DAPL bash /workspace/DAPL/scripts/install_rtk.sh
#
# RTK: https://github.com/rtk-ai/rtk
# Compresses CLI output before it reaches LLM context (60-90% token savings).

set -euo pipefail

RTK_INSTALL_DIR="${RTK_INSTALL_DIR:-${HOME}/.local/bin}"
RTK_VERSION="${RTK_VERSION:-}"  # e.g. v0.42.3; empty = latest

info()  { printf '\033[0;32m[INFO]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
error() { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

detect_target() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    case "${os}:${arch}" in
        Linux:x86_64|Linux:amd64)  echo "x86_64-unknown-linux-musl" ;;
        Linux:aarch64|Linux:arm64)   echo "aarch64-unknown-linux-gnu" ;;
        Darwin:x86_64|Darwin:amd64)  echo "x86_64-apple-darwin" ;;
        Darwin:arm64|Darwin:aarch64) echo "aarch64-apple-darwin" ;;
        *) error "Unsupported platform: ${os} ${arch}" ;;
    esac
}

resolve_version() {
    if [[ -n "${RTK_VERSION}" ]]; then
        echo "${RTK_VERSION}"
        return
    fi
    curl -sI "https://github.com/rtk-ai/rtk/releases/latest" \
        | grep -i '^location:' \
        | sed -E 's|.*/tag/([^[:space:]]+).*|\1|' \
        | tr -d '\r'
}

install_binary() {
    local version target url tmp archive
    version="$(resolve_version)"
    [[ -n "${version}" ]] || error "Could not resolve RTK release version"

    target="$(detect_target)"
    url="https://github.com/rtk-ai/rtk/releases/download/${version}/rtk-${target}.tar.gz"

    info "Installing RTK ${version} (${target})"
    tmp="$(mktemp -d)"
    archive="${tmp}/rtk.tar.gz"

    curl -fsSL "${url}" -o "${archive}"
    if tar -tzf "${archive}" | grep -qE '^/|(^|/)\.\.(/|$)'; then
        rm -rf "${tmp}"
        error "Unsafe paths in archive — aborting"
    fi
    tar -xzf "${archive}" -C "${tmp}"

    mkdir -p "${RTK_INSTALL_DIR}"
    install -m 755 "${tmp}/rtk" "${RTK_INSTALL_DIR}/rtk"
    ln -sf "${RTK_INSTALL_DIR}/rtk" /usr/local/bin/rtk
    rm -rf "${tmp}"
    info "Binary installed: ${RTK_INSTALL_DIR}/rtk"
}

configure_path() {
    local line='export PATH="${HOME}/.local/bin:${PATH}"'
    local profile_d="/etc/profile.d/rtk.sh"

    # docker exec bash -lc is non-interactive; .bashrc returns early ([ -z "$PS1" ]).
    # /etc/profile.d/ is sourced by login shells before that guard runs.
    printf '%s\n' "# RTK (Rust Token Killer)" "${line}" > "${profile_d}"
    chmod 644 "${profile_d}"
    info "Added PATH to ${profile_d}"

    export PATH="${RTK_INSTALL_DIR}:${PATH}"
}

verify() {
    if command -v rtk >/dev/null 2>&1; then
        info "Verification OK: $(rtk --version)"
    else
        warn "rtk not in PATH yet. Run: export PATH=\"\${HOME}/.local/bin:\${PATH}\""
    fi
}

main() {
    install_binary
    configure_path
    verify
    info "Done. Example: rtk git status"
}

main "$@"
