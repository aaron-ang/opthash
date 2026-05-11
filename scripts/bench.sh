#!/usr/bin/env bash
# Run a Criterion bench with low-noise mitigations applied.
#
# Default (no sudo): pins to one core via taskset and disables ASLR via setarch.
# Optional (with sudo): also sets the perf governor, disables Intel turbo, and
# runs at SCHED_FIFO/99. The script gracefully degrades — sudo is not required.
#
# Optional env knobs:
#   BENCH=speedup    cargo bench --bench target (default speedup)
#   CORE=2           physical core to pin to via taskset
#   BASELINE=        if set, passes --baseline <name>; else --save-baseline ref
#
# Forwarded args (after `--`) are appended to the Criterion command line — pass
# `--measurement-time 10 --sample-size 200` here for tighter confidence bands.

set -euo pipefail

BENCH=${BENCH:-speedup}
CORE=${CORE:-2}
BASELINE=${BASELINE:-}

# sudo strips PATH and points HOME at /root; recover the invoking user's
# rustup + cargo so the rustup shim can resolve their default toolchain.
if [[ -n "${SUDO_USER:-}" ]]; then
    user_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    if [[ -x "$user_home/.cargo/bin/cargo" ]]; then
        export PATH="$user_home/.cargo/bin:$PATH"
        export CARGO_HOME="${CARGO_HOME:-$user_home/.cargo}"
        export RUSTUP_HOME="${RUSTUP_HOME:-$user_home/.rustup}"
    fi
fi
command -v cargo >/dev/null 2>&1 || { echo "error: cargo not found in PATH" >&2; exit 1; }

if [[ $EUID -eq 0 ]]; then
    if command -v cpupower >/dev/null 2>&1; then
        cpupower frequency-set -g performance >/dev/null
    fi
    if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
    fi
fi

if [[ -n "$BASELINE" ]]; then
    criterion_args=(--baseline "$BASELINE")
else
    criterion_args=(--save-baseline ref)
fi

cmd=(taskset -c "$CORE" setarch -R cargo bench --bench "$BENCH" -- "${criterion_args[@]}" "$@")

# Under sudo, prefix with chrt and drop back to the invoking user for cargo
# so build artifacts stay user-owned. SCHED_FIFO survives the UID drop —
# it's a process attribute, not a credential.
launcher=()
if [[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] && command -v chrt >/dev/null 2>&1; then
    launcher=(chrt -f 99 sudo -u "$SUDO_USER"
        --preserve-env=PATH,CARGO_HOME,RUSTUP_HOME --)
fi

exec "${launcher[@]}" "${cmd[@]}"
