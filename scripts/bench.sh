#!/usr/bin/env bash
# Run a Criterion bench with low-noise mitigations applied.
#
# Default (no sudo): pins to one core via taskset and disables ASLR via setarch.
# Optional (with sudo): also sets the perf governor, disables Intel turbo, and
# runs at SCHED_FIFO/99. The script gracefully degrades — sudo is not required.
#
# Hybrid-CPU aware: if CORE is unset, claims a free core in the perf cluster
# (cores at max cpufreq) via flock so concurrent runs don't collide. Falls
# back to the full cluster CPU list when every core is claimed.
#
# Optional env knobs:
#   BENCH=all        cargo bench --bench target (default all = speedup + latency)
#   CORE=            physical core to pin to via taskset; auto-claim if unset
#   BASELINE=        if set, passes --baseline <name>; else --save-baseline ref
#   LOCK_DIR=        per-core flock files (default /tmp/opthash-bench-locks)
#
# Forwarded args (after `--`) are appended to the Criterion command line — pass
# `--measurement-time 10 --sample-size 200` here for tighter confidence bands.

set -euo pipefail

BENCH=${BENCH:-all}
BASELINE=${BASELINE:-}
LOCK_DIR=${LOCK_DIR:-/tmp/opthash-bench-locks}

# Low-noise primitives below are Linux-only; elsewhere we fall through to
# plain `cargo bench`.
IS_LINUX=0
[[ $(uname) == Linux ]] && IS_LINUX=1

# Cores at the system's max cpufreq — perf cluster on hybrid SoCs, every
# core on homogeneous CPUs.
detect_perf_cores() {
	local max_freq=0
	local path f
	for path in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do
		f=$(<"$path") 2>/dev/null || continue
		((f > max_freq)) && max_freq=$f
	done
	for path in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do
		f=$(<"$path") 2>/dev/null || continue
		if ((f == max_freq)); then
			local c=${path#/sys/devices/system/cpu/cpu}
			c=${c%/cpufreq/cpuinfo_max_freq}
			echo "$c"
		fi
	done
}

# Lock fd is held for the script's lifetime; the kernel releases it on exit.
claim_perf_core() {
	mkdir -p "$LOCK_DIR" 2>/dev/null || true
	local perf_cores=()
	while IFS= read -r c; do perf_cores+=("$c"); done < <(detect_perf_cores)
	if ((${#perf_cores[@]} == 0)); then
		echo "warn: no cpufreq info; defaulting to CORE=0" >&2
		CORE=0
		return
	fi
	local c lock
	for c in "${perf_cores[@]}"; do
		lock="$LOCK_DIR/core-${c}.lock"
		# Permission-denied (e.g. lock owned by another user) → try next core.
		exec {LOCK_FD}>"$lock" 2>/dev/null || continue
		if flock -n "$LOCK_FD"; then
			CORE=$c
			echo "info: claimed perf core $c (lock: $lock)" >&2
			return
		fi
		exec {LOCK_FD}>&-
		unset LOCK_FD
	done
	# All perf cores busy or unlockable: restrict to the cluster, let OS schedule.
	CORE=$(
		IFS=,
		echo "${perf_cores[*]}"
	)
	echo "info: all perf cores busy; restricting to cluster CORE=$CORE" >&2
}

if ((IS_LINUX)) && [[ -z ${CORE:-} ]]; then
	claim_perf_core
fi

# sudo strips PATH/HOME; recover invoker's rustup so the shim resolves their
# default toolchain.
if [[ -n "${SUDO_USER:-}" ]]; then
	user_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
	if [[ -x "$user_home/.cargo/bin/cargo" ]]; then
		export PATH="$user_home/.cargo/bin:$PATH"
		export CARGO_HOME="${CARGO_HOME:-$user_home/.cargo}"
		export RUSTUP_HOME="${RUSTUP_HOME:-$user_home/.rustup}"
	fi
fi
command -v cargo >/dev/null 2>&1 || {
	echo "error: cargo not found in PATH" >&2
	exit 1
}

if ((IS_LINUX)) && [[ $EUID -eq 0 ]]; then
	if command -v cpupower >/dev/null 2>&1; then
		cpupower frequency-set -g performance >/dev/null
	fi
	if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
		echo 1 >/sys/devices/system/cpu/intel_pstate/no_turbo
	fi
fi

if [[ -n "$BASELINE" ]]; then
	criterion_args=(--baseline "$BASELINE")
else
	criterion_args=(--save-baseline ref)
fi

if [[ "$BENCH" == "all" ]]; then
	bench_targets=(speedup latency)
else
	bench_targets=("$BENCH")
fi

# Under sudo, prefix chrt and drop back to invoking user so build artifacts
# stay user-owned. SCHED_FIFO survives the UID drop (process attribute).
launcher=()
if ((IS_LINUX)) && [[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] && command -v chrt >/dev/null 2>&1; then
	launcher=(chrt -f 99 sudo -u "$SUDO_USER"
		--preserve-env=PATH,CARGO_HOME,RUSTUP_HOME --)
fi

pin_wrapper=()
if ((IS_LINUX)) && [[ -n "${CORE:-}" ]]; then
	pin_wrapper=(taskset -c "$CORE" setarch -R)
fi

for target in "${bench_targets[@]}"; do
	cmd=("${pin_wrapper[@]}" cargo bench --bench "$target" -- "${criterion_args[@]}" "$@")
	"${launcher[@]}" "${cmd[@]}"
done
