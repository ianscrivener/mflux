---
name: macmon
description: Read Apple Silicon hardware metrics with the macmon CLI. Use when the user wants CPU, GPU, or ANE power, or chip temperature. Use for memory pressure, fan speed, battery state, or network and disk throughput. Use for load tests on Apple Silicon macOS. Triggers on macmon, chip power, cpu temp, gpu power, ane power, memory pressure, fan speed, Apple Silicon metrics.
---

# macmon — Apple Silicon metrics from the CLI

## Purpose

macmon is a sudoless performance monitor for Apple Silicon Macs. It reads the same data as powermetrics, without root access. It supports macOS 14 and later, chips M1 to M6.

- Repo: https://github.com/vladkens/macmon
- License: MIT
- macmon is a Rust binary and library. This skill covers the CLI.

## When to use this skill

Use the skill when the user asks for any of these:

- CPU, GPU, or ANE power draw.
- CPU or GPU frequency and utilization.
- CPU, GPU, or chip temperature.
- RAM use, swap use, or memory pressure.
- Fan speed.
- Battery charge or plug state.
- Network or disk throughput.
- Load generation to test a metrics pipeline.
- Metrics for Prometheus or Grafana.

## Before you measure

1. Run `macmon --version` to check the tool.
2. If the command fails, install macmon.
3. Read `macmon --help` for the current command set. The tool evolves.

## Install

| Method | Command |
| --- | --- |
| Homebrew | `brew install macmon` |
| MacPorts | `sudo port install macmon` |
| Cargo | `cargo install macmon` |
| Nix | `nix-env -i macmon` |

## Command map

Top level: `macmon [OPTIONS] [COMMAND]`

Global options:

| Option | Meaning |
| --- | --- |
| `-i, --interval <MS>` | Update interval in milliseconds. Default 1000. |
| `-h, --help` | Print help. |
| `-V, --version` | Print version. |

Commands:

| Command | Purpose |
| --- | --- |
| `pipe` | Print one JSON object per sample to stdout. Use for automation. |
| `serve` | Serve metrics over HTTP. JSON at /json, Prometheus at /metrics. |
| `debug` | Print sensor and hardware state. Use for troubleshooting. |
| `find-clpc` | Find CLPC power counters. Use for maintainers. |
| `stress` | Generate CPU, GPU, or ANE load. Use to test a metrics pipeline. |
| `help` | Print help for a command. Example: `macmon help pipe`. |

## Measure with macmon pipe

`pipe` is the command to measure.

Options:

| Option | Meaning |
| --- | --- |
| `-s, --samples <N>` | Number of samples. Default 0 = run until stopped. |
| `-i, --interval <MS>` | Interval in milliseconds. Default 1000. |
| `--soc-info` | Add the soc object with chip and model data to each sample. |

Examples:

```bash
macmon pipe -s 1              # one sample, then exit
macmon pipe -s 5 -i 500       # five samples at 500 ms
macmon pipe -s 1 --soc-info   # one sample with chip data
macmon pipe -s 0 | jq         # run until Ctrl-C (avoid in scripts)
macmon pipe -s 10 -i 1000 | jq -r '.chip_power, .gpu_power'
```

Rules:

- Always set `-s` in a script. A run with `-s 0` never ends on its own.
- Treat each output line as one JSON object. The output is newline delimited.
- Parse with `jq` or with python3.

Illustrative sample:

```json
{
  "timestamp": "2026-11-18T12:00:00+00:00",
  "temp": {"cpu_temp_avg": 45.2, "gpu_temp_avg": 41.0},
  "memory": {"ram_total": 38654705664, "ram_usage": 15090450432, "pressure_level": 0, "pressure_percent": 14},
  "cpu_power": 1.28, "gpu_power": 0.45, "ane_power": 0.0, "chip_power": 3.32,
  "cpu_tiers": [{"label": "E", "freq_mhz": 1937, "scaled_ratio": 0.166, "active_ratio": 0.227, "cores": []}]
}
```

## JSON output reference

Units and null semantics carry meaning. A null field means "no valid sensor reading". Do not read null as zero.

### Timestamp and memory

| Field | Unit or format | Meaning |
| --- | --- | --- |
| timestamp | ISO 8601 UTC | Sample time. |
| memory.ram_total, memory.ram_usage | Bytes | Physical RAM. |
| memory.swap_total, memory.swap_usage | Bytes | Swap file. |
| memory.pressure_level | 0, 1, or 2 | 0 normal, 1 warning, 2 critical. Null when unavailable. |
| memory.pressure_percent | 0–100 | Pressure as in Activity Monitor. Null when unavailable. |

### Temperature

| Field | Unit | Meaning |
| --- | --- | --- |
| temp.cpu_temp_avg | Celsius | Average CPU temperature. Null when no sensor gives a valid reading. |
| temp.gpu_temp_avg | Celsius | Average GPU temperature. Null when no sensor gives a valid reading. |

### Fans and battery

| Field | Unit or format | Meaning |
| --- | --- | --- |
| fans | Array | One entry per fan. Each entry: name, rpm, max_rpm. |
| battery_percent | 0–100 | Charge. Null without a battery. |
| battery_charging | Bool | True while charging. Null without a battery. |
| battery_plugged | Bool | True while the Mac uses external power. Null without a battery. |

### CPU

| Field | Meaning |
| --- | --- |
| cpu_active_ratio | 0–1. Share of the interval with CPU work. Not frequency adjusted. Weighted by core count. |
| cpu_scaled_ratio | 0–1. The same share adjusted for operating frequency. Weighted by core count. |
| cpu_tiers | One object per core tier. Each object: label, freq_mhz, scaled_ratio, active_ratio, cores. |

Each core in cpu_tiers has: die_id, core_id, freq_mhz, scaled_ratio, active_ratio.

Tier labels by chip:

| Chip | Labels |
| --- | --- |
| M1 to M4 | E, P |
| M5 | E, S |
| M5 Pro/Max | P, S |
| M6 | E, P, S |
| Unknown type (for example a VM) | `_` |

Deprecated top-level fields: ecpu_freq_mhz, ecpu_scaled_ratio, ecpu_active_ratio, ecpu_cores for the lowest tier, and pcpu_* for the highest tier. Use cpu_tiers instead.

### GPU

| Field | Unit | Meaning |
| --- | --- | --- |
| gpu_freq_mhz | MHz | GPU frequency. |
| gpu_active_ratio | 0–1 | Share of the interval with GPU work. Not frequency adjusted. |
| gpu_scaled_ratio | 0–1 | GPU share adjusted for operating frequency. |
| gpu_mem_alloc | Bytes | System memory allocated by the GPU. Null when unavailable. |
| gpu_mem_in_use | Bytes | System memory in use by the GPU. Null when unavailable. |

### Power

All power fields are Watts.

| Field | Meaning |
| --- | --- |
| cpu_power | CPU draw. |
| gpu_power | GPU draw. |
| ane_power | Neural Engine draw. |
| chip_power | Whole chip. Never less than the sum of cpu, gpu, ane, ram, dcs, and media power. |
| sys_power | Whole system. On a Mac without a battery, macmon reads it from the power supply. |
| ram_power | DRAM draw. |
| dcs_power | Memory controller draw. Null without a source. |
| media_power | Video encoder and image processor. macmon reports it only on M6. Null elsewhere. |
| dc_in_power | Adapter input. Null without a source. |
| display_power | Built-in display backlight. Null without a source. |

macmon deprecates all_power and gpu_ram_power. all_power equals chip_power.

### Network, disk, uptime

| Field | Unit | Meaning |
| --- | --- | --- |
| net_in, net_out | Bytes per second | Ethernet and Wi-Fi traffic. Null without such an interface. |
| disk_read, disk_write | Bytes per second | Disk traffic, without disk images. Null when unavailable. |
| uptime | Seconds | Time since boot. Null when unavailable. |

### Extra fields in newer builds

macmon 0.12.0 also prints cpu_usage_pct, gpu_usage, ecpu_usage, and pcpu_usage. The README does not document them. Do not rely on them. The tables above are authoritative.

## How to interpret ratios

- active_ratio is the share of the interval in which the component did work.
- scaled_ratio is the same share, adjusted by the operating frequency.
- Formula: R_active = Σtᵢ/T. Formula: R_scaled = Σtᵢfᵢ/(T·f_max).
- Tier ratios average the tier cores. The combined cpu ratios average all cores from all tiers.
- A zero power or frequency reading can mean "counter unavailable", not "no consumption".

## Serve over HTTP with macmon serve

Endpoints:

| Endpoint | Format | Content |
| --- | --- | --- |
| GET /json | JSON | Current sample. Same fields as `pipe --soc-info`. |
| GET /metrics | Prometheus text | All metrics as gauges. |

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | 0.0.0.0 | Listen address. Use 127.0.0.1 for local only. |
| `-p, --port` | 9090 | Port. |
| `-i, --interval` | 1000 | Interval in milliseconds. |
| `--install` | Not installed | Install a LaunchDaemon. Starts at boot. Needs sudo. |
| `--uninstall` | Not installed | Stop and remove the service. |
| `--agent` | Off | With --install: install a LaunchAgent instead. Starts at login. No sudo. |

Examples:

```bash
macmon serve                     # 0.0.0.0:9090, interval 1000 ms
macmon serve -p 8080 -i 500      # custom port and interval
macmon serve --host 127.0.0.1    # local only
curl http://127.0.0.1:9090/json
curl http://127.0.0.1:9090/metrics
```

Service examples:

```bash
sudo macmon serve --install       # LaunchDaemon, port 9090
sudo macmon serve -p 8080 --install
sudo macmon serve --uninstall     # remove the daemon
macmon serve --install --agent    # LaunchAgent, no sudo
macmon serve --uninstall          # remove the agent
```

### Prometheus metric names

All metrics are gauges. Names from macmon 0.12.0 output:

| Metric | Meaning |
| --- | --- |
| macmon_cpu_power_watts | CPU power. |
| macmon_gpu_power_watts | GPU power. |
| macmon_ane_power_watts | Neural Engine power. |
| macmon_chip_power_watts | Whole chip power. |
| macmon_sys_power_watts | Whole system power. |
| macmon_ram_power_watts | DRAM power. |
| macmon_dcs_power_watts | Memory controller power. |
| macmon_dc_in_power_watts | Adapter input power. |
| macmon_cpu_temp_celsius | Average CPU temperature. |
| macmon_gpu_temp_celsius | Average GPU temperature. |
| macmon_fan_speed_rpm | Fan speed. Label: fan. Example: fan="fan0". |
| macmon_cpu_active_ratio | CPU active ratio. |
| macmon_cpu_scaled_ratio | CPU scaled ratio. |
| macmon_cpu_tier_freq_mhz | Tier frequency. Label: tier (E, P, S). |
| macmon_cpu_tier_scaled_ratio | Tier scaled ratio. Label: tier. |
| macmon_cpu_tier_active_ratio | Tier active ratio. Label: tier. |
| macmon_memory_ram_total_bytes | Total RAM. |
| macmon_memory_ram_used_bytes | Used RAM. |
| macmon_memory_swap_total_bytes | Total swap. |
| macmon_memory_swap_used_bytes | Used swap. |
| macmon_memory_pressure_level | 0 normal, 1 warning, 2 critical. |
| macmon_memory_pressure_percent | Pressure percent. |
| macmon_gpu_memory_alloc_bytes | GPU allocated memory. |
| macmon_gpu_memory_in_use_bytes | GPU memory in use. |
| macmon_battery_percent | Battery charge. |
| macmon_battery_charging | 1 while charging, 0 otherwise. |
| macmon_battery_plugged | 1 on external power, 0 on battery. |
| macmon_net_in_bytes_per_second | Received bytes per second. |
| macmon_net_out_bytes_per_second | Sent bytes per second. |
| macmon_disk_read_bytes_per_second | Disk read throughput. |
| macmon_disk_write_bytes_per_second | Disk write throughput. |
| macmon_uptime_seconds | Time since boot. |

macmon still prints these old names: macmon_all_power_watts (use macmon_chip_power_watts), macmon_cpu_usage_ratio (use macmon_cpu_scaled_ratio), macmon_gpu_usage_ratio (use macmon_gpu_scaled_ratio), macmon_gpu_ram_power_watts, and macmon_ecpu_* / macmon_pcpu_* (use macmon_cpu_tier_*).

## Generate load with macmon stress

Use stress to test a metrics pipeline, or to see how power fields respond to load.

Modes:

| Mode | Load |
| --- | --- |
| pulse | Default. One second busy, then one second idle. |
| cpu | Continuous CPU load. |
| gpu | Continuous GPU load. |
| ane | Repeated OCR. The main compute stage runs on the Neural Engine. |
| all | Continuous CPU, GPU, and ANE load. |

Options:

| Option | Meaning |
| --- | --- |
| `-w, --workers <N>` | CPU worker threads. Ignored in gpu and ane modes. |
| `-d, --duration <S>` | Stop after S seconds. Applies after ANE warmup in ane and all modes. Runs until stopped when omitted. |
| `--pulse [<S>]` | Pulse: S seconds busy, then S seconds idle. Default 2 when enabled. |
| `-i, --interval <MS>` | Update interval. Default 1000. |

Examples:

```bash
macmon stress cpu -d 10
macmon stress gpu -d 10
macmon stress all -d 15 --pulse 3
```

Rules:

- Always set `-d` or `--pulse` in a script. Without a limit, the run never ends.
- In ane and all modes, macmon warms the Neural Engine up before the timed run.

## Troubleshoot

- Run `macmon debug` to see what the tool detects: chip, topology, memory, sensors, and raw sensor tables.
- Run `macmon find-clpc` to locate CLPC power counters under CPU, GPU, and ANE load. Add `--powermetrics` to compare the result with Apple's powermetrics. The check needs sudo only for powermetrics.
- On macOS 27, Apple broke the CPU and ANE power source. macmon reads hidden energy reports from the AppleCLPC driver through IOReport. It uses opaque counter IDs and undocumented mappings.
- In a macOS VM, power and frequency read 0, temperatures are null, and fans are empty. The soc object has vm: true.

## Worked patterns

### One-shot measurement

```bash
macmon pipe -s 1 -i 1000 | jq
```

Read cpu_power, gpu_power, chip_power, temp, memory, and fans from the single object.

### Compare idle with load

1. Measure idle: `macmon pipe -s 5 -i 1000 > idle.json`
2. Start load in the background: `macmon stress all -d 12`
3. Measure under load: `macmon pipe -s 10 -i 1000 > load.json`
4. Compare the fields with jq.

### Stream to Prometheus

1. Start `macmon serve --host 127.0.0.1 -p 9090` in the background.
2. Scrape http://127.0.0.1:9090/metrics.

## Reference

- Repo: https://github.com/vladkens/macmon
- Help: `macmon --help` and `macmon <command> --help`
- Local version used to write this skill: 0.12.0.