# Instrumentation

The metric layer every later phase reads. Implements three measurements as
low-overhead, real-time counters and shows they predict GPU degradation.

- **WAR** (Warp Alignment Ratio) -- `adapter_slots/metrics/war.py`
- **WARτ** (time-windowed WAR) and **global WAR** -- `adapter_slots/metrics/gwar.py`
- **H_align** (alignment entropy) -- `adapter_slots/metrics/entropy.py`

Collection and export: `adapter_slots/instrumentation/{batch_logger,sgmv_tracker,
prometheus_exporter}.py`. Correlation between the metrics and measured SGMV
intensity / throughput is validated by `analysis/compute_correlations.py`
(Theorems 4.4, 7.3, 7.4, 11.2).
