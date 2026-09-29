# Latency Benchmark Report (Paddle-OCR)

**15 images × 1 run** · 20 CPUs · Python 3.13.14

**Models:**

| Role | Model |
|---|---|
| Text line detector (`det_model`) | `PP-OCRv5_server_det` |
| Arabic + English + digits (`ar_rec_model`) | `arabic_PP-OCRv5_mobile_rec` |
| English / digits (`en_rec_model`) | `PP-OCRv6_medium_rec` *(None = Arabic only)* |

---

## Startup

| Metric | Value |
|---|---|
| Model load | 5.53 s |
| First request (cold) | **9.84 s** — Screenshot 2026-09-24 151226.png, OCR 9,793 ms |
| Memory before load | 92 MB |
| Memory after load | 2,470 MB |
| Memory peak | 20,365 MB |

---

## Sequential — one request at a time (15 requests, service time, no queueing)

### Latency breakdown (ms)

| Stage | min | mean | p50 | p90 | p95 | p99 | max | stdev | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **total** | 5,894 | 18,463 | 24,164 | 24,750 | 25,024 | 25,185 | 25,225 | 7,781 | 100.0% |
| db_insert | 2.9 | 4.7 | 3.4 | 9.8 | 10.1 | 10.1 | 10.2 | 2.7 | 0.0% |
| ocr | 5,857 | 18,431 | 24,124 | 24,720 | 24,994 | 25,153 | 25,193 | 7,782 | 99.8% |
| extract | 7.0 | 20.2 | 22.6 | 24.8 | 25.5 | 25.8 | 25.9 | 5.3 | 0.1% |
| db_update | 3.1 | 3.9 | 3.5 | 4.0 | 5.9 | 9.3 | 10.2 | 1.8 | 0.0% |
| other | 1.2 | 2.9 | 1.5 | 6.9 | 8.0 | 9.9 | 10.3 | 2.8 | 0.0% |

### Summary

| Metric | Value |
|---|---|
| Avg latency | **18.46 s** |
| p95 | 25.02 s |
| p99 | 25.18 s |
| Throughput | 0.05 req/s (**195 receipts/hour**) |
| Within SLO (5 s) | **0.0%** |
| Tail ratio p95 / p50 | 1.04× |
| Cold vs warm | 0.97× (same image, first request vs later ones) |
| CPU per request | 74.68 s (4.0 of 20 cores busy on average) |
| OCR per megapixel | 19,600 ms/MP |
| Read rate | **100.0%** (requests answered with transaction details) |
| Errors | 0 · db write errors 0 |

---

## Per Image — mean of 1 run, slowest first

| Image | KB | Resolution | Lines | Fields | Total ms | ± | OCR ms | Extr ms | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| WhatsApp Image 2026-09-24 at 2.42.39 PM (3) | 59 | 913×1600 | 20 | 9 | 25,225 | – | 25,193 | 23.9 | yes |
| WhatsApp Image 2026-09-24 at 2.42.38 PM.jpeg | 60 | 924×1599 | 14 | 7 | 24,938 | – | 24,908 | 14.0 | yes |
| WhatsApp Image 2026-09-24 at 2.42.40 PM (1) | 59 | 920×1600 | 17 | 9 | 24,468 | – | 24,436 | 23.0 | yes |
| WhatsApp Image 2026-09-24 at 2.42.39 PM (1) | 57 | 881×1600 | 18 | 9 | 24,401 | – | 24,361 | 22.4 | yes |
| WhatsApp Image 2026-09-24 at 2.42.39 PM.jpeg | 60 | 884×1599 | 15 | 7 | 24,348 | – | 24,325 | 14.8 | yes |
| WhatsApp Image 2026-09-24 at 2.42.39 PM (2) | 59 | 884×1599 | 19 | 9 | 24,338 | – | 24,306 | 23.6 | yes |
| WhatsApp Image 2026-09-24 at 2.42.40 PM (2) | 57 | 881×1600 | 18 | 9 | 24,176 | – | 24,145 | 23.2 | yes |
| WhatsApp Image 2026-09-24 at 2.42.39 PM (4) | 57 | 881×1600 | 18 | 9 | 24,164 | – | 24,124 | 23.0 | yes |
| WhatsApp Image 2026-09-24 at 2.42.40 PM.jpeg | 58 | 874×1600 | 18 | 7 | 23,744 | – | 23,710 | 25.4 | yes |
| WhatsApp Image 2026-09-27 at 11.36.29 AM.jpeg | 33 | 1260×786 | 2 | 4 | 15,121 | – | 15,105 | 7.0 | yes |
| Screenshot 2026-09-24 151226.png | 104 | 583×765 | 16 | 9 | 10,105 | – | 10,066 | 25.9 | yes |
| Screenshot 2026-09-24 160605.png | 69 | 482×775 | 17 | 9 | 8,818 | – | 8,792 | 16.4 | yes |
| Screenshot 2026-09-27 104634.png | 83 | 433×773 | 17 | 9 | 8,754 | – | 8,723 | 22.6 | yes |
| Screenshot 2026-09-24 153123.png | 83 | 455×745 | 17 | 9 | 8,448 | – | 8,412 | 22.4 | yes |
| Screenshot 2026-09-24 155843.png | 52 | 457×455 | 18 | 9 | **5,894** | – | 5,857 | 15.9 | yes |

---

## Burst — all 15 images at once, workers = 4

> *What users see when requests queue.*

### Latency (ms)

| Stage | min | mean | p50 | p90 | p95 | p99 | max | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **total** | 13,929 | 78,502 | 68,818 | 131,012 | 135,784 | 139,152 | 139,994 | 41,903 |
| wait | 41.8 | 44,155 | 46,166 | 99,200 | 105,202 | 109,953 | 111,140 | 38,722 |
| ocr | 13,750 | 34,304 | 39,931 | 46,143 | 46,465 | 46,979 | 47,107 | 13,289 |

### Summary

| Metric | Value |
|---|---|
| Last reply after | **140.07 s** |
| Throughput | 0.11 req/s |
| Within SLO (5 s) | **0.0%** |
| Queueing share of p95 | **77.5%** |
| Errors | 0 · db write errors 0 |

---

## Key Takeaways

- **OCR dominates** sequential latency — **99.8%** of total time per request.
- Throughput is **~195 receipts/hour** in sequential mode, **~396/hour** under burst (0.11 req/s).
- **No request meets the 5 s SLO** in either mode.
- Burst mode trades latency for throughput: p95 total jumps from **25.0 s → 135.8 s**, with **77.5%** of p95 spent waiting in queue.
- Cold start adds **~4.3 s** over warm (9.84 s vs ~5.5 s model load baseline).
- Memory peaks at **~20 GB** after load — worth watching for capacity planning.
