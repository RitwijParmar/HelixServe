# TickYantra narrated demo

The final video is an end-to-end engineering walkthrough, not a synthetic benchmark reel. It covers:

1. the TickYantra identity and low-latency systems positioning;
2. the boundary between the TickYantra control plane and SGLang's CUDA data plane;
3. the request lifecycle, prefix affinity, fairness deadline, and Prometheus surface;
4. the interactive pressure-chamber controls and adaptive law;
5. the exact GCP L4 benchmark table and 99.5% native prompt-cache hit rate;
6. the adverse gateway result, the two defects it exposed, and the tested corrections;
7. the verified GCP deployment, pinned image digest, loopback binding, and cost guard.

## Artifacts

- `final/tickyantra_end_to_end_demo.mp4` — narrated 1080p H.264/AAC video.
- `final/tickyantra_demo_poster.png` — video poster.
- `final/voiceover_script.txt` — complete narration transcript.
- `final/timeline.json` — scene-level timings.
- `final/video_verification.json` — codec, dimensions, duration, audio, and fast-start checks.
- `screens/` — browser-captured control-lab source frames.

Rebuild with:

```bash
python tooling/build_tickyantra_demo_video.py
python tooling/verify_tickyantra_demo_video.py
```

The video repeats the benchmark's core disclosure: the interactive control lab visualizes behavior, while performance claims come only from committed SGLang/L4 artifacts.
