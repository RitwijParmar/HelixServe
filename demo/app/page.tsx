"use client";

import { useEffect, useMemo, useState } from "react";

type Tick = { ttft: number; queue: number; limit: number; hit: boolean };
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

export default function Home() {
  const [running, setRunning] = useState(true);
  const [rate, setRate] = useState(18);
  const [prefix, setPrefix] = useState(72);
  const [target, setTarget] = useState(450);
  const [mode, setMode] = useState<"adaptive" | "static">("adaptive");
  const [ticks, setTicks] = useState<Tick[]>([{ ttft: 322, queue: 3, limit: 8, hit: true }]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setTicks((current) => {
      const last = current[current.length - 1];
      const pressure = Math.max(0, rate - last.limit * 1.75);
      const ttft = Math.round(clamp(250 + pressure * 42 - prefix * 1.9 + Math.sin(current.length * .73) * 38, 75, 1800));
      const queue = Math.round(clamp(pressure * 1.3 + Math.cos(current.length) * 2, 0, 48));
      let limit = mode === "static" ? 12 : last.limit;
      if (mode === "adaptive") {
        if (ttft > target * 1.1) limit -= 1;
        if (ttft < target * .75 && queue > 0) limit += 1;
        limit = clamp(limit, 2, 32);
      }
      return [...current.slice(-23), { ttft, queue, limit, hit: (current.length * 37) % 100 < prefix }];
    }), 650);
    return () => window.clearInterval(timer);
  }, [mode, prefix, rate, running, target]);

  const latest = ticks[ticks.length - 1];
  const p95 = useMemo(() => {
    const values = ticks.map((tick) => tick.ttft).sort((a, b) => a - b);
    return values[Math.max(0, Math.ceil(values.length * .95) - 1)];
  }, [ticks]);
  const hits = Math.round(ticks.filter((tick) => tick.hit).length / ticks.length * 100);

  return <main>
    <nav><a className="brand" href="#top"><span className="brandMark">TY</span>TICKYANTRA</a><div className="navLinks"><a href="#lab">LIVE LAB</a><a href="#system">SYSTEM</a><a href="https://github.com/RitwijParmar/TickYantra">GITHUB ↗</a></div></nav>
    <section className="hero" id="top"><div className="eyebrow"><span /> SGLANG CONTROL PLANE · V2.0</div><h1>TAIL LATENCY,<br/><em>UNDER CONTROL.</em></h1><p className="lede">A robotic admission controller for real GPU inference. It senses TTFT pressure, protects queue deadlines, and routes repeated prefixes with mechanical precision.</p><div className="heroActions"><a className="primary" href="#lab">ENTER CONTROL LAB</a><a className="secondary" href="https://github.com/RitwijParmar/TickYantra">INSPECT SOURCE</a></div><div className="rail"><span>01</span><i/><span>SENSE</span><i/><span>ADMIT</span><i/><span>EXECUTE</span></div></section>
    <section className="lab" id="lab">
      <div className="sectionHead"><div><span className="kicker">INTERACTIVE CONTROL-PLANE SIMULATOR</span><h2>PRESSURE CHAMBER</h2></div><button onClick={() => setRunning(!running)}>{running ? "Ⅱ PAUSE" : "▶ RUN"}</button></div>
      <p className="disclosure">Behavioral visualization—not a GPU benchmark. Published performance numbers require committed SGLang/L4 artifacts.</p>
      <div className="labGrid"><aside className="controls">
        <div className="controlLabel">CONTROL MODE<div className="segmented"><button className={mode === "adaptive" ? "active" : ""} onClick={() => setMode("adaptive")}>ADAPTIVE</button><button className={mode === "static" ? "active" : ""} onClick={() => setMode("static")}>STATIC</button></div></div>
        <label>ARRIVAL RATE <output>{rate} req/s</output><input aria-label="Arrival rate" type="range" min="4" max="40" value={rate} onChange={(e) => setRate(Number(e.target.value))}/></label>
        <label>SHARED PREFIX <output>{prefix}%</output><input aria-label="Shared prefix percentage" type="range" min="0" max="100" value={prefix} onChange={(e) => setPrefix(Number(e.target.value))}/></label>
        <label>TTFT TARGET <output>{target} ms</output><input aria-label="TTFT target" type="range" min="150" max="900" step="25" value={target} onChange={(e) => setTarget(Number(e.target.value))}/></label>
        <div className="rule"><span>CONTROL LAW</span><code>p95 &gt; 1.10 × target → limit − 1<br/>p95 &lt; 0.75 × target → limit + 1</code></div>
      </aside><div className="telemetry">
        <div className="metrics"><article><span>P95 TTFT</span><strong className={p95 > target ? "warning" : ""}>{p95}<small>ms</small></strong><i>{p95 <= target ? "SLO HELD" : "PRESSURE"}</i></article><article><span>ACTIVE LIMIT</span><strong>{latest.limit}<small>req</small></strong><i>{mode.toUpperCase()}</i></article><article><span>QUEUE DEPTH</span><strong>{latest.queue}<small>req</small></strong><i>{latest.queue < 10 ? "NOMINAL" : "CONGESTED"}</i></article><article><span>PREFIX HITS</span><strong>{hits}<small>%</small></strong><i>RADIX AFFINITY</i></article></div>
        <div className="chart" role="img" aria-label="Recent TTFT measurements"><div className="targetLine" style={{bottom:`${clamp(target / 18, 8, 92)}%`}}><span>SLO {target}ms</span></div>{ticks.map((tick,index) => <b key={index} className={tick.ttft > target ? "hot" : ""} style={{height:`${clamp(tick.ttft / 18,3,100)}%`}}/>)}</div>
        <div className="flow"><div><span>01</span><strong>INGRESS</strong><small>{rate} req/s</small></div><i>→</i><div><span>02</span><strong>YANTRA GATE</strong><small>limit {latest.limit}</small></div><i>→</i><div><span>03</span><strong>SGLANG</strong><small>RADIX + CUDA</small></div><i>→</i><div><span>04</span><strong>L4 GPU</strong><small>REAL TOKENS</small></div></div>
      </div></div>
    </section>
    <section className="system" id="system"><div><span className="kicker">ENGINEERING POSITION</span><h2>CONTROL THE QUEUE.<br/>TRUST THE ENGINE.</h2></div><div className="principles"><article><b>01</b><h3>NO TOY PATH</h3><p>SGLang owns tokenization, paged KV, continuous batching, RadixAttention, and CUDA execution.</p></article><article><b>02</b><h3>FAIR AFFINITY</h3><p>Hot prefixes move first until deadline pressure forces age-based fairness.</p></article><article><b>03</b><h3>RAW EVIDENCE</h3><p>Every request retains status, TTFT, ITL samples, E2E, and token counts.</p></article></div></section>
    <footer><span>TICKYANTRA / LOW-LATENCY INFERENCE SYSTEMS</span><span>BUILT BY RITWIJ PARMAR</span><a href="https://github.com/RitwijParmar/TickYantra">SOURCE ↗</a></footer>
  </main>;
}
