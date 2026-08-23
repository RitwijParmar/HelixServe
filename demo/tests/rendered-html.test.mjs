import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the TickYantra product surface", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>TickYantra — Tail Latency Under Control<\/title>/i);
  assert.match(html, /TAIL LATENCY,/);
  assert.match(html, /UNDER CONTROL\./);
  assert.match(html, /PRESSURE CHAMBER/);
  assert.match(html, /Behavioral visualization—not a GPU benchmark/);
  assert.match(html, /CONTROL THE QUEUE\./);
  assert.match(html, /TRUST THE ENGINE\./);
  assert.match(html, /github\.com\/RitwijParmar\/TickYantra/);
});

test("keeps benchmark claims explicit and simulation controls accessible", async () => {
  const [page, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(page, /aria-label="Arrival rate"/);
  assert.match(page, /aria-label="Shared prefix percentage"/);
  assert.match(page, /aria-label="TTFT target"/);
  assert.match(page, /Published performance numbers require committed SGLang\/L4 artifacts/);
  assert.match(page, /SGLang owns tokenization, paged KV, continuous batching/);
  assert.doesNotMatch(page, /fetch\(|XMLHttpRequest|\/v1\/completions/);

  assert.match(layout, /TickYantra — Tail Latency Under Control/);
  assert.match(layout, /summary_large_image/);
  assert.match(layout, /RitwijParmar\/TickYantra\/main\/demo\/public\/og\.png/);
});
