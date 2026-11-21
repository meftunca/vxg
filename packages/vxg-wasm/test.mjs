import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { VxgCodec } from './index.mjs';

const candidates = [
  process.env.VXG_WASM_PATH,
  join(process.cwd(), 'vxg.wasm'),
  join(process.cwd(), '../../target/wasm32-unknown-unknown/release/vxg.wasm'),
  join(homedir(), '.cargo/target/wasm32-unknown-unknown/release/vxg.wasm'),
];
const wasmPath = candidates.find((p) => p && existsSync(p));
assert.ok(wasmPath, 'vxg.wasm not found; build with cargo build --release --lib --target wasm32-unknown-unknown');

const codec = await VxgCodec.init(await fs.readFile(wasmPath));

const services = ['auth', 'billing', 'edge', 'search'];
const sample = Array.from({ length: 200 }, (_, i) =>
  JSON.stringify({
    service: services[i % services.length],
    user: `u${i % 13}`,
    status: i % 10 === 0 ? 500 : 200,
    latency: 20 + (i % 40) * 5,
  })
).join('\n') + '\n';

const frame = codec.compress(sample);
const restored = new TextDecoder().decode(codec.decompress(frame));
assert.equal(restored, sample);

const distinctOut = codec.sql(frame, 'SELECT DISTINCT service FROM t ORDER BY service ASC LIMIT 2 OFFSET 1');
assert.equal(distinctOut, 'service\nbilling\nedge\n');

const havingOut = codec.sql(
  frame,
  'SELECT service, COUNT(*) AS cnt, COUNT(DISTINCT user) AS uniq_u FROM t WHERE status = 200 GROUP BY service HAVING cnt >= 40 AND uniq_u >= 5 ORDER BY service ASC'
);
assert.ok(havingOut.startsWith('service\tcnt\tuniq_u\n'));
assert.equal(havingOut.trim().split('\n').length, 5);

console.log(`@vxg/wasm Node test OK (${sample.length}B -> ${frame.length}B, SQL DISTINCT+HAVING+OFFSET verified)`);
