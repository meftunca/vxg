# `@vxg/wasm`

Zero-WASI WebAssembly wrapper for **VXG** — adaptive block-independent lossless codec & compressed-domain SQL engine (`SELECT`, `DISTINCT`, `COUNT(DISTINCT col)`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, `OFFSET`).

## Quick Start

```js
import { VxgCodec } from '@vxg/wasm';

const vxg = await VxgCodec.init();

const frame = vxg.compress(ndjsonText);
const result = vxg.sql(
  frame,
  'SELECT service, COUNT(*) AS cnt, COUNT(DISTINCT user) AS users FROM logs GROUP BY service HAVING cnt >= 10 ORDER BY cnt DESC LIMIT 5',
  { format: 'json' }
);
console.log(result);
```
