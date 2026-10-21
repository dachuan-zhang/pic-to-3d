# vendored dependencies

`three.module.js` — three.js r160, MIT licensed, unmodified, taken from
`https://unpkg.com/three@0.160.0/build/three.module.js`.

Vendored rather than loaded from a CDN so the viewer works offline, with no
network dependency and no import map (some browsers / content blockers break
either one). It is a single self-contained bundle — no other files needed.

Upgrade with:

```bash
curl -o viewer/vendor/three.module.js \
  https://unpkg.com/three@<version>/build/three.module.js
```
