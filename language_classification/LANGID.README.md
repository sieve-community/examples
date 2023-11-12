# Language Detection using LangID

LangID is a language identification tool that can be used to detect the language of a given text. The original implementation can be found [here](https://github.com/saffsd/langid.py).

It returns a payload of the following format:

```json
{
  "language_code": "en",
  "confidence": 0.9999999999999999
}
```

The language code is a two-letter code as defined by [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

Here is the list of supported language codes:

```
af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu
```
