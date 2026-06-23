# Assets

| File | Purpose |
|------|---------|
| `demo.mp4` | Product demo video |
| `demo-poster.jpg` | README thumbnail (frame from demo; click opens video) |
| `mascot.png` | Repo mascot |

Regenerate poster after re-recording demo (macOS):

```bash
qlmanage -t -s 1200 -o . demo.mp4 && mv -f demo.mp4.png demo-poster.jpg
```
