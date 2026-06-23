# Assets

| File | Purpose |
|------|---------|
| `demo.mp4` | Product demo (also attach to GitHub Release for inline README player) |
| `mascot.png` | Repo mascot / video poster |

## README video embed

GitHub **does not** play `<video src="assets/demo.mp4">` from a relative path — it renders as a broken empty block.

For an inline player in the root README, attach `demo.mp4` to the **`v0.1.0` release** on GitHub:

1. Repo → **Releases** → open `v0.1.0` (or create it from the `v0.1.0` tag)
2. **Edit release** → drag `demo.mp4` into assets → save

The README uses:

`https://github.com/SahilSheikh12299/codegraph-mcp/releases/download/v0.1.0/demo.mp4`

The **▶ Watch or download** link uses `raw.githubusercontent.com` and works even before the release asset is attached.
