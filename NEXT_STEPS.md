# NEXT_STEPS — from skeleton to submission

The skeleton runs end to end in mock mode. To turn it into a real submission,
work through this list in order and **update the Submission Form honestly** as
you go (Yes / Partial / No + a link to evidence for each claim).

## 1. Enable the real providers (biggest single win)

```bash
cp .env.example .env
# then set in .env:
MOCK_MODE=false
STT_BACKEND=openai
TTS_BACKEND=openai
FOOTAGE_SOURCES=pexels
OPENAI_API_KEY=...      # never commit this
PEXELS_API_KEY=...
```

Then run a job with `testdata/request_script.json` and check the video actually
shows real footage that matches the narration. If relevance looks weak, tune the
query string in `pipeline.select_clips` (add the `style` and `visual_hint`).

## 2. Add a second footage source (Advanced level)

Add a `PixabayFootageSource` (same interface as `PexelsFootageSource`) and set
`FOOTAGE_SOURCES=pexels,pixabay`. Write a one-line rule for choosing between
them and store the reason per scene in `timeline.json` (the pipeline already has
a `reason` field).

## 3. Deploy the live demo URL

- Build the image: `docker build -t broll .`
- Run it on any small host (Fly.io / Render / a VM) with the real keys in env.
- Put a **rate limit** in front and note the **cold-start time**.
- Keep it running for the whole review period.

## 4. Collect the evidence the brief asks for

- **Test pack**: inputs, outputs (including failures), logs, and an index
  linking each claim in `REPORT.md` to a file. Put them in the repo.
- **Demo video** (≤ 8 min, unedited): one clean run, one failure and how the
  system handles it, the tests running, and one design decision explained in
  code.
- **Video pack**: every output video with its `timeline.json`, captions and
  `manifest.json` — including weak and failed results.

## 5. Fill the metrics honestly

In `REPORT.md` (and the Submission Form's Metrics tab), report:
turnaround (cold + warm), cost per minute of video, coverage, your own 1–5
relevance per scene, reliability (success rate, repeat-clip rate — must be 0),
and peak memory / GPU time if used. Where you did not do something, write "not
done" — an honest gap is fine; a false claim disqualifies the whole submission.

## 6. Before you submit

- [ ] `./run_tests.sh` passes
- [ ] `docker compose up` starts cleanly on a fresh machine
- [ ] every claim in the form has a working evidence link
- [ ] `SOURCES.md` and `AI_USE.md` are accurate and complete
- [ ] you can explain and modify every file (the live session will test this)
- [ ] reply to the invitation email with the completed Submission Form attached
