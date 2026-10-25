# Deploying to Render (free tier)

The `Dockerfile` in this repo is ready for Render. It installs the free
providers, bakes in the Piper voice model, and listens on the port Render gives it.

## 1. Push the project to GitHub

From `D:\incubrix-broll`:

```bash
git init
git add .
git commit -m "IncuBrix B-roll engine"
# create an EMPTY repo on github.com first, then:
git remote add origin https://github.com/<your-username>/incubrix-broll.git
git branch -M main
git push -u origin main
```

Check that `.env` is NOT pushed (it is in `.gitignore`) - your Pixabay key must
never be committed.

## 2. Create the Render service

1. Go to https://dashboard.render.com/ and sign in with GitHub.
2. **New +** -> **Web Service** -> connect your `incubrix-broll` repo.
3. Runtime: **Docker** (Render detects the Dockerfile). Instance type: **Free**.
4. Add environment variables (Environment tab):

   | Key | Value |
   | --- | --- |
   | `PIXABAY_API_KEY` | your Pixabay key |
   | `PUBLIC_BASE_URL` | `https://<your-service>.onrender.com` (fill in after the first deploy) |
   | `MOCK_MODE` | `false` |
   | `FOOTAGE_SOURCES` | `pixabay` |
   | `WHISPER_MODEL` | `tiny` |

5. Click **Create Web Service**. First build takes ~5-10 minutes.

## 3. Test it

Once it says "Live", open:

- `https://<your-service>.onrender.com/health` -> `{"status":"ok"}`
- `https://<your-service>.onrender.com/ready` -> check `mock_mode: false`,
  `footage_sources: ["pixabay"]`, `pixabay_key_set: true`

Then submit a job (from any terminal):

```bash
curl -s -X POST https://<your-service>.onrender.com/api/v1/broll/jobs \
  -H "Content-Type: application/json" -H "Idempotency-Key: demo-1" \
  -d @testdata/request_script.json
```

Poll the returned job id, then open the `video_16x9` link it gives you.

## Things to report honestly in the Submission Form

- **Cold start.** Render's free tier sleeps after ~15 minutes idle; the next
  request takes 30-60 s to wake, plus first-run model loading. Put this number
  in the form.
- **Ephemeral disk.** Render's free tier has no persistent disk, so rendered
  files live only until the next restart/redeploy. Output links can break after a
  restart. (Integration Standard point 7 wants S3-compatible storage; note that
  this demo uses local storage.)
- **Memory.** Free instances have 512 MB. `WHISPER_MODEL=tiny` keeps it inside
  that. If the service crashes on a job, that is the likely cause - report it.
