# Bizz01 Aegis — Mech Royale Bot

A **Python 3.13, standard-library-only** agent for the [Builders Mech Royale Buildathon](https://joinbuilders.org/buildathon). The `Dockerfile` is already at the repository root and starts the bot over the `mechfight/1` JSON-lines protocol.

## What the bot does

- **Prioritizes placement**: enemies are a threat; avoid being boxed in by the coordinated practice house bots.
- **Reads the real OSU arena** from the mounted `/arena/v2/world.json` and `terrain.i16`; builds a collision-aware navigation grid at startup, then routes around buildings with bounded A*.
- **Predicts the shrinking zone** and travels inside the *next* circle before it becomes dangerous.
- **Escapes, dodges and repairs**: distances itself from the nearest hunters, avoids oncoming projectiles, boosts in emergencies, jumps/jets when stuck, and values regenerative upgrades.
- **Collects safe loot**: takes a second weapon (prefers laser), opportunistically picks up cubes, and contests supply drops only when uncontested.
- **Fights selectively**: leads moving targets with gun shots; solves a ballistic firing angle for mortar; handles charged laser release and lightning range; can farm isolated NPCs and crates.
- **Operates within tight limits**: no package installation, no network, no disk writes; one deterministic action per observation.

## Files

- `example_bot.py`: the complete agent (the filename matches the starter Dockerfile convention).
- `mechfight.py`: unmodified official Python transport wrapper from the uploaded kit.
- `Dockerfile`: competition container definition.
- `test_windows.ps1`: optional PowerShell helper that unpacks the official kit and runs checks.
- `README.md`: these instructions.

## Run on Windows 11 (PowerShell)

1. Install and start **Docker Desktop** and ensure `python` is available on your PATH.
2. Extract `Bizz01_MechRoyale.zip`. Keep the downloaded `mechfight-kit.tar.gz` alongside the extracted folder, or in `Downloads`.
3. Open PowerShell **inside the bot directory** and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1
```

The script runs the official `check/check.py` both natively and inside Docker with the same no-network, 1-CPU, 512 MB, read-only, unprivileged restrictions as scoring. To specify a different kit path:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_windows.ps1 -KitArchive "C:\Users\yourname\Downloads\mechfight-kit.tar.gz"
```

## Push to GitHub and enter

Open PowerShell in **this folder**, create a new public GitHub repository, and substitute its actual URL:

```powershell
git init
git add .
git commit -m "Bizz01 Aegis Mech Royale bot"
git branch -M main
git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
git push -u origin main
```

Submit the **public repository URL** on the Builders Mech Royale track page. Leave the optional commit hash blank to submit the latest default-branch commit. Keep the repository public through the competition finals. The tournament accepts **4 submissions per hour**, and the code at the submitted commit must be **under 20 MB**.

## Test status / limitations

The official 300-observation local conformance check passed with **300 responses within 50 ms** and no malformed protocol messages. The check replays *recorded* observations, so it **does not test game strength**. Docker is not available in the authoring environment, so the match-sandbox check must be run on your computer. Competitive strength must be measured with a scored submission. No leaderboard performance is claimed.

The bot deliberately uses heuristics rather than a trainable model: milliseconds of compute time, deterministic behavior, and robust survival were prioritized over a slower model. The routing grid ignores roof access, and projectile aiming is approximate, so there is room to improve after seeing practice results.
