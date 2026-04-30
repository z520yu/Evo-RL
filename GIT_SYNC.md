# Git Sync

Current shared branch: `pi05-piper-sync`

Remote layout:

- `origin`: upstream repo `MINT-SJTU/Evo-RL`
- `z520yu`: your GitHub fork `z520yu/Evo-RL`

Tracked in this branch:

- `lenovo_README_PI05_PIPER.md`
- `gemini_README_PI05_PIPER_CLIENT.md`
- `lenovo_README_PI05_VALUE_ACP.md`
- `gemini_PIPER_MULTITASK_DATA_COLLECTION.md`
- `.gitignore`

Machine-specific docs use a filename prefix:

- `lenovo_*`: this machine
- `gemini_*`: the other machine

Local only, do not commit:

- `piper_multitask_v1/`
- `outputs/`
- `logs/`

## Other computer: first pull

```bash
cd /path/to/Evo-RL
git remote add z520yu https://github.com/z520yu/Evo-RL.git
git fetch z520yu
git switch -c pi05-piper-sync --track z520yu/pi05-piper-sync
```

## Other computer: later updates

```bash
cd /path/to/Evo-RL
git switch pi05-piper-sync
git pull --rebase z520yu pi05-piper-sync
```

## After editing on either computer

```bash
git add <files>
git commit -m "Describe your change"
git push
```
