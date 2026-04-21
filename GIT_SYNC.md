# Git Sync

Current shared branch: `pi05-piper-sync`

Remote layout:

- `origin`: upstream repo `MINT-SJTU/Evo-RL`
- `z520yu`: your GitHub fork `z520yu/Evo-RL`

Tracked in this branch:

- `README_PI05_PIPER.md`
- `.gitignore`

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
