# Reproduce the MVP checks

```bash
python -m pytest -q tests
bash -n scripts/review_assistant/install.sh scripts/review_assistant/run.sh
docker compose -f docker/compose.review-assistant.yaml config
```

For a local UI smoke test:

```bash
REVIEW_ASSISTANT_DATABASE=/tmp/review-assistant.sqlite \
  python -m streamlit run \
  src/review_assistant/web_app.py --server.headless=true
```

No model or dataset is downloaded by these commands.
