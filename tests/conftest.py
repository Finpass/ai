import os

# Tests must never report to the real Sentry or Phoenix, whatever the developer's .env says.
# Environment variables beat .env, and this runs before any `src` module reads its settings.
os.environ["SENTRY_DSN"] = ""
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""
