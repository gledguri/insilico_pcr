#!/bin/zsh
# Start the ispcr app and open it in your browser.
# Uses ./env (conda environment) when it exists, otherwise the `streamlit` of the active Python environment.
cd "$(dirname "$0")"

if [[ -x ./env/bin/streamlit ]]; then STREAMLIT=./env/bin/streamlit
elif command -v streamlit >/dev/null; then STREAMLIT=streamlit
else echo "Streamlit not found. Install first: pip install -r requirements.txt (see README.md)"; exit 1; fi

# Use 8501, or the next free port if something else is using it
PORT=8501
while lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT+1)); done
URL="http://localhost:$PORT"

echo "Starting ispcr at $URL  (press Ctrl+C to stop)"
$STREAMLIT run app/streamlit_app.py --server.port $PORT --server.headless true "$@" &
APP=$!
trap "kill $APP 2>/dev/null" INT TERM EXIT

# Wait until the app answers, then open the browser
for i in {1..60}; do
  curl -s -o /dev/null "$URL" && { open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null; break; }
  sleep 0.5
done
wait $APP
