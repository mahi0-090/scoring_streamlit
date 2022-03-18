mkdir -p ~/.streamlit/

echo "[theme]
primaryColor=\"#f21111\"
backgroundColor=\"#0e1117\"
secondaryBackgroundColor=\"#31333F\"
textColor=\"#fafafa\"
font=\"sans serif\"
\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
