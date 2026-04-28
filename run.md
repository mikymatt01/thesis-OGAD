### Delete images
- docker compose down -v --rmi all --remove-orphans

### How to execute
- tmux
- docker compose up
- control + B and D

### Delete tmux session
- tmux ls
- tmux kill-session -t <id>

### Delete data dir
docker run --rm -v $(pwd):/app alpine rm -rf /app/data