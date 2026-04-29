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
docker run --rm -v $(pwd):/app alpine rm -rf /app/data/predictions/graphons 
docker run --rm -v $(pwd):/app alpine rm /app/data/predictions/graphs/digress/generated_samples1.txt 
docker run --rm -v $(pwd):/app alpine rm /app/data/predictions/graphs/digress/generated_samples2.txt 