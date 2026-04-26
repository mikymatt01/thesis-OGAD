trap "docker compose down -v --rmi all --remove-orphans" EXIT
docker compose run -d prep_data
docker compose run -d train_model_a
docker compose run -d train_model_b
docker compose run -d eval_metrics