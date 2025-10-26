cd docker
docker compose up -d --build
for /f "skip=1" %%i in ('docker compose ps -q rlstack') do docker exec -it %%i bash
