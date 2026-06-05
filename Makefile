.PHONY: build up down logs clean shell-backend shell-frontend restart

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	-docker compose down -v 2>/dev/null
	-docker container stop $$(docker container ls -aq) 2>/dev/null
	-docker container rm $$(docker container ls -aq) 2>/dev/null
	docker system prune -a --volumes -f

shell-backend:
	docker compose exec backend /bin/bash

shell-frontend:
	docker compose exec frontend /bin/sh

restart: clean build up