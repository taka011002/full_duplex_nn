build:
	docker-compose build --no-cache
up:
	docker-compose up -d

down:
	docker-compose down

restart:
	make down
	make up

bash:
	docker-compose exec app /bin/bash