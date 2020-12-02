# Full deplex nn

## dockerで実行(ネイティブで実行させた方が2倍くらい早い)
### build
```shell
make
```
```shell
docker run -w /app/simulations --rm -it -d --env-file .env.docker-compose -v $(pwd):/app full_duplex_nn_app python snr_ber_average_ibo.py
```