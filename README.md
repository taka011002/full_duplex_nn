# Nonlinear Interference Suppression Using Neural Network for Full-Duplex System

# Verified environment
- Python : v3.8.6

Specification of the Anaconda environment is recommended.

# Required libraries
- Tensorflow: v2.2.0
- keras-rectified-adam
- slack_sdk
- dataclasses-json
- tqdm

Command to install the library
```shell
conda install -y tensorflow \
&& pip install keras-rectified-adam slack_sdk dataclasses-json tqdm
```
# Environment variables
Look at the .env.example  
It is recommended to use dotenv to load the .env file.  
In order to work properly, the base path of this program must be PYTHONPATH.

# For local
```shell
cd simulations # move to simulations directory
python snr_ber_average_ibo.py
```
If you want to run it in the background.  
You need to install jq.
```shell
brew install jq
```
run command
```shell
cd simulations
./sh/snr_ber_average_ibo.sh
# or
./sh/base.sh snr_ber_average_ibo.sh
```

# For Docker
# build
```shell
make
```
# run
If you want to run snr_ber_average_ibo.py
```shell
docker run -w /app/simulations --rm -it -d --env-file .env.docker-compose -v $(pwd):/app full_duplex_nn_app python snr_ber_average_ibo.py
```