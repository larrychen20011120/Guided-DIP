# create data/ directory
mkdir -p data

# export kaggle personal tokens
export KAGGLE_KEY=$(grep '^KAGGLE_KEY=' .env | cut -d '=' -f2-)
export KAGGLE_USERNAME=$(grep '^KAGGLE_USERNAME=' .env | cut -d '=' -f2-)

kaggle datasets download phucthaiv02/butterfly-image-classification/ -p data/ --unzip
sleep 2