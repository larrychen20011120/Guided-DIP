# Guided-DIP
Guiding DIP Early Stopping with DDPM-inspired Supervision

## How to start the code
### Install Dependencies
```
pip install -r requirements.txt
```
### Download dataset
* First, you should copy the `.env.example` to `.env`
```
cp .env.example .env
```
* Next, replace the your kaggle token to the following two lines
```
KAGGLE_USERNAME="your user name in kaggle"
KAGGLE_KEY="your api key in kaggle"
```
* Then, run the following shell script. It will download the image from kaggle dataset
```bash
./download_data.sh
```


## Acknowledgements 
* [PyTorch Deep Image Prior] (https://github.com/safwankdb/Deep-Image-Prior)
* [DDPM from scratch in Pytorch] (https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch)
* [chatGPT] (https://chatgpt.com/)