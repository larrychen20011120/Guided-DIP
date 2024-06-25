# Guided-DIP
Guiding DIP Early Stopping with DDPM-inspired Supervision

## How to start the code
### Install Dependencies
* running on your local device
```
pip install -r requirements.txt
```
* running in colab
```
pip install python-dotenv
```
### create the .env file
First, you should copy the `.env.example` to `.env`
```
cp .env.example .env
```
### Download dataset if you need it
```
I recommend you run the code with the images in data/test directory without installing any files first.
```
* replace the your kaggle token to the following two lines in the `.env` file
```
KAGGLE_USERNAME="your user name in kaggle"
KAGGLE_KEY="your api key in kaggle"
```
* Then, run the following shell script. It will download the image from kaggle dataset
```bash
./download_data.sh
```
### Notebook details
* test.ipynb is the demonstration of simple DDPM diffusion forward process and DIP training process on the clean image
* main.ipynb is the notebook for running the Guided-DIP and DIP on noisy image

## How to run with different experiments
* all the settings of dip, ddpm and guided-dip can be altered by rewriting the `config.yaml`, you should follow the `models.model.py` configuration parsing for details.
* Here, I show some changes you can try to rewrite 



## Acknowledgements 
* [PyTorch Deep Image Prior](https://github.com/safwankdb/Deep-Image-Prior)
* [DDPM from scratch in Pytorch](https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch)
* [SegNet Pytorch](https://github.com/vinceecws/SegNet_PyTorch/tree/master)
* [chatGPT](https://chatgpt.com/)