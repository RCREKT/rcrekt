# RCREKT
Implementation of the research paper Robust Continual Relation Extraction with Knowledge Transfer

## Framwork of RCREKT


<div align="center">
<img src=figs/framework.png width=80% />
</div>

## Dependencies

Use anaconda to create python environment:

> conda create --name yourname python=3.8 \
> conda activate yourname

Install Pytorch (suggestions>=1.7) and related environmental dependencies:

> pip install -r requirements.txt

Pre-trained BERT weights:
* Download *bert-base-uncased* into the *datasets/* directory [[google drive]](https://drive.google.com/drive/folders/1BGNdXrxy6W_sWaI9DasykTj36sMOoOGK).

### Run the Code

> python run_continual.py  --dataname TACRED 


