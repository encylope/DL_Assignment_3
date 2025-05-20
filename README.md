# DL_Assignment_3
Link to the report : https://api.wandb.ai/links/na21b075-indian-institute-of-technology-madras/qqmn8abm

Link to the Github repo: https://github.com/encylope/DL_Assignment_3

This repository contains the implementation and experiments for DL Assignment 3, where a sequence-to-sequence (Seq2Seq) model is trained to perform transliteration from Hindi script to Latin script using various RNN-based architectures.
## Task Overview

The task is to transliterate words from Roman script (transliterated Hindi) to Devanagari script using a character-level encoder-decoder architecture. Key techniques explored include:

Vanilla RNN, GRU, and LSTM architectures

Use of attention mechanisms

Beam search decoding

WandB for hyperparameter tuning

Visualization of model outputs 

## It contains

A3.ipynb -	Main Jupyter Notebook for training and evaluation

pred_attention.csv	- Model predictions with attention

pred_vanilla.csv	- Model predictions without attention

README.md	- This documentation file

Run the notebook cell by cell.

To run model on test data the followin code executed

         model = Test_Model(lang="hi",embed_dim=256,enc_layers=3,dec_layers=3,type_layer="lstm",units=256,dropout=0.2,attention=True)     
To run the model with WandB sweep, use the following code:

# Creating the WandB config

     sweep_config = {
           "name": "Sweep_Assignment3",
           "method": "random",
           "parameters": {
                 "decorder_encoder_layers": {
                    "values": [1, 2, 3]
                 },
                 "units": {
                     "values": [64, 128, 256]
                 },
                 "type_of_Layer": {
                     "values": ["gru", "rnn","lstm"]
                 },
                  "embeded_dim": {
                     "values": [256,64, 128]
                 },
                 "dropout": {
                     "values": [0.29, 0.37]
                 },
                 "Beam_width": {
                     "values": [3, 7, 5]
                 },
                 "Teacher_forcing_ratio": {
                     "values": [0.9, 0.5,0.2]
                 },
                 "Attention": {
                     "values": [True,False]
                 },
                 "epochs":{
                     "values":[10,20,30]
                 }
             }
         }

To visualize the model outputs, use the following code:

    visualize_model_outputs(model, n=15)

To visualise the model connectivity, use the following code:
# Sample some words from the test data

    test_words = get_test_words(5)

# Visualise connectivity for "test_words"

for word in test_words:

    visualise_connectivity(model, word, activation="scaler")



