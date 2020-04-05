# NLIDB

## Dependencies
- TF 1.4
- python 2.7
- python3 (for annotating WikiSQL)
- tqdm
- editdistance

## How to use

Since data has been preprocessed and stored in numpy files. You can either omit first two steps go straight to train/load model or rebuild dataset.
- Set PATH (optional)
  
      export WIKI_PATH={PATH to WikiSQL}
      export GLOVE_PATH={PATH to GloVe}
  Store 'glove.840B.300d.txt' from https://nlp.stanford.edu/projects/glove/ in GLOVE_PATH.
  
  Store raw dataset from https://github.com/salesforce/WikiSQL in WIKI_PATH.
  
- Prepare WikiSQL for training and evaluation (optional)

  1. Annotate WikiSQL
  
     Annotated data has been saved in data/DATA/wiki.
     
         python3 utils/annotation/annotate.py
         
  2. Prepare Glove
      
     Edit path for 'glove.840B.300d.txt', set rebuild = True.
     
         python utils/glove.py
      
  3. Build data
      
     Data has been stored in data folder.
      
         python utils/data_manager.py
      
- Train or load model 
    
   Download [pretrained model](https://drive.google.com/open?id=1nugvgpLwuc9o2uRuSU5cLM1LHu4MrrqJ) to model/ folder.
   
      python main.py --mode train --data 'wikisql'
      
      


