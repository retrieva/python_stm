# Structured topic model

論文: [A Model of Text for Experimentation in the Social Sciences](https://scholar.princeton.edu/sites/default/files/bstewart/files/a_model_of_text_for_experimentation_in_the_social_sciences.pdf)

```
python stm_main.py -f <filename> -k <number of topic> -i <iteration count>
```

```
optional arguments:
  -h, --help       show this help message and exit
  -f FILENAME      Set corpus filepath. Fileformat is csv
  -d DOCUMENT      Set document field name
  -c CORPUS        Using range of Brown corpus' files(start:end)
  --alpha ALPHA    Parameter alpha for LDA(default=1.0)
  --beta BETA      Parameter beta for LDA(default=0.1)
  -k TOPICS        Number of topics(default=20)
  -i ITERATION     Iteration count(default=100)
  -x X             Set prevalences column name
  -y Y             Set covariates column name
  --parser PARSER  Select parser eng_nltk or mecab(default=mecab)
  --sigma SIGMA    Initial value of sigma diagonals(default=0.1)
  --stopwords      Exclude stop words by using corpus from nltk
  --seed SEED      Random seed
  --df DF          Threshold of document freaquency to cut words
  --interact       Consider interaction between covariates adn topics
  --sinit          Smart initialize of parameters for LDA
 ```

注意
- `-y` をつけた場合、計算速度が落ちます。また、perplexityがあまり下がらない傾向があります。