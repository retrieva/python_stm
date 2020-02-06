# Structured topic model

論文: A Model of Text for Experimentation in the Social Sciences

```
python stm_main.py -f <filename> -k <number of topic> -i <iteration count>
```

```
optional arguments:
  -h, --help       show this help message and exit
  -f FILENAME      set corpus filepath.fileformat is csv
  -d DOCUMENT      set document field name
  -c CORPUS        using range of Brown corpus' files(start:end)
  --alpha ALPHA    parameter alpha for LDA(default=1.0)
  --beta BETA      parameter beta for LDA(default=0.1)
  -k TOPICS        number of topics(default=20)
  -i ITERATION     iteration count(default=100)
  -x X             set prevalences column name
  -y Y             set covariates column name
  --parser PARSER  select parser eng_nltk or mecab(default=mecab)
  --sigma SIGMA    initial value of sigma diagonals(default=0.1)
  --stopwords      exclude stop words by using corpus from nltk
  --seed SEED      random seed
  --df DF          threshold of document freaquency to cut words
  --interact       consider interaction between covariates adn topics
  --sinit          smart initialize of parameters for LDA
 ```

注意
- `-y` をつけた場合、計算速度が落ちます。また、perplexityがあまり下がらない傾向があります。