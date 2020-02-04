# Structured topic model

論文: AModel of Text for Experimentation in the Social Sciences

```
python stm_main.py -f <filename> -k <number of topic> -i <iteration count>
```

```
  -h, --help     show this help message and exit
  -f FILENAME    corpus filename csv
  -c CORPUS      using range of Brown corpus' files(start:end)
  --alpha ALPHA  parameter alpha for LDA(default=1.0)
  --beta BETA    parameter beta for LDA(default=0.1)
  -k K           number of topics
  -i ITERATION   iteration count
  -x X           prevalences column
  -y Y           covariates column
  --sigma SIGMA  initial value of sigma diagonals
  --stopwords    exclude stop words by using corpus from nltk
  --seed SEED    random seed
  --df DF        threshold of document freaquency to cut words
  --interact     consider interaction between covariates and topic
  --sinit        smart initialize of parameters for LDA
 ```

注意
1. `-y` をつけない場合、初期値によっては発散します。
2. `-y` をつけた場合、計算速度が落ちます。また、perplexityがあまり下がらない傾向があります。