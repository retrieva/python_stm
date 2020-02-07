# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# (c)2018-2019 Hiroki Iida / Retrieva Inc.

import numpy as np
import pandas as pd
import stm


def main():
    import argparse
    import vocabulary
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename", help="set corpus filepath.fileformat is csv")
    parser.add_argument("-d", dest="document", help="set document field name")
    parser.add_argument("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_argument("--alpha", dest="alpha", type=float, help="parameter alpha for LDA(default=1.0)", default=1.0)
    parser.add_argument("--beta", dest="beta", type=float, help="parameter beta for LDA(default=0.1)", default=0.1)
    parser.add_argument("-k", dest="topics", type=int, help="number of topics(default=20)", default=20)
    parser.add_argument("-i", dest="iteration", type=int, help="iteration count(default=100)", default=100)
    parser.add_argument("-x", dest="X", type=str, help="set prevalences column name", default=None)
    parser.add_argument("-y", dest="Y", type=str, help="set covariates column name", default=None)
    parser.add_argument("--parser", dest="parser", help="select parser eng_nltk or mecab(default=mecab)", default="mecab")
    parser.add_argument("--sigma", dest="sigma", help="initial value of sigma diagonals(default=0.1)", default=0.1)
    parser.add_argument("--stopwords", dest="stopwords", help="exclude stop words by using corpus from nltk",
                        action="store_true", default=False)
    parser.add_argument("--seed", dest="seed", type=int, help="random seed")
    parser.add_argument("--df", dest="df", type=int, help="threshold of document freaquency to cut words", default=0)
    parser.add_argument("--interact", dest="interact", action="store_true",
                        help="consider interaction between covariates adn topics", default=False)
    parser.add_argument("--sinit", dest="smartinit", action="store_true",
                        help="smart initialize of parameters for LDA", default=False)
    options = parser.parse_args()
    if not (options.filename or options.corpus):
        parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        load_doc = pd.read_csv(options.filename)
        if options.parser.lower() == "eng_nltk":
            corpus = vocabulary.load_dataframe(load_doc[options.document])
        elif options.parser.lower() == "mecab":
            corpus = vocabulary.load_dataframe_jp(load_doc[options.document])
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus:
            parser.error("corpus range(-c) forms 'start:end'")

    if options.seed is not None:
        np.random.seed(options.seed)

    print("proc voca")
    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]

    # process prevarence, if it is pointed
    print("proc X")
    if options.X is not None:
        X = pd.get_dummies(load_doc[options.X.split(',')], drop_first=True).values
        X = np.concatenate((np.ones(X.shape[0])[:, np.newaxis], X), axis=1)
    else:
        X = options.X

    print("proc Y")
    if options.Y is not None:
        Y = pd.get_dummies(load_doc[[options.Y]], drop_first=True).values.flatten()
    else:
        Y = options.Y

    if options.df > 0:
        docs = voca.cut_low_freq(docs, options.df)

    print("set STM obj")
    stm_obj = stm.STM_factory_method(options.topics, X, Y, docs, voca.size(), options.sigma, options.interact)
    print("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.topics, options.alpha, options.beta))

    # import cProfile
    # cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    print("lda_initialize")
    stm_obj.lda_initialize(options.alpha, options.beta, 10, voca, options.smartinit)
    print("stm_learning")
    stm_obj.learning(options.iteration, voca)


if __name__ == "__main__":
    main()
