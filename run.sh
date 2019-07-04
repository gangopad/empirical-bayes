python preprocess.py
python LDA/lda.py
python ModMax/spectral.py
DAT_PATH="/Users/anirbang/DeltaSierra/Publications/EmpiricalBayes/data/"
K=100
loss=L2

for corpus in nips newsgroup twitter
do
    python preprocess.py
    python LDA/lda.py
    python ModMax/spectral.py
    python learn_topics.py $DAT_PATH/M_$corpus.full_docs.mat.trunc.mat settings.example $DAT_PATH/vocab.$corpus.txt.trunc $K $loss demo_$loss\_out.$corpus.$K
done

python classifier.py
