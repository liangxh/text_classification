#! /bin/bash
set -e

MODE=$1
KEY=$2

DIR_LAB=${HOME}/lab
PATH_TO_TEXT=${DIR_LAB}/task2/${MODE}/${KEY}.text
PATH_TO_TEXT_ARFF=${DIR_LAB}/task2/${MODE}/${KEY}.arff
PATH_TO_VEC_ARFF=${DIR_LAB}/task2/${MODE}/${KEY}.lexicon_feat.arff
PATH_TO_VEC=${DIR_LAB}/task2/${MODE}/${KEY}.lexicon_feat

WEKA_JAR=${DIR_LAB}/weka/weka/weka.jar

echo 'generating .arff for text..'

cd ${DIR_LAB}/text_classification
python -m scripts.task2.process_arff text -i ${PATH_TO_TEXT} -o ${PATH_TO_TEXT_ARFF}


echo 'building lexicon feature vector...'

java -Xmx4G -cp ${WEKA_JAR} weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector  \
	-i ${PATH_TO_TEXT_ARFF} -o ${PATH_TO_VEC_ARFF} \
	-stemmer weka.core.stemmers.NullStemmer \
	-stopwords-handler "weka.core.stopwords.Null " \
	-I 1 -U -tokenizer "weka.core.tokenizers.TweetNLPTokenizer " \
	-A -D -F -H -J -N -P -Q -R -T

echo 'transforming .arff of lexicon feature to customized format'

python -m scripts.task2.process_arff vec -i ${PATH_TO_VEC_ARFF} -o ${PATH_TO_VEC}

echo 'remove temporary files'

rm ${PATH_TO_TEXT_ARFF}
rm ${PATH_TO_VEC_ARFF}
