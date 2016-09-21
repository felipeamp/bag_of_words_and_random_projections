#!/bin/bash

URL_DATASET=http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz
FILENAME=docword.nytimes.txt
DATASET_DIR=dataset

echo "Downloading database file from UCI repository..."
wget -c -t 10 -P $DATASET_DIR $URL_DATASET

echo "Uncompressing downloaded file..."
gzip -d ${DATASET_DIR}/${FILENAME}.gz

echo "Selecting only the first one thousand documents..."
# Deletes the first three lines and from line 1,001 to
# the end of the file
sed -e '1,3 d' -e '/^1001 /,$ d' ${DATASET_DIR}/${FILENAME} > ${DATASET_DIR}/${FILENAME}_preprocessed.txt

echo "Done!"

