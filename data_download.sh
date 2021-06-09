mkdir dataset
cd dataset

wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip glove.6B.300d.txt glove.6B.200d.txt glove.6B.50d.txt
wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/shown_photos.tgz
wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids

cd ..