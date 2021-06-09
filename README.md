# LifelogQA-EXP
Run baseline experiments on lifelogQA

TODO!
- Re-implement MemexQA system specified in https://memexqa.cs.cmu.edu/ in Pytorch
- Apply on Lifelog data and observe the result

# Dataset website
https://memexqa.cs.cmu.edu/index.html
Explore dataset: https://memexqa.cs.cmu.edu/explore/1.html

# Dataset download
If on server: the data is already downloaded. Run `source env_variables_setup.sh` to link to the path.
Otherwise, run `sh data_download.sh` to download and extract data to `dataset` folder.

To process data, run
```
python preprocess.py $MEMEX_DATA/qas.json $MEMEX_DATA/album_info.json \
$MEMEX_DATA/test_question.ids $MEMEX_DATA/photos_inception_resnet_v2_l2norm.npz \
$MEMEX_DATA/glove.6B.100d.txt $MEMEX_DATA/preprocessed
```

# Train FTVA model
```
python ftva.py $MEMEX_DATA/preprocessed models/ --modelname fvta --hidden_size 50 --image_feat_dim 2537 \
--use_image_trans --image_trans_dim 100 --use_char --char_emb_size 100 --char_out_size 100 --add_tanh \
--simiMatrix 2 --use_question_att --use_3d --use_time_warp --warp_type 5 --is_test --load_best \
--batch_size 1
```