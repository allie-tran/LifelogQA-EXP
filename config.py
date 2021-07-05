
params = {
    'load': False,
    'is_train': True,

    # Hyper Params
    'gpu': 'gpu0',
    'learning_rate': 0.01,
    'exp_decay_rate': 0.999,
    'dropout': 0.2,
    'num_epochs': 5,
    'hidden_size': 100,
    'batch_size': 6,

    # Model agnostic thresholds - verify before run
    'word_count_thres': 2,
    'char_count_thres': 10,
    'num_albums_thres': 4,
    'num_photos_thres': 8,
    'sent_des_size_thres': 10,
    'sent_album_title_size_thres': 8,
    'sent_photo_title_size_thres': 8,
    'sent_when_size_thres': 6,
    'sent_where_size_thres': 6,
    'answer_size_thres': 6,
    'question_size_thres': 25,
    'word_size_thres': 16,

    'char_emb_size': 100,
    'char_out_size': 100,
    'image_trans_dim': 100,
    'char_channel_size': 100,
    'char_channel_width': 5,
    'image_feat_dim': 2537,
}

paths = {
    'out_path': '/Users/naushad/Documents/memexqa/prepro_v1.1/',
    'shared_path': '/Users/naushad/Documents/memexqa/shared_out/'
}


