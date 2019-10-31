# Pointer_Generator_Summarizer Tensorflow 2.0.0 (V3)


The pointer generator is a deep neural network built for abstractive summarizations. 
For more informations on this model, https://arxiv.org/pdf/1704.04368

With my collaborator Kevin Sylla , we re-made this model in tensorflow for our research project. This neural net will be our baseline model.
We will do some experiments with this model, and propose a new architecture based on this one.

In this project, you can:
- train models
- test ²
- evaluate ²

This project reads .bin format files. For our experiments, we will be working on the ccn and dailymail datasets.
You can download the preprocessed files with this link : 
https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

Or do the pre-processing by yourself with this link :
https://github.com/abisee/cnn-dailymail


You may launch the program with the following command: (have a look at the main.py script for more informations about the attributes)
<br>
<br>

**python main.py \
--max_enc_len=400 \
--max_dec_len=100 \
--max_dec_steps=120 \
--min_dec_steps=30 \
--batch_size=4 \
--beam_size=4 \
--vocab_size=50000 \
--embed_size=128 \
--enc_units=256 \
--dec_units=256 \
--attn_units=512 \
--learning_rate=0.15 \
--adagrad_init_acc=0.1 \
--max_grad_norm=0.8 \
--mode="eval" \
--checkpoints_save_steps=5000 \
--max_steps=38000 \
--num_to_test=5 \
--max_num_to_eval=100 \
--vocab_path="../../Datasets/tfrecords_folder/tfrecords_folder/vocab" \
--data_dir="../../Datasets/tfrecords_folder/tfrecords_folder/val" \
--model_path="../pgn_model_dir/checkpoint/ckpt-37000" \
--checkpoint_dir="../pgn_model_dir/checkpoint" \
--test_save_dir="../pgn_model_dir/test_dir/" **
