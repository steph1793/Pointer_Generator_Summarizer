import tensorflow as tf
from model import PGN
from training_helper import train_model
from batcher import batcher

def train(params):
	assert params["mode"].lower() == "train", "change training mode to 'train'"

	tf.compat.v1.logging.info("Building the model ...")
	model = PGN(params)


	tf.compat.v1.logging.info("Creating the batcher ...")
	b = batcher(params["data_dir"], params["vocab_path"], params)

	tf.compat.v1.logging.info("Creating the checkpoint manager")
	logdir = "{}/logdir".format(params["model_dir"])
	checkpoint_dir = "{}/checkpoint".format(params["model_dir"])
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	ckpt.restore(ckpt_manager.latest_checkpoint)
	if ckpt_manager.latest_checkpoint:
		print("Restored from {}".format(ckpt_manager.latest_checkpoint))
	else:
		print("Initializing from scratch.")

	tf.compat.v1.logging.info("Starting the training ...")
	train_model(model, b, params, ckpt, ckpt_manager)