DATA_KEY = "Genecorpus-30M"
DATA_DIR = "cell_type_train_data.dataset"
MODEL_CHECKPOINT_DIR = "cell_type_classifier_checkpoints"
DATA_NOT_FOUND_MESSAGE = f"""Data not found in the {DATA_DIR} directory, and not found in the S3 cache.
Please download the data from https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset and place it in the working directory inside of the {DATA_DIR} folder."""

IMAGE = "public.ecr.aws/outerbounds/geneformer:latest"

# set model parameters
# max input size
MAX_INPUT_SIZE = 2**11  # 2048

# set training hyperparameters
# max learning rate
MAX_LR = 5e-5
# how many pretrained layers to freeze
FREEZE_LAYERS = 0
# number gpus
NUM_GPUS = 1
# number cpu cores
NUM_CPUS = 4
# batch size for training and eval
GENEFORMER_BATCH_SIZE = 1 # NOTE: 12 raised OOM on 16GB GPU
# learning schedule
LR_SCHEDULE_FN = "linear"
# warmup steps
WARMUP_STEPS = 5  # 500
# number of epochs
EPOCHS = 2  # 10
# optimizer
OPTIMIZER = "adamw"
