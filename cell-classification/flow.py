# built-in
import os
import sys
import subprocess

# 3rd party
from metaflow import FlowSpec, step, kubernetes, current, Parameter, current, card
from metaflow.profilers import gpu_profile
from metaflow.cards import Markdown

# custom
from utils import DataStore, ModelOps
from config import *


class CellClassificationFinetuning(FlowSpec, DataStore, ModelOps):
    """
    This workflow runs a finetuning of the geneformer model for a cell classification task.
    You can find the source code that is heavily used in the mixins for the class here:
        https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/cell_classification.ipynb

    Workflow and training hyperparameters are set in config.py.
    """

    pretrained_geneformer_path = Parameter(
        "path",
        help="location of the Geneformer model in local execution context or inside your Docker image",
        type=str,
        default="/Geneformer/geneformer-12L-30M",
    )

    @step
    def start(self):
        s3_path = os.path.join(DATA_KEY, DATA_DIR)
        if not self.already_exists(s3_path):
            if not os.path.exists(DATA_DIR):
                sys.exit(DATA_NOT_FOUND_MESSAGE)
            self.upload(local_path=DATA_DIR, store_key=s3_path)
        self.next(self.preprocess)

    @step
    def preprocess(self):
        from datasets import load_from_disk

        self.download(
            download_path=DATA_DIR, store_key=os.path.join(DATA_KEY, DATA_DIR)
        )
        train_dataset = load_from_disk(DATA_DIR)
        (
            trainset_dict,
            traintargetdict_dict,
            evalset_dict,
            organ_list,
        ) = self._preprocess(train_dataset)
        self.model_splits = []
        for organ in organ_list[:1]:
            organ_trainset = trainset_dict[organ]
            organ_evalset = evalset_dict[organ]
            organ_label_dict = traintargetdict_dict[organ]
            train_key = os.path.join(DATA_KEY, str(current.run_id), f"{organ}_trainset")
            self.upload_hf_dataset(organ_trainset, train_key)
            eval_key = os.path.join(DATA_KEY, str(current.run_id), f"{organ}_evalset")
            self.upload_hf_dataset(organ_evalset, eval_key)
            self.model_splits.append(
                {
                    "organ": organ,
                    "organ_trainset_key": train_key,
                    "organ_evalset_key": eval_key,
                    "organ_label_dict": organ_label_dict,
                }
            )
        self.next(self.finetune, foreach="model_splits")

    # @kubernetes(gpu=NUM_GPUS, cpu=NUM_CPUS, image=IMAGE)
    @gpu_profile(interval=1)
    @card(type="blank", refresh_interval=5, customize=True, id="train_progress")
    @card(type="blank", id="outputs")
    @step
    def finetune(self):
        from datasets import load_from_disk

        self.download(
            download_path=self.input["organ"] + "_trainset",
            store_key=self.input["organ_trainset_key"],
        )
        self.download(
            download_path=self.input["organ"] + "_evalset",
            store_key=self.input["organ_evalset_key"],
        )
        output_dir = self._finetune(
            self.input["organ"],
            load_from_disk(self.input["organ"] + "_trainset"),
            load_from_disk(self.input["organ"] + "_evalset"),
            self.input["organ_label_dict"],
            MODEL_CHECKPOINT_DIR,
            pretrained_path=self.pretrained_geneformer_path
        )
        store_key = os.path.join(DATA_KEY, str(current.run_id), output_dir)
        self.upload(local_path=output_dir, store_key=store_key)
        current.card['outputs'].append(Markdown(f"# Saving model to S3 key {store_key}"))
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("Flow is done!")


if __name__ == "__main__":
    CellClassificationFinetuning()
