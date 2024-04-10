from metaflow.metaflow_config import DATATOOLS_S3ROOT
from metaflow import S3
import os
import shutil
import subprocess
from config import *
from tempfile import TemporaryDirectory


class DataStore:

    _store_root = DATATOOLS_S3ROOT

    @property
    def root(self):
        return self._store_root

    @staticmethod
    def _walk_directory(root):
        path_keys = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                # create a tuple of (key, path)
                path_keys.append(
                    (
                        os.path.relpath(os.path.join(path, name), root),
                        os.path.join(path, name),
                    )
                )
        return path_keys

    def _upload_directory(self, local_path, store_key=""):
        final_path = os.path.join(self._store_root, store_key)
        with S3(s3root=final_path) as s3:
            s3.put_files(self._walk_directory(local_path))

    def already_exists(self, store_key=""):
        final_path = os.path.join(self._store_root, store_key)
        with S3(s3root=final_path) as s3:
            if len(s3.list_paths()) == 0:
                return False
        return True

    def _download_directory(self, download_path, store_key=""):
        """
        Parameters
        ----------
        download_path : str
            Path to the folder where the store contents will be downloaded
        store_key : str
            Key suffixed to the store_root to save the store contents to
        """
        final_path = os.path.join(self._store_root, store_key)
        os.makedirs(download_path, exist_ok=True)
        with S3(s3root=final_path) as s3:
            for s3obj in s3.get_all():
                move_path = os.path.join(download_path, s3obj.key)
                if not os.path.exists(os.path.dirname(move_path)):
                    os.makedirs(os.path.dirname(move_path), exist_ok=True)
                shutil.move(s3obj.path, os.path.join(download_path, s3obj.key))

    def upload(self, local_path, store_key=""):
        """
        Parameters
        ----------
        local_path : str
            Path to the store contents to be saved in cloud object storage.
        store_key : str
            Key suffixed to the store_root to save the store contents to.
        """
        if os.path.isdir(local_path):
            self._upload_directory(local_path, store_key)
        else:
            final_path = os.path.join(self._store_root, store_key)
            with S3(s3root=final_path) as s3:
                s3.put_files([(local_path, local_path)])

    def download(self, download_path, store_key=""):
        """
        Parameters
        ----------
        store_key : str
            Key suffixed to the store_root to download the store contents from
        download_path : str
            Path to the folder where the store contents will be downloaded
        """
        if not self.already_exists(store_key):
            raise ValueError(
                f"Model with key {store_key} does not exist in {self._store_root}"
            )
        self._download_directory(download_path, store_key)

    def upload_hf_dataset(self, dataset, store_key=""):
        """
        Parameters
        ----------
        dataset : datasets.Dataset
            Huggingface dataset to be saved in cloud object storage.
        store_key : str
            Key suffixed to the store_root to save the store contents to.
        """
        with TemporaryDirectory() as temp_dir: 
            # OK for small data. For big data, use a different method.
            dataset.save_to_disk(temp_dir)
            self.upload(temp_dir, store_key)


class ModelOps:
    def _preprocess(self, train_dataset):
        print(
            "\nPreprocessing data - each organ type represented will be printed sequentially...\n"
        )
        from collections import Counter

        dataset_list = []
        evalset_list = []
        organ_list = []
        target_dict_list = []

        for organ in Counter(train_dataset["organ_major"]).keys():
            # collect list of tissues for fine-tuning (immune and bone marrow are included together)
            if organ in ["bone_marrow"]:
                continue
            elif organ == "immune":
                organ_ids = ["immune", "bone_marrow"]
                organ_list += ["immune"]
            else:
                organ_ids = [organ]
                organ_list += [organ]

            print(organ)

            # filter datasets for given organ
            def if_organ(example):
                return example["organ_major"] in organ_ids

            trainset_organ = train_dataset.filter(if_organ, num_proc=NUM_CPUS)

            # per scDeepsort published method, drop cell types representing <0.5% of cells
            celltype_counter = Counter(trainset_organ["cell_type"])
            total_cells = sum(celltype_counter.values())
            cells_to_keep = [
                k for k, v in celltype_counter.items() if v > (0.005 * total_cells)
            ]

            def if_not_rare_celltype(example):
                return example["cell_type"] in cells_to_keep

            trainset_organ_subset = trainset_organ.filter(
                if_not_rare_celltype, num_proc=NUM_CPUS
            )

            # shuffle datasets and rename columns
            trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=42)
            trainset_organ_shuffled = trainset_organ_shuffled.rename_column(
                "cell_type", "label"
            )
            trainset_organ_shuffled = trainset_organ_shuffled.remove_columns(
                "organ_major"
            )

            # create dictionary of cell types : label ids
            target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
            target_name_id_dict = dict(
                zip(target_names, [i for i in range(len(target_names))])
            )
            target_dict_list += [target_name_id_dict]

            # change labels to numerical ids
            def classes_to_ids(example):
                example["label"] = target_name_id_dict[example["label"]]
                return example

            labeled_trainset = trainset_organ_shuffled.map(
                classes_to_ids, num_proc=NUM_CPUS
            )

            # create 80/20 train/eval splits
            labeled_train_split = labeled_trainset.select(
                [i for i in range(0, round(len(labeled_trainset) * 0.8))]
            )
            labeled_eval_split = labeled_trainset.select(
                [
                    i
                    for i in range(
                        round(len(labeled_trainset) * 0.8), len(labeled_trainset)
                    )
                ]
            )

            # filter dataset for cell types in corresponding training set
            trained_labels = list(Counter(labeled_train_split["label"]).keys())

            def if_trained_label(example):
                return example["label"] in trained_labels

            labeled_eval_split_subset = labeled_eval_split.filter(
                if_trained_label, num_proc=NUM_CPUS
            )

            dataset_list += [labeled_train_split]
            evalset_list += [labeled_eval_split_subset]

        trainset_dict = dict(zip(organ_list, dataset_list))
        traintargetdict_dict = dict(zip(organ_list, target_dict_list))
        evalset_dict = dict(zip(organ_list, evalset_list))

        return trainset_dict, traintargetdict_dict, evalset_dict, organ_list

    def compute_metrics(self, pred):
        from sklearn.metrics import accuracy_score, f1_score

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    def _finetune(
        self,
        organ,
        organ_trainset,
        organ_evalset,
        organ_label_dict,
        checkpoint_path,
        pretrained_path="/Geneformer/geneformer-12L-30M",  # absolute path set in Dockerfile
    ):

        print("Finetuning model for organ: ", organ)

        from transformers import BertForSequenceClassification
        from transformers import Trainer
        from transformers.training_args import TrainingArguments
        from geneformer import DataCollatorForCellClassification
        import datetime
        import pickle
        import subprocess
        import seaborn as sns

        sns.set()

        # set logging steps
        logging_steps = round(len(organ_trainset) / GENEFORMER_BATCH_SIZE / 10)

        # reload pretrained model
        model = BertForSequenceClassification.from_pretrained(
            pretrained_path,
            num_labels=len(organ_label_dict.keys()),
            output_attentions=False,
            output_hidden_states=False,
        ).to("cuda")

        # define output directory path
        current_date = datetime.datetime.now()
        datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
        output_dir = f"{checkpoint_path}/{datestamp}_geneformer_CellClassifier_{organ}_L{MAX_INPUT_SIZE}_B{GENEFORMER_BATCH_SIZE}_LR{MAX_LR}_LS{LR_SCHEDULE_FN}_WU{WARMUP_STEPS}_E{EPOCHS}_O{OPTIMIZER}_F{FREEZE_LAYERS}/"

        # ensure not overwriting previously saved model
        saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
        if os.path.isfile(saved_model_test) == True:
            raise Exception("Model already saved to this directory.")

        # make output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # set training arguments
        training_args = {
            "learning_rate": MAX_LR,
            "do_train": True,
            "do_eval": True,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": logging_steps,
            "group_by_length": True,
            "length_column_name": "length",
            "disable_tqdm": False,
            "lr_scheduler_type": LR_SCHEDULE_FN,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": 0.001,
            "per_device_train_batch_size": GENEFORMER_BATCH_SIZE,
            "per_device_eval_batch_size": GENEFORMER_BATCH_SIZE,
            "num_train_epochs": EPOCHS,
            "load_best_model_at_end": True,
            "output_dir": output_dir,
        }

        training_args_init = TrainingArguments(**training_args)

        # create the trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=organ_trainset,
            eval_dataset=organ_evalset,
            compute_metrics=self.compute_metrics,
        )
        # train the cell type classifier
        trainer.train()
        predictions = trainer.predict(organ_evalset)
        with open(f"{output_dir}predictions.pickle", "wb") as fp:
            pickle.dump(predictions, fp)
        trainer.save_metrics("eval", predictions.metrics)
        trainer.save_model(output_dir)
        return output_dir
