import sys

# sys.path.append("../similarity_grouped_query_attention_pytorch")
sys.path.insert(0, "..")
import wandb
import torch
import config
from utils import train
import os
import torch.distributed as dist


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, run,dataset_name,kv_heads,weight_flag,logging_name):
    setup(rank, world_size)
    val_rouge_dict, test_rouge_dict = train(
        rank,
        world_size,
        dataset_name = dataset_name,
        kv_heads=int(kv_heads),
        logging_name=logging_name,
        run=run,
        model_name=config.MODEL_NAME,
        similarity_flag=False,
        weight_flag=weight_flag,
    )
    print(f"validation rogue dict:{val_rouge_dict}")
    print(f"Test rogue dict:{test_rouge_dict}")
    cleanup()


if __name__ == "__main__":
    dataset_name, kv_heads,weight_flag,logging_name = sys.argv[1:]

    weight_flag = int(weight_flag)==1

    if dataset_name not in ["arxiv","wmt14","pubmed","cnn_dailymail","multi_news"]:
        raise "Usage Error dataset should be in : arxiv,wmt14,pubmed,cnn_dailymail,multi_news"
    
    wandb.login(key=config.WANDB_API_KEY)
    run = wandb.init(
        project=config.WANDB_PROJECT,
        config={"model": config.MODEL_NAME, "gqa_list": config.GQA_LIST},
        entity=config.WANDB_ENTITY,
        group=logging_name.upper(),
    )

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main,
        args=(
            world_size,
            run,
            dataset_name,
            kv_heads,
            weight_flag,
            logging_name
        ),
        nprocs=world_size,
        join=True,
    )
