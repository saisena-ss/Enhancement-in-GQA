## ENHANCEMENT TO GROUPED-QUERY ATTENTION

### Abstract
The attention mechanism forms the foundational blocks for transformer language models. Recent approaches show that scaling the model achieves human-level performance. However, with increasing demands for scaling and constraints on hardware memory, the inference costs of these models remain high. To reduce the inference time, Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) were proposed in (Shazeer, 2019) and (Ainslieet al., 2023).
 
 In this paper, we propose a variation of Grouped-Query Attention, termed Weighted Grouped-Query Attention (WGQA). We introduce new learnable parameters for each key and value head in the T5 decoder attention blocks, enabling the model to take a weighted average during finetuning. Our model achieves an average of 0.53% improvement over GQA, and the performance converges into traditional MHA with additional overhead during inference. We believe that the introduction of these parameters and subsequent finetuning informs the model about the grouping mechanism during training, thereby enhancing performance in fewer steps. Additionally, we demonstrate the scaling laws in our analysis by comparing the results between T5-small and T5-base.


### How to run:
To run experiments, navigate to the benchmark directory and run the following commands:

```bash
python ./main_distributed.py cnn_dailymail 1 1 COLRANDWMQA true col 12355
```

Here the first one "1" after the dataset represents the number of key-value heads, second "1" represents the weight flag whether to use any weights for key-value heads, "COLRANDWMQA" represents the logging name, true represents whether to use random weights or not for additional parameters, "col" represents column wise GQA and 12355 is the master port address for distributed data-parallel.

