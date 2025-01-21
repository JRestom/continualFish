# 

* EM computes Exact Matching (the metric to evaluate the accuracy of the model)
* FFT performs full finetuning on a given model for the dataset selected
* getGSM8K will download the GSM8K (math reasoning dataset) from huggingface (First you need to log in using your huggingFace token)
* getModel will download the model (for now we chose tinyLlama, which has around 1B parameters)
* answers will display some examples of the model answering a question from the dataset 

# TODO:
* EM is very slow and needs to be batched for faster computation
* The model is not learning the task, we need to try with a different model or check the training scheme (look into mixtral versions)
* Start with the other datasets (perhaps the non math related will be easier to learn and anyway will be needed for the sequential training)
