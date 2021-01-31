# Long Short-Term Session Search: Joint Personalized Reranking and Next Qery Prediction (WWW 2021)
This repo share data and code of `LostNet` to facilitate reproducibility of our paper's results. 
## Model Architecture
![](https://github.com/QiannanCheng/LostNet/blob/main/image.png)
## Requirements
* python=3.6
* tensorflow-gpu=1.9.0
* numpy=1.16.4
## Datasets
We employ two benchmark datasets in our experiments: the AOL search log and SogouQ.
### Original corpus
* **AOL:** The AOL search log is an English language dataset containing 3 months of real users’ query click data, and you can manually download them at [here](http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/).
* **SogouQ:** SogouQ is a Chinese web search log dataset that includes about one month of queries and user clicks from the Sogou search engine, and you can get them at [here](https://www.sogou.com/labs/resource/q.php).
### Preprocessed data
* Both datasets have already been processed into our defined format, which could be directly used by our model. <br>
You can manually download the datasets at [here](https://drive.google.com/drive/folders/1SoeXgZDLTdUhqfQV1I3I8HivsU_wEwxb?usp=sharing), and please create folder `data` in the root directory and put the files in it.
## Quick Running
