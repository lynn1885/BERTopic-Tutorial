# 模板代码的使用

这是B站视频课程[通俗易懂的BERTopic主题模型教程（可代替LDA、DTM），39集全套教程](https://space.bilibili.com/10989976/channel/collectiondetail?sid=2819272&spm_id_from=333.788.0.0)对应的完整笔记、数据、代码。详细视频教程请点击链接查看。

1. **⭐ 模板代码是`main.ipynb`，下载该项目，按视频课程配置好环境后，该文件中的代码可一键运行，提升科研效率！！！⭐**

1. **安装依赖 & 运行代码**：
    - 整个教程基于Python 3.10.x版本（具体到我的环境，是Python 3.10.6）
    - 需要先安装Anaconda，[下载Anaconda](https://www.anaconda.com/download)
    - 先安装HDBSCAN：`conda install -c conda-forge hdbscan`
    - 安装其他依赖：`pip install torch==2.0.1 transformers==4.29.1 tqdm==4.65.0 numpy==1.23.2 jieba==0.42.1 bertopic==0.15.0 nbformat==5.9.0`
    - 如果运行过程中提示缺少其他依赖，自行安装即可
    - 然后，直接运行`main.ipynb`，就能看到运行结果了

# 使用自己的数据
1. **准备数据**：首先，您需要准备一份`文本.txt`和一份`时间.txt`，放入`data/`目录
    - `文本.txt`是切词前的语料，一个文档对应一行
    - `时间.txt`对应每条文本的年份
    - `文本.txt`和`时间.txt`行数相同，比如都是1000行，代表1000行文本及其对应时间
    - 放入`data/`目录，您可以参考`data/`目录下的文件示例

1. **切词**：来到`分词`目录，运行`cut_word.py`，会生成`data/切词.txt`
    - 用户字典在`分词/userdict.txt`中设置
    - 停用词在`分词/stopwords.txt`中设置

1. **生成词嵌入**：来到`embedding`目录，运行其中一个ipynb文件，会生成`emb.npy`
    - 比如运行 `embedding_bert.ipynb`，会调用bert-base-chinese模型生成词向量
    - 比如运行 `embedding_sentence_transformer.ipynb`，会调用Sentencetransformers模型生成词向量
    - 把生成的词向量，复制到`data`目录，修改为`embedding_bbc.npy`等文件名，具体参考`data/`目录中的文件命名
    - 如果要使用autodl等线上GPU平台，则可以将`data/文本.txt`和`线上代码平台/`目录中的`embedding_xxx.ipynb`上传到线上平台，运行，生成Embedding文件并下载到本地

1. **运行**：运行`main.ipynb`，生成聚类结果

# 项目目录
1. `data/`：项目所需的各类数据
1. `embedding/`：其中包含的是生成词向量的相关代码，用于在本地运行
1. `线上平台代码/`：同样的，其中包含的是生成词向量的相关代码，但用于在autodl等线上GPU平台运行
1. `笔记/`：视频课程中用到的笔记
1. `test/`：视频课程中编写的各类代码案例
1. `main.ipynb`：使用BERTopic的模板文件，您的项目可基于该文件进行改写 ⭐
1. `README.md`：项目说明


# [⭐点击这里，查看课程笔记⭐](./笔记/note.md)