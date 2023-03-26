

1.   环境配置

```shell
pip install spacy==3.3.1
pip install pymusas

# 安装 en_dual_none_contextual
pip install https://github.com/UCREL/pymusas-models/releases/download/en_dual_none_contextual-0.3.1/en_dual_none_contextual-0.3.1-py3-none-any.whl
# or
pip intsall en_dual_none_contextual-0.3.1-py3-none-any.whl

python -m spacy download en_core_web_sm
```



2.   测试脚本

```python
import spacy

nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
english_tagger_pipeline = spacy.load('en_dual_none_contextual')
nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)

doc = nlp("The Nile is a major north-flowing river in Northeastern Africa.")
for word in doc:
    print(word._.pymusas_tags)
```



>   参考链接：

1.   可能遇到的问题 `TypeError: load_model_from_init_py() got an unexpected keyword argument 'enable'`：https://github.com/UCREL/pymusas/issues/34

2.   `en_dual_none_contextual-0.3.1-py3-none-any.whl` 下载链接：https://www.epsiotapi.com:20443/share.cgi?ssid=57a0512daf384f59aebc9ba13660a4d8#57a0512daf384f59aebc9ba13660a4d8