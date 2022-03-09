import pandas as pd
import langdetect   # langdetect，可以判断字符串的语言
import hanlp     #  hanlp拥有：中文分词、命名实体识别、摘要关键字、依存句法分析、简繁拼音转换、智能推荐。
import re
from snownlp import SnowNLP    # 分析语句的态度，积极还是消极
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from fastapi import FastAPI
app = FastAPI()
import schedule
import time
import uvicorn

def add_detect_lang(data, column):
    dtf = data.copy()
    dtf['lang'] = dtf[column].apply(lambda x: langdetect.detect(x) if x.strip() != "" else "")
    #  langdetect.detect(x)  检测x属于哪种语言
    # apply ()的作用就是合并方法和对象的方法和属性，并将方法和对象的this指向合并后的对象
    return dtf

def sent_tokenize(paragraph):
    return re.split('。|！|\!|\.|？|\?|\,|，',paragraph)

def add_text_length(data, column):
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    dtf = data.copy()
    # 词长
    dtf['word_count'] = dtf[column].apply(lambda x: len(HanLP(str(x), tasks = ["tok/fine"])["tok/fine"]) )
    # 字符长度
    dtf['char_count'] = dtf[column].apply(lambda x: len(x) )
    dtf['sentence_count'] = dtf[column].apply(lambda x: len(sent_tokenize(str(x))) )
    dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']
    dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']
    print(dtf[['char_count','word_count','sentence_count','avg_word_length','avg_sentence_lenght']].describe().T[["min","mean","max"]])
    return dtf

def add_sentiment(data, column):
    dtf = data.copy()
    dtf["sentiment"] = dtf[column].apply(lambda x: SnowNLP(str(x)).sentiments )
    return dtf

def train_model(modelName):
    model = SentenceTransformer(modelName)
    train_examples = []
    with open('./train.json', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            sentence1 = line['sentence1']
            sentence2 = line['sentence2']
            label = float(line['label'])
            train_examples.append(InputExample(texts=[sentence1, sentence2], label=label))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, output_path='./newModel')

def embeddingQ(data, reData):
    dfr = data.copy()
    dfr = dfr.values.tolist()
    model = SentenceTransformer('./newModel')
    sentence_embeddings = model.encode(dfr)
    columnsM = []
    for listt in range(len(sentence_embeddings[0])):
        columnName = 'Embedding' + str(listt)
        columnsM.append(columnName)
    emd = pd.DataFrame(sentence_embeddings, columns = columnsM)
    reData = pd.concat([emd, reData], axis=1)
    return reData

# evaluate data drift with Evidently Profile
def detect_dataset_drift(reference, production, column_mapping, confidence=0.95, threshold=0.5, get_ratio=False):
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    drifts = []
    num_features = column_mapping.numerical_features if column_mapping.numerical_features else []
    cat_features = column_mapping.categorical_features if column_mapping.categorical_features else []
    for feature in num_features + cat_features:
        drifts.append(json_report['data_drift']['data']['metrics'][feature]['p_value'])

    n_features = len(drifts)
    n_drifted_features = sum([1 if x < (1. - confidence) else 0 for x in drifts])

    if get_ratio:
        return n_drifted_features / n_features
    else:
        return True if n_drifted_features / n_features >= threshold else False

# @app.get("/")
def uploadfile():
    # 创建分词标准
    # reference
    dfr1 = pd.read_excel("./eservice_data_lite.xlsx", engine="openpyxl")
    dfrr = dfr1["Question"]# 将读取的数据转化为daframe，横坐标为question
    dfr = dfr1[["Question"]]
    dfr = add_detect_lang(dfr, "Question")# 判断哪国语言
    dfr = add_text_length(dfr, "Question")# 判断句子长度
    dfr = add_sentiment(dfr, "Question")# 判断句子态度，积极或者消极

    dfp1 = pd.read_excel("./线上理解正确数据_1128.xlsx", engine="openpyxl")
    dfpp = dfp1["Question"]
    dfp = dfp1[["Question"]]
    dfp = add_detect_lang(dfp, "Question")
    dfp = add_text_length(dfp, "Question")
    dfp = add_sentiment(dfp, "Question")

    # 保存
    dfr["prediction"] = dfr["Question"].apply(lambda x: 1)
    dfp["prediction"] = dfp["Question"].apply(lambda x: 1)
    dfp.to_csv("./production.csv", encoding="utf_8_sig")
    dfr.to_csv("./reference.csv", encoding="utf_8_sig")

    reference = dfr[['word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_lenght', 'sentiment']]
    current = dfp[['word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_lenght', 'sentiment']]

    train_model('all-MiniLM-L6-v2')
    reference = embeddingQ(dfrr, reference)
    current = embeddingQ(dfpp, current)


    column_mapping = ColumnMapping()
    column_mapping.numerical_features = list(reference.columns)

    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    data_drift_dashboard.calculate(reference, current, column_mapping=column_mapping)

    data_drift_dashboard.save("./data_drift_dashboard_monitor.html")

    detect_dataset_drift(reference, current, column_mapping, get_ratio=True)
    # fileName = "./data_drift_dashboard.html"
    # return FileResponse(fileName, filename="data_drift_dashboard.html")
    print("OK!")

# schedule.every().monday.do(uploadfile)
if __name__ == '__main__':
    uploadfile()
    # uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True, debug=True, workers=1)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)