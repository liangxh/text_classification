# Semeval2018 Task2

## 文件路徑

* 默認訓練數據在 ${HOME}/lab/task2，有特殊情況可以透過修過task2/model/config中的dir_task2參數

## 自定義詞匯

* task_key: 子任務的標識符, 如us為英語數據集, es為西班牙語數據集
* mode: 指 train/trial/test

## 數據預處理

### 分詞

* 對文本進行分詞以及一些特殊處理(如數字轉換成'\<NUM\>'等)，生成 .tokenized文件，以空格分開各個token

```
python -m task2.scripts.dataset tokenize -k us -m train
python -m task2.scripts.dataset tokenize -k us -m trial
```

### 生成詞典

* 基於分詞結果統計token出現情況, 默認基於train數據集進行統計, 生成 .vocab文件，可以根據其中的數據決定模型中字典集的大小

```
python -m task2.scripts.dataset vocab -k us
```

### 生成Lexicon Feature

* 利用第三方庫AffectiveTweets為每條tweet生成一個向量
* 運行此程序僅返回提示信息，需要手動在終端運行提示的命令

```
python -m task2.scripts.dataset lexicon -k us -m train
```

### 生成調試用的數據子集

* 為方便測試代碼是否能正確運行，透過生成測試用子集進行調試

```
python -m task2.scripts.dataset subset -i us -o us2 -N 2 -m train -l 4000
python -m task2.scripts.dataset subset -i us -o us2 -N 2 -m trial -l 4000
python -m task2.scripts.dataset vocab -k us2
```

## 運行

* 本庫已實現幾個神經網絡，見task2/nn/
* 每個神經網絡對應一個配置模板，見task2/config/，調參數只需要拷貝出模板並修改參數即可
* 訓練: ```python -m task2.main train -c task2/task_config/gru.yaml```
    * 模型將保存至 ${HOME}/lab/task2/train_checkpoint/{TASK_KEY}{ALGORITHM_NAME}/
    * 配置文件也會備份到該目錄下

## 其他

### Frederic Godin模塊索引生成

Frederic Goldin為Word2vec的模塊之一, 需要建立索引方便專門的任務進行讀取
```
python -m task2.scripts.word_embed fg -n 100000
```
