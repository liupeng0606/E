# semeval 2019的第一个问题使用集成学习，学习器用的是我以前写的cnn，rnn分类器，但是出现了错误，问了学长，学长也解决不了
```

clf_cnn = KerasClassifier(build_fn=create_cnn_model,batch_size=32, nb_epoch=10)
clf_rnn = KerasClassifier(build_fn=create_bilstm_attention_model,nb_epoch=10, batch_size=32)


eclf1 = VotingClassifier(estimators=[   ('clf_cnn1', clf_cnn),
                                        ('clf_cnn2', clf_cnn),
                                        ('clf_rnn1', clf_rnn),
                                        ('clf_rnn2', clf_rnn)
                                    ])

eclf1.fit(x_train, y_train)


y_pred = eclf1.predict(x_test)

```
## 错误log

```
Traceback (most recent call last):
  File "/home/liu/PycharmProjects/hello/E.py", line 159, in <module>
    y_pred = eclf1.predict(x_test)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/ensemble/voting_classifier.py", line 242, in predict
    maj = self.le_.inverse_transform(maj)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/preprocessing/label.py", line 273, in inverse_transform
    y = column_or_1d(y, warn=True)
  File "/usr/local/lib/python2.7/dist-packages/sklearn/utils/validation.py", line 788, in column_or_1d
    raise ValueError("bad input shape {0}".format(shape))
ValueError: bad input shape (1, 2)

Process finished with exit code 1
```

## 第一个问题使用集成学习，我尝试使用官方的demo，没有出现问题
```
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                      random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

eclf1 = eclf1.fit(x_train, y_train)

y_pred = eclf1.predict(x_test)


score = accuracy_score(y_test, y_pred)

print score
```

## semeval 2019的第二个问题，官方说可能需要外部的数据源，我不知道如何使用

```
 Usage of external information:
Participants may use external sources of information to perform classification for both Subtask A and Subtask B:

 Intra-forum evidence: from the QatarLiving forum itself. Old threads in the forum may contain enough information to estimate the factuality of the answers in Subtask B. You can download an archive with QL threads from here: http://alt.qcri.org/semeval2016/task3/data/uploads/QL-unannotated-data-subtaskA.xml.zip
Web information: The use of all sources of web information is allowed as participants seem fit. For example, to perform factuality classification for Subtask B, participants may query a search engine to fetch relevant documents from the Internet.
Example uses of such sources are described in https://arxiv.org/pdf/1803.03178.pdf.
```

