from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
fit_result = mlb.fit_transform([(1, 2), (3,)])
print(fit_result)
print(mlb.classes_)
fit_result = mlb.fit_transform([set(['sci-fi', 'thriller']), set(['comedy'])])
print(fit_result)
print(mlb.classes_)
list = [["请自由","df"],["asdf1"]]
fit_result = mlb.fit_transform(list)
print(fit_result)
print(mlb.classes_)
