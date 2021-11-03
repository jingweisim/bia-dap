#https://www.youtube.com/watch?v=5X27excCyXk
from sklearn.model_selection import train_test_split
X = df['Text']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)

DT.score(xv_test,y_test)

predict_DT = DT.predict(xv_test)

print('Decision Tree score:')
print(classification_report(y_test,predict_DT))

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)

GBC.score(xv_test,y_test)

predict_GBC = GBC.predict(xv_test)

print('Gradient Boosting score:')
print(classification_report(y_test,predict_GBC))

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)

RFC.score(xv_test,y_test)

predict_RFC = RFC.predict(xv_test)

print('Random Forest Score:')
print(classification_report(y_test,predict_RFC))
