# 这里开始做一个机器学习特征筛选的包
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np

from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import KMeansSMOTE, SMOTE, SVMSMOTE
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================定义通用函数=================================
# 得分计算函数
def round_score(score):
    return np.round(score, 3)


## 模型评价函数===========================================
def cv_score(X_train, y_train, random_state=0):
    model = RandomForestClassifier(n_estimators=1000, max_depth=9, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    score_funcs = [
        'accuracy',
        'precision',
        'recall',
        'roc_auc'
    ]

    scores = cross_validate(model, X_train, y_train, scoring=score_funcs, cv=cv, return_estimator=True)
    print('Mean Accuracy:', round_score(scores['test_accuracy'].mean()))
    print('Mean Precision:', round_score(scores['test_precision'].mean()))
    print('Mean Recall:', round_score(scores['test_recall'].mean()))
    print('Mean ROC AUC:', round_score(scores['test_roc_auc'].mean()))

    return scores


def predict_mean_score(scores, X_test):
    pred_scores = []
    for i in range(len(scores['estimator'])):
        model = scores['estimator'][i]
        pred = model.predict_proba(X_test)
        pred_scores.append(pred)
    pred_score = np.mean(pred_scores, axis=0)

    return pred_score


# ===========================专用函数==========================
## 用来比较3种数据增强方法效果
def data_enhance(df_final_feature,df_merge_y):
    """

    :param df_final_feature:
    :param df_merge_y:
    :param cv:奇怪 不知道有咩有这个有什么区别
    :return:df_scores
    """


    # 计分结果
    score_funcs = [
        "precision-recall",
        'accuracy',
        'precision',
        'recall',
        'roc_auc',
    ]

    # 原始数据
    X = df_final_feature
    Y = df_merge_y
    # 实例化数据增强方法
    Estimator = ["Original", "SVM", "SMOTE", "KMeans"]  # 调整顺序
    kmeans = KMeansSMOTE(random_state=123, k_neighbors=3)
    smote = SMOTE(random_state=123, k_neighbors=3)
    svms = SVMSMOTE(random_state=123, k_neighbors=3)
    estimator_list = ["original", svms, smote, kmeans]  # 调整顺序

    # 创建积分矩阵
    ar = np.random.rand(20).reshape(len(estimator_list), len(score_funcs))  # 行是数据形式，列是评价方法
    df_scores = pd.DataFrame(ar, columns=score_funcs)
    df_scores["label"] = 1

    # 开始数据增强
    for index, estimator in enumerate(estimator_list):
        df_scores["label"].iloc[index] = Estimator[index]
        if estimator != "original":
            x, y = estimator.fit_resample(X, Y);
            print(f"{estimator}结果是 : ", np.unique(y, return_counts=True))

        else:
            x = X;
            y = Y;
            print("这是原始数据")

        # 下面开始执行
        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=0,
            stratify=y,
            shuffle=True  # y_imbalance 是 label的那一列
        )
        scores = cv_score(X_train, y_train, random_state=42)

        # 开始给df
        for name in score_funcs:
            if name != "precision-recall":
                df_scores[name].iloc[index] = round_score(scores["test_" + name].mean())
            else:
                pred_score = predict_mean_score(scores, X_test)
                average_precision = average_precision_score(y_test, pred_score[:, 1])
                print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                df_scores["precision-recall"].iloc[index] = round_score(average_precision)

    return df_scores

## 根绝data_enhance函数返回的df_scores进行计算并打分的函数
def data_enhance_compare(df_scores,location,name):
    """

    :param df_scores:
    :param location:
    :param name:包含格式后缀
    :return: fig
    """

    df_tmp = df_scores.melt(id_vars="label")
    pal2 = sns.color_palette(['#2a557f', '#45bc9c', '#f05076', '#ffcd6e'])
    sns.set_palette(pal2)

    fig = plt.figure(figsize=(6, 5), dpi=300)
    sns.pointplot(x="variable", y="value", hue="label",
                  data=df_tmp)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="lower right")
    plt.savefig(location+name, bbox_inches="tight")
    return fig

## 可视化数据增强前后的分布区别
def data_enhance_show(df_X, df_Y, location, kind="SMOTE"):
    """
    location = ""
    tmp = data_enhance_show(X,Y[["Taste_num"]],location,kind="SMOTE")
    查看数据增强前后的数据分布情况
    fig_names = ["PCA", "KPCA", "ISOMAP", "T-SNE", "MDS示"]
    :param df_X:
    :param df_Y:
    :param location:
    :return:df_duplicated
    """
    # 开始数据增强
    if kind == "SMOTE":
        estimator = SMOTE(random_state=123, k_neighbors=3)
    elif kind == "SVM":
        estimator = SVMSMOTE(random_state=123, k_neighbors=3)
    elif kind == "KMeans":
        estimator = KMeansSMOTE(random_state=123, k_neighbors=3)
    x, y = estimator.fit_resample(df_X, df_Y)
    x["Label"] = "Enhancing"
    df_enhancer = pd.concat([x, y], axis=1)

    df_X["Label"] = "Original"
    df_original = pd.concat([df_X, df_Y], axis=1)

    # 先把两个df拼接在一起
    df_duplicated = pd.concat([df_original, df_enhancer])  # .reset_index(level=0)
    df_duplicated.drop_duplicates(df_original.columns[:-2], inplace=True, ignore_index=True)

    # 开始绘制
    fig_names = ["PCA", "KPCA", "ISOMAP", "T-SNE", "MDS"]
    Xtest = df_duplicated.iloc[:, :-2]
    df_D = df_duplicated
    #     实例化降维器
    pca = PCA(n_components=2, random_state=123)
    kpca = KernelPCA(n_components=2, kernel="poly",  ## 核函数为rbf核
                     gamma=0.2, random_state=123)
    ## 流行学习进行数据的非线性降维
    isomap = Isomap(n_neighbors=2,  ## 每个点考虑的近邻数量
                    n_components=2)  ## 降维到3维空间中
    tsne = TSNE(n_components=2, perplexity=25,  ## TSNE进行数据的降维,降维到3维空间中
                early_exaggeration=3, random_state=123)
    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=123)  ## MDS进行数据的降维,降维到3维空间中

    model_estimators = [pca, kpca, isomap, tsne, mds]
    #     model_estimators = [isomap,tsne,mds]

    Xtest_tmp = StandardScaler().fit_transform(Xtest)
    for (name, method) in zip(fig_names, model_estimators):  # 五种循环的方法
        #         print(Xtest)
        tmp_X = method.fit_transform(Xtest_tmp)
        X_test_MDS = pd.DataFrame(tmp_X, columns=["Features_1", "Features_2"])

        df_fig = pd.concat([df_D, X_test_MDS], axis=1)
        plt.figure(dpi=300)
        # 尝试修改一下颜色
        pall = sns.color_palette(['#73a2c6', '#f4777f'])
        sns.set_palette(pall)

        sns.scatterplot(data=df_fig, x="Features_1", y="Features_2", hue="Taste_num", style="Label")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc="lower right")
        plt.xticks(fontsize=12);
        plt.yticks(fontsize=12)
        plt.title(name + " Display", fontsize=18)
        plt.xlabel("Feature_1", fontsize=15)
        plt.ylabel("Feature_2", fontsize=15)
        plt.savefig(location + name + ".svg", bbox_inches="tight")
        plt.show()

    return df_duplicated



