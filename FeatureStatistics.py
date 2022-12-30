import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=LoadData.loadDataset()

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(data)

fig, axes = plt.subplots(ncols=3, nrows=3)

plt.xlim(1, 10)
plt.xlabel('Clump Thickness')
plt.ylabel('Probability')
sns.distplot(data['Clump Thickness'],hist_kws={'edgecolor':'black'},ax=axes[0,0])
print('Mean of Clump Thickness : ' + str(round(data['Clump Thickness'].mean(),3)))
print('Variance of Clump Thickness : ' + str(round(data['Clump Thickness'].var(),3)))
print('Skewness of Clump Thickness : ' + str(round(data['Clump Thickness'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Uniformity of Cell Size')
plt.ylabel('Probability')
sns.distplot(data['Uniformity of Cell Size'],hist_kws={'edgecolor':'black'},ax=axes[0,1])
print('Mean of Uniformity of Cell Size : ' + str(round(data['Uniformity of Cell Size'].mean(),3)))
print('Vriance of Uniformity of Cell Size : ' + str(round(data['Uniformity of Cell Size'].var(),3)))
print('Skewness of Uniformity of Cell Size : ' + str(round(data['Uniformity of Cell Size'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Uniformity of Cell Shape')
plt.ylabel('Probability')
sns.distplot(data['Uniformity of Cell Shape'],hist_kws={'edgecolor':'black'},ax=axes[0,2])
print('Mean of Uniformity of Cell Shape : '+ str(round(data['Uniformity of Cell Shape'].mean(),3)))
print('Vriance of Uniformity of Cell Shape : '+ str(round(data['Uniformity of Cell Shape'].var(),3)))
print('Skewness of Uniformity of Cell Shape : '+ str(round(data['Uniformity of Cell Shape'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Marginal Adhesion')
plt.ylabel('Probability')
sns.distplot(data['Marginal Adhesion'],hist_kws={'edgecolor':'black'},ax=axes[1,0])
print('Mean of Marginal Adhesion is : '+ str(round(data['Marginal Adhesion'].mean(),3)))
print('Variance of Marginal Adhesion is : '+ str(round(data['Marginal Adhesion'].var(),3)))
print('Skewness of Marginal Adhesion is : '+ str(round(data['Marginal Adhesion'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Single Epithelial Cell Size')
plt.ylabel('Probability')
sns.distplot(data['Single Epithelial Cell Size'],hist_kws={'edgecolor':'black'},ax=axes[1,1])
print('Mean of Single Epithelial Cell Size : '+ str(round(data['Single Epithelial Cell Size'].mean(),3)))
print('Variance of Single Epithelial Cell Size : '+ str(round(data['Single Epithelial Cell Size'].var(),3)))
print('Skewness of Single Epithelial Cell Size : '+ str(round(data['Single Epithelial Cell Size'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Bare Nuclei')
plt.ylabel('Probability')
sns.distplot(data['Bare Nuclei'],hist_kws={'edgecolor':'black'},ax=axes[1,2])
print('Mean of Bare Nuclei : ' + str(round(data['Bare Nuclei'].mean(),3)))
print('Variance of Bare Nuclei : ' + str(round(data['Bare Nuclei'].var(),3)))
print('Skewness of Bare Nuclei : ' + str(round(data['Bare Nuclei'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Bland Chromatin')
plt.ylabel('Probability')
sns.distplot(data['Bland Chromatin'],hist_kws={'edgecolor':'black'},ax=axes[2,0])
print('Mean of Bland Chromatin : ' + str(round(data['Bland Chromatin'].mean(),3)))
print('Vriance of Bland Chromatin : ' + str(round(data['Bland Chromatin'].var(),3)))
print('Skewness of Bland Chromatin : ' + str(round(data['Bland Chromatin'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Normal Nucleoli')
plt.ylabel('Probability')
sns.distplot(data['Normal Nucleoli'],hist_kws={'edgecolor':'black'},ax=axes[2,1])
print('Mean of Normal Nucleoli : '+ str(round(data['Normal Nucleoli'].mean(),3)))
print('Variance of Normal Nucleoli : '+ str(round(data['Normal Nucleoli'].var(),3)))
print('Skewness of Normal Nucleoli : '+ str(round(data['Normal Nucleoli'].skew(),3)))
print()

plt.xlim(1, 10)
plt.xlabel('Mitoses')
plt.ylabel('Probability')
sns.distplot(data['Mitoses'],hist_kws={'edgecolor':'black'},ax=axes[2,2])
print('Mean of Mitoses : ' + str(round(data['Mitoses'].mean(),3)))
print('Vrainace of Mitoses : ' + str(round(data['Mitoses'].var(),3)))
print('Skewness of Mitoses : ' + str(round(data['Mitoses'].skew(),3)))
print()


fig.tight_layout()
plt.show()

plt.xlabel('Class')
plt.ylabel('Count')
sns.countplot(x="Class", data=data, palette="bwr")
print('Mean of Class : ' + str(round(data['Class'].mean(),3)))
print('Variance of Class : ' + str(round(data['Class'].var(),3)))
print('Skewness of Class : ' + str(round(data['Class'].skew(),3)))
plt.show()


