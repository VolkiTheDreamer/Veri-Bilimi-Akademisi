import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def numericImputer(X:pd.DataFrame):
    #integer için median, float için mean olacak
    #bunu bilerek hatalı bir şekilde hazırlıyorum, çünkü test datasındaki nullar replace edilirken kendi ortalaması ile replace edilecek, trainiki ile değil
    X.loc[:,X.select_dtypes(np.float64).columns]=X.loc[:,X.select_dtypes(np.float64).columns].fillna(X.select_dtypes(np.float64).mean())
    X.loc[:,X.select_dtypes(np.int64).columns]=X.loc[:,X.select_dtypes(np.int64).columns].fillna(X.select_dtypes(np.int64).median())
    return X.values #numpy döndürsün diye   
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    # Class Constructor, bi parametre alacaksa onu da veriyoruz
    # normalde columntaransformden zaten bi kolon listesi gelecek(nums), ama biz hepsini değil, sadece bi kısmını kullanacağız
    # aslında olur da ilerde diğer kolonlarda da outlier gelir diye hepsine yapmak lazım ama bu, işlem süresini uzatacaktır
    def __init__(self, featureindices):
        self.featureindices = featureindices
  
    def fit(self, X:np.array, y = None):
        # y’yi kullamayacak olsak bile ilgili fonk içinde bulundururuz, None atarız
        Q1s = np.quantile(X[:,self.featureindices],0.25,axis=0)
        Q3s = np.quantile(X[:,self.featureindices],0.75,axis=0)
        IQRs = Q3s-Q1s
        self.top=(Q3s + 1.5 * IQRs)
        self.bottom=(Q1s - 1.5 * IQRs)
        return self 
    
    def transform(self, X:np.array, y = None ):
        X[:,self.featureindices]=np.where(X[:,self.featureindices]>self.top,self.top,X[:,self.featureindices])
        X[:,self.featureindices]=np.where(X[:,self.featureindices]<self.bottom,self.bottom,X[:,self.featureindices])
        return X