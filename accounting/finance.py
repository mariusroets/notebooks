import math
import numpy as np
import pandas as pd

def _fv(amount, i, n):
    return amount*((1+i)**n)
def _pv(amount, i, n):
    return amount/((1+i)**n)
def _periods(fv, pv, i):
    return math.log((pv/fv), 1+i)

def time_value(fv=None, pv=None, i=None, n=None):
    if fv is None:
        return fv(pv, i, n)
    if pv is None:
        return pv(fv, i, n)
    if n is None:
        return periods(fv, pv, i)

def annuity(pv=None, i=None, cashflow=None, n=None, fv=0):
    if pv is None:
        return cashflow*(1/i - 1/(i*((1+i)**n)))
    if not cashflow:
        return pv/(1/i - 1/(i*((1+i)**n)))
    
def annuity_detail(i, cashflow, n, fv=0):
    avalue = np.round(annuity(i=i, cashflow=cashflow, n=n))
    end_pv = np.round(_pv(fv, i, n))
    df = pd.DataFrame({'cashflow': np.full(n+1, cashflow), 'annuity': avalue, 'pv_end': end_pv, 'value': avalue, 'capital': 0, 'interest': 0})
    for key, row in df.iterrows():
        if key:
            df.loc[key, 'interest'] = np.round(df.loc[key-1, 'annuity']*i)
            df.loc[key, 'capital'] = cashflow - df.loc[key, 'interest']
            df.loc[key, 'annuity'] = df.loc[key-1, 'annuity'] - df.loc[key, 'capital']
        else:
            df.loc[key, 'cashflow'] = 0
        df.loc[key, 'value'] = df.loc[key, 'annuity'] + df.loc[key, 'pv_end']
    return df

def ap_turnover(cogs=None, acc_payable=None, acc_payable_prev=None, days=False):
    ratio = cogs/((acc_payable + acc_payable_prev)/2)
    if days:
        return 365/ratio
    return ratio

class Bond:
    """Do calculations on a bond

    Keyword Arguments:
        face_value (TODO): TODO
        coupon (TODO): TODO
        maturity (TODO): TODO
        compounding (TODO): TODO
    """

    def __init__(self, face_value=1, coupon=0.1, maturity=1, compounding=1):
        self._face_value = face_value
        self._coupon = coupon
        self._maturity = maturity
        self._compounding = compounding
       
    @property
    def face_value(self):
        """The face value of the bond"""
        return self._face_value

    @property
    def coupon(self):
        """The coupon rate of the bond"""
        return self._coupon

    @property
    def maturity(self):
        """The years to maturity of the bond"""
        return self._maturity

    @property
    def compounding(self):
        """Number of compounding periods per year"""
        return self._compounding

    def value(self, i):
        """Gives the present value of the bond at market rate i

        Args:
            i (float): The market interest rate at which to calculate the bond value

        Returns: 
            float: The present value of the bond
        """
        return annuity(cashflow=self.face_value*(self.coupon/self.compounding),
                       i=i,
                       n=self.maturity*self.compounding,
                       fv=self.face_value)

    def detail(self, i):
        """Give detail valuation for each compounding period over the lifetime of the bond

        Args:
            i (float): The market interest rate at which to calculate the bond value

        Returns:
            pd.DataFrame: The detail valuation for each compounding period

        """
        
