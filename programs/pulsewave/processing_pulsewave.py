"""
脈波に対する基本的な処理

20200821 Kaito Iuchi
"""


""" 標準ライブラリのインポート """
import glob
import os
import sys
import time
import numba
import math
import re
""" サードバーティライブラリのインポート """
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.sparse import eye
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy.sparse import csc_matrix, lil_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn


#===============================
#データ補間
#===============================
def interpolate_nan(data):
    """
    nanを近傍データで補間 (1次元 or 2次元)

    Parameters
    ---------------
    data : 1D [データ長] or 2D [データ数, データ長] / データ長軸で補間
        補間対象データ

    Returns
    ---------------
    data_inpl : 1D or 2D
        nanを補間したデータ

    """
    
    x = np.arange(data.shape[0])
    nan_indx = np.isnan(data)
    not_nan_indx = np.isfinite(data)
    
    # 始端，終端に1つもしくは連続してnanがあれば，左右近傍ではなく，片側近傍で補間
    count_nan_lft = 0
    count = 0
    while True:
        if not_nan_indx[count] == False:
            count_nan_lft += 1
            not_nan_indx[count] = True
        else:
            break
        count += 1
        
    count_nan_rgt = 0
    count = 1
    while True:
        if not_nan_indx[-count] == False:
            count_nan_rgt += 1
            not_nan_indx[-count] = True
        else:
            break
        count += 1
    
    if count_nan_lft > 0:
        data[:count_nan_lft] = np.nanmean(data[count_nan_lft+1:count_nan_lft+5])
    if count_nan_rgt > 0:
        data[-count_nan_rgt:] = np.nanmean(data[-count_nan_rgt-5:-count_nan_rgt-1])
    
    func_inpl = interp1d(x[not_nan_indx], data[not_nan_indx])
    data[nan_indx] = func_inpl(x[nan_indx])
        
    return data

def upsample_data(data_original, sampling_rate_upsampled):
    """
    元データを，指定したサンプリングレートにアップサンプリングする（3次スプライン補間を使用）．

    Parameters
    ----------
    data_original: np.array (1 dim)
        元データ
    sampling_rate_upsampled: int
        アップサンプリング後のサンプリングレート

    Returns
    -------
    data_upsampled: np.array (1 dim)
        アップサンプリング後のデータ
    """

    # 元のデータ点に対応する時間値を計算
    sampling_rate_original = len(data_original)
    time_original = np.linspace(0, 1, sampling_rate_original)

    # 3次スプライン補間関数の作成
    spline_interpolator = interp1d(time_original, data_original, kind="cubic")

    # アップサンプリング後のデータ点の生成
    time_upsampled = np.linspace(0, 1, sampling_rate_upsampled)
    data_upsampled = spline_interpolator(time_upsampled)

    return data_upsampled

def resample_timestamp(time_stmp, data, resample_rate, data_num=0, kind='cubic', time_range=0, start_time=0):
    """
    タイムスタンプデータを等間隔でリサンプリングする．

    Parameters
    ---------------
    time_stmp : np.ndarray (1dim [データ数])
        タイムスタンプ
    data : np.ndarray (1dim [データ数])
        時系列データ
    sample_rate : int
        リサンプルレート
    data_num : int
        データの個数

    Returns
    ---------------
    data_new : nd.ndarray (1dim [フレーム数])
        リサンプリングしたデータ

    """

    time_stmp_isstr = type(time_stmp[0]) == str 
    
    global test1
    test1 = time_stmp
    
    # タイムスタンプが数値の場合
    if time_stmp_isstr == False:

        hoge, uni_indx = np.unique(time_stmp, return_index=True)
        time_stmp = time_stmp[uni_indx]
        data = data[uni_indx]

        start_time = time_stmp[0]
        end_time = time_stmp[-1]
        time_lng = end_time - start_time
        if data_num == 0:
            x_new = np.linspace(start_time, end_time, int(resample_rate * time_lng))
        else:
            x_new = np.linspace(start_time, end_time, data_num)

        spln = interp1d(time_stmp, data, kind=kind)
        data_new = spln(x_new)
    
    # タイムスタンプが文字列(h:m:s.s)の場合
    else:  
        data_num = time_stmp.shape[0]
        time_stmp2 = np.zeros([0])
        for num in range(data_num):
            time_tmp = re.findall(r'\d+', time_stmp[num])
            time_tmp = np.array(time_tmp, dtype=np.float)
            
            time = time_tmp[0] * 60 * 60 + time_tmp[1] * 60 + time_tmp[2] + time_tmp[3] * 0.001 # h:m:s.s
            
            time_stmp2 = np.concatenate([time_stmp2, [time]])
                
        if isinstance(time_range, list):
            time_strt = int(time_range[0])
            time_lng = int(time_range[1])
        else:
            time_strt = int(time_stmp2[0])
            time_lng = int(time_stmp2[-1])

        hoge, uni_indx = np.unique(time_stmp2, return_index=True)
        time_stmp2 = time_stmp2[uni_indx]
        data = data[uni_indx]
    
        x_new = np.linspace(time_strt + start_time, time_lng, int(resample_rate * (time_lng - time_strt - start_time)))
        spln = interp1d(time_stmp2, data, kind=kind, fill_value='extrapolate')
        data_new = spln(x_new)
    
    return data_new
    

def interpolate_outlier(data, flag_intplt, th_constant=3):
    """
    時系列データの異常値の検出と置換を行う．

    Parameters
    ---------------
    data : np.ndarray (1dim / [データ長])
        時系列データ
    flag_intplt : bool
        補間するかnp.nanを置換するかのフラグ / True: 補間, False: np.nan
    th_constant : np.int
        正常値と異常値の閾値を決めるための定数

    Returns
    ---------------
    data_new : np.ndarray (2dim / [データ長])
        異常値置換後の時系列データ
    indx : np.ndarray (1dim)
        異常値のインデックス
    
    """
    
    # 第1四分位数の算出
    q1 = np.percentile(data, 25)
    
    # 第3四分位数の算出
    q3 = np.percentile(data, 75)
    
    # InterQuartile Range
    iqr = q3 - q1
    
    # 異常値判定のための閾値設定
    th_lwr = q1 - iqr * th_constant
    th_upr = q3 + iqr * th_constant
    
    indx = (data < th_lwr) | (th_upr < data)
    data[indx] = np.nan
    
    if flag_intplt:
        data_new = interpolate_nan(data)
    else:
        data_new = data
    
    if np.sum(indx) > 0:
        print(' [i]%d' %(np.sum(indx)), end='')
    
    return data_new, indx
    

def polyfit_data(data, sample_rate, deg):
    """
    時系列データを多項式近似する．

    Parameters
    ---------------
    data : np.ndarray (2dim / [データ数, データ長])
        時系列データ
    sample_rate : int
        時系列データのサンプルレート

    Returns
    ---------------
    data_poly : np.ndarray (2dim / [データ数, データ長])

    """
    
    if data.ndim == 1:
        
        data_lng = data.shape[0]
        x = np.linspace(0, data_lng, data_lng)
        y = data
        notnan_indx = np.isfinite(y)
        res = np.polyfit(x[notnan_indx], y[notnan_indx], deg)
        data_poly = np.poly1d(res)(x)
    
    else:
    
        data_num = data.shape[0]
        data_lng = data.shape[1]
        
        data_poly = np.zeros([data_num, data_lng])
        x = np.linspace(0, data_lng, data_lng)
        for num in range(data_num):
            y = data[num, :]
            notnan_indx = np.isfinite(y)

            res = np.polyfit(x[notnan_indx], y[notnan_indx], deg)
            data_poly[num, :] = np.poly1d(res)(x)
    
    return data_poly



#=====================================
#　デトレンド処理・バンドパス処理
#=====================================
def detrend_pulse(pulse, sample_rate):
    """
    脈波をデトレンドする．
    脈波が短すぎるとエラーを出力．(T < wdth の場合)
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_dt : np.float (1 dim)
        デトレンドされた脈波
    
    """
    @ numba.jit
    def inv_jit(A):
        return np.linalg.inv(A)

    t1 = time.time()
    
    # デトレンドによりデータ終端は歪みが生じるため，1秒だけ切り捨てる．    
    pulse_length = pulse.shape[0]
    print(pulse_length)
    pulse = np.concatenate([pulse, pulse[-2 * sample_rate:]])
    virt_length = pulse_length + 2 * sample_rate
    # デトレンド処理 / An Advanced Detrending Method With Application to HRV Analysis
    pulse_dt = np.zeros(virt_length)
    order = len(str(virt_length))
    print(order)
    lmd = sample_rate * 12 # サンプルレートによって調節するハイパーパラメータ

    wdth = sample_rate * 2 # データ終端が歪むため，データを分割してデトレンドする場合，wdth分だけ終端を余分に使用する．

    if order > 4:
        splt = int(sample_rate / 16) # 40
        T = int(virt_length/splt)
        # wdth = T
        for num in range(splt):
            print('\r\t[Detrending Pulse] : %d / %d' %(num + 1, splt), end='')
            if num < (splt - 1):
                I = np.identity(T + wdth)
                flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
                D2 = spdiags(flt.T, np.array([0, 1, 2]), T + wdth - 2, T + wdth)
                preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
                inv_tmp = inv_jit(preinv)
                tmp = (I - inv_tmp) @ pulse[num * T : (num + 1) * T + wdth]
                tmp = tmp[0 : -wdth]
                pulse_dt[num * T : (num + 1) * T] = tmp
            else:
                # I = eye(T, T)
                I = np.identity(T)
                flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
                D2 = spdiags(flt.T, np.array([0, 1, 2]), T - 2, T)
                preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
                inv_tmp = inv_jit(preinv)
                tmp = (I - inv_tmp) @ pulse[num * T: (num + 1) * T]
                pulse_dt[num * T : (num + 1) * T] = tmp

    else:
        T =len(pulse)
        pulse_dt = np.zeros(len(pulse))
        I = np.identity(T)
        flt = np.ones([T + wdth - 2, 1]) * np.array([1, -2, 1])
        D2 = spdiags(flt.T, np.array([0, 1, 2]), T - 2, T)
        preinv = I + lmd ** 2 * np.conjugate(D2.T) * D2
        inv_tmp = inv_jit(preinv)
        pulse_dt[:] = (I - inv_tmp) @ pulse

    pulse_dt = pulse_dt[0 : len(pulse)-sample_rate*2]

    t2 = time.time()
    elapsed_time = int((t2 - t1) * 10)
    print(f'\tTime : {elapsed_time * 0.1} sec')
    
    return pulse_dt


def sg_filter_pulse(pulse, sample_rate):
    """
    SGフィルタリングにより脈波をデノイジングする．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_sg : np.float (1 dim)
        デノイジングされた脈波
    
    """ 
    
    # SGフィルタリング
    pulse_sg = signal.savgol_filter(pulse, int(sample_rate / 2.0) + 1, 5)
    
    return pulse_sg


def bandpass_filter_pulse(pulse, band_width, sample_rate):
    """
    バンドパスフィルタリングにより脈波をデノイジングする．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    band_width : float (1dim / 2cmps)
        通過帯 [Hz] (e.g. [0.75, 4.0])
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_sg : np.float (1 dim)
        デノイジングされた脈波
    
    """ 
    
    # バンドパスフィルタリング
    nyq = 0.5 * sample_rate
    b, a = signal.butter(1, [band_width[0]/nyq, band_width[1]/nyq], btype='band')
    pulse_bp = signal.filtfilt(b, a, pulse)
    
    return pulse_bp

#=====================================
#その他計算
#=====================================

def calculate_snr(pulse, sample_rate, output_folder):
    """
    脈波データからSN比を算出する．

    Parameters
    ---------------
    pulse : np.ndarray (1dim)
        脈波データ
    sample_rate : int
        脈波のサンプルレート
    output_folder : string
        出力フォルダ

    Returns
    ---------------
    snr : np.float
        SN比

    """
    
    sn = pulse.shape[0]
    psd = np.fft.fft(pulse)
    psd = np.abs(psd/sn/2)
    frq = np.fft.fftfreq(sn, d=1/sample_rate)
    
    sn_hlf = math.ceil(sn/2)
    psd_hlf = psd[:sn_hlf]
    frq_hlf = frq[:sn_hlf]
    frq_dff = frq_hlf[1] - frq_hlf[0]

    np.savetxt(output_folder + '/PSD.csv', psd_hlf)

    peak1_indx = psd_hlf.argsort()[-1]
    peak2_indx = psd_hlf.argsort()[-2]
    
    if peak2_indx < peak1_indx:
        peak1_indx = peak2_indx

    wd = int(0.3 / frq_dff)
    
    harmonic_1st = np.arange(peak1_indx - wd, peak1_indx + wd+1)
    harmonic_2nd = np.arange(peak1_indx * 2 - wd, peak1_indx * 2 + wd+1)
    harmonic_3rd = np.arange(peak1_indx * 3 - wd, peak1_indx * 3 + wd+1)
    sgnl = np.sum(psd_hlf[harmonic_1st]) * frq_dff + np.sum(psd_hlf[harmonic_2nd]) * frq_dff + np.sum(psd_hlf[harmonic_3rd]) * frq_dff
    ttl = np.sum(psd_hlf[:]) * frq_dff
    nois = ttl - sgnl
    
    snr = 10 * np.log10(sgnl / nois)

    return snr


def calculate_hrv(pulse, sample_rate):
    """
    脈波データから心拍変動(IBI，心拍数，PSD)を算出する．

    Parameters
    ---------------
    pulse : np.ndarray (1dim)
        脈波データ
    sample_rate : int
        脈波のサンプルレート

    Returns
    ---------------
    ibi : nd.ndarray (1dim [100[fps] × 合計時間])
        IBI
    pulse_rate : np.float
        心拍数
    frq : np.ndarray (1dim)
        周波数軸
    psd : np.ndarray (1dim)
        パワースペクトル密度

    """
    
    # ピーク検出
    peak1_indx, peak2_indx = detect_pulse_peak(pulse, sample_rate)
    
    # 下側ピーク数 / 心拍変動は下側ピークで算出した方が精度が良い．
    peak_num = peak2_indx.shape[0]
    ibi = np.zeros([peak_num - 1])
    flag = np.zeros([peak_num - 1])
    
    # IBI算出
    for num in range(peak_num - 1):
        ibi[num] = (peak2_indx[num + 1] - peak2_indx[num]) / sample_rate
        
    # ibiが[0.25, 1.5][sec]の範囲内に無い場合，エラーとする．/ [0.5, 1.5]
    error_indx = np.where((ibi < 0.33) | (1.5 < ibi))
    flag[error_indx] = False
    ibi[error_indx] = np.nan

    global count_flag
    if np.any(flag):
        print('[!]')
        count_flag += 1

    ibi_num = ibi.shape[0]
    # スプライン補間は，次数以上の要素数が必要
    if ibi_num > 3:
        spln_kind = 'cubic'
    elif ibi_num > 2:
        spln_kind = 'quadratic'
    elif ibi_num > 1:
        spln_kind = 'slinear'
    else:
        ibi = np.nan

    total_time = np.sum(ibi)
    if np.isnan(total_time) != True:
        # エラーが発生した箇所の補間
        ibi = interpolate_nan(ibi)
        total_time = np.sum(ibi)
        # 心拍数の算出
        pulse_rate = 60 / np.mean(ibi)
        # スプライン補間 / 1fpsにリサンプリング
        # リサンプリングレート
        fs = 2
        sample_num = int(total_time * fs)
        x = np.linspace(0.0, total_time, ibi_num)
        f_spln = interp1d(x, ibi, kind=spln_kind)
        x_new = np.linspace(0.0, int(total_time), sample_num)
        ibi_spln = f_spln(x_new)

        sn = ibi_spln.shape[0]
        psd = np.fft.fft(ibi_spln)
        psd = np.abs(psd)
        frq = np.fft.fftfreq(n=sn, d=1/fs)
        sn_hlf = math.ceil(sample_num / 2)
        psd = psd[:sn_hlf]
        frq = frq[:sn_hlf]

    else:
        ibi = np.nan
        pulse_rate = np.nan
        frq = np.nan
        psd = np.nan

    return ibi, pulse_rate, frq, psd


def scndiff_pulse(one_pulse, resample_rate):
    """
    1脈動分の脈波を2皆微分する．

    Parameters
    ---------------
    pulse : np.float (1 dim)
        1脈動分の脈波
    resample_rate : int
        スプライン補間でリサンプリングする際のサンプルレート

    Returns
    ---------------
    pulse_nrml : np.float
        正規化した1脈動分の脈波
    peak_indx : int
        脈波データの上側ピークのインデックス
    flag : int
        使えるデータか使えないデータかを格納する． / True : 使用可能, False : 使用不可
    
    """
    
    one_pulse_length = one_pulse.shape[0]
    
    # 500フレームにリサンプリング
    x = np.linspace(0.0, 1.0, one_pulse_length)
    f_spln = interp1d(x, one_pulse, kind='cubic')
    x_new = np.linspace(0.0, 1.0, resample_rate)
    one_pulse = f_spln(x_new)

    # デトレンド処理    
    x = [0, resample_rate - 1]
    y = [one_pulse[0], one_pulse[-1]]
    
    res = np.polyfit(x, y, 1)
    x2 = np.arange(0, resample_rate)
    poly = np.poly1d(res)(x2)
    
    frst_drv = np.diff(one_pulse) * resample_rate
    scnd_drv = np.diff(frst_drv) * resample_rate
    
    scnd_drv = polyfit_data(scnd_drv, resample_rate, 15)

    flag = True
    return frst_drv, scnd_drv, flag


def normalize_pulse(one_pulse, resample_rate, upper_pulse):
    """
    1脈動分の脈波を正規化する．

    Parameters
    ---------------
    pulse : np.float (1 dim)
        1脈動分の脈波
    resample_rate : int
        スプライン補間でリサンプリングする際のサンプルレート
    upper_pulse : boolean
        True : 上向き脈波 / False : 下向き脈波

    Returns
    ---------------
    pulse_nrml : np.float
        正規化した1脈動分の脈波
    peak_indx : int
        脈波データの上側ピークのインデックス
    flag : int
        使えるデータか使えないデータかを格納する． / True : 使用可能, False : 使用不可
    
    """
    
    one_pulse_length = one_pulse.shape[0]
    flag = True
    
    # 500フレームにリサンプリング
    x = np.linspace(0.0, 1.0, one_pulse_length)
    f_spln = interp1d(x, one_pulse, kind='cubic')
    x_new = np.linspace(0.0, 1.0, resample_rate)
    one_pulse = f_spln(x_new)

    # デトレンド処理    
    x = [0, resample_rate - 1]
    y = [one_pulse[0], one_pulse[-1]]
    
    res = np.polyfit(x, y, 1)
    x2 = np.arange(0, resample_rate)
    poly = np.poly1d(res)(x2)
    
    pulse_nrml = np.abs(one_pulse - poly)

    # マイナス値を0にする．
    pulse_nrml[pulse_nrml < 0] = 0
    
    # ピーク検出
    peak_indx = np.argmax(pulse_nrml)
    
    # ピークが時間的に遅い場合はエラーだと見なす． / 下向き脈波の場合は早すぎる場合
    if upper_pulse == True:
        if peak_indx > int(resample_rate * 0.6):
            flag = False
    else:
        if peak_indx < int(resample_rate * 0.4):
            flag = False
    
    pulse_nrml = pulse_nrml / pulse_nrml[peak_indx]   

    # 下向き脈波の場合は波形を反転する．
    if upper_pulse == False:
        pulse_nrml = -1 * pulse_nrml + pulse_nrml[peak_indx]

    return pulse_nrml, peak_indx, flag


def detect_pulse_peak(pulse, sample_rate):
    """
    脈波の上側ピークと下側ピークを検出する．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    peak1_indx : int (1 dim)
        脈波の上側ピーク
    peak2_indx : int (1 dim)
        脈波の下側ピーク
    
    """ 
    
    # ピーク検出
    peak1_indx = signal.argrelmax(pulse, order=int(sample_rate / 3.0))[0]
    peak2_indx = signal.argrelmin(pulse, order=int(sample_rate / 3.0))[0]
    
    return peak1_indx, peak2_indx


def reverse_pulse(pulse):
    """
    脈波の振幅を逆転させる．
    
    Parameters
    ---------------
    pulse : ndarray (1dim)
        脈波データ

    Returns
    ---------------
    pulse : ndarray (1dim)
        振幅を逆転させた脈波データ
    
    """ 
    
    # iPPGは脈波の振幅が逆になっているからその修正
    pulse = pulse * -1
    pulse = pulse - np.min(pulse)
    
    return pulse


def obtain_envelope(data, sample_rate, order_dntr=2.5):
    """
    時系列データから包絡線を取得する．

    Parameters
    ---------------
    data : np.ndarray (1dim)
        時系列データ
    sample_rate : int
        時系列データのサンプルレート

    Returns
    ---------------
    envlp_upr : nd.ndarray (1dim)
        時系列データの上側包絡線
    envlp_lwr : nd.ndarray (1dim)
        時系列データの下側包絡線

    """
    
    data_lngth = data.shape[0]

    # 血圧波形は2.5が良さげ
    peak1_indx = signal.argrelmax(data, order=int(sample_rate / order_dntr))[0]
    peak2_indx = signal.argrelmin(data, order=int(sample_rate / order_dntr))[0]
    
    # 異常値を近傍で補間
    if peak1_indx.size > 1:
        peak1, outlier_indx = interpolate_outlier(data[peak1_indx], True, th_constant=100)
    else:
        peak1 = np.array([np.nan])
    if peak2_indx.size > 1:
        peak2, outlier_indx = interpolate_outlier(data[peak2_indx], True, th_constant=100)
    else:
        peak2 = np.array([np.nan])

    peak1_num = peak1.shape[0]
    # スプライン補間は，次数以上の要素数が必要
    if peak1_num > 3:
        spln_kind1 = 'cubic'
    elif peak1_num > 2:
        spln_kind1 = 'quadratic'
    elif peak1_num > 1:
        spln_kind1 = 'slinear'
    else:
        spln_kind1 = 0

    peak2_num = peak2.shape[0]
    # スプライン補間は，時数以上の要素数が必要
    if peak2_num > 3:
        spln_kind2 = 'cubic'
    elif peak2_num > 2:
        spln_kind2 = 'quadratic'
    elif peak2_num > 1:
        spln_kind2 = 'slinear'
    else:
        spln_kind2 = 0


    if spln_kind2 != 0:
        spln_upr = interp1d(peak1_indx, peak1, kind=spln_kind1, fill_value=np.nan, bounds_error=False)
        x_new_upr = np.arange(data_lngth)
        envlp_upr = spln_upr(x_new_upr)
        envlp_upr = interpolate_nan(envlp_upr)
    else:
        envlp_upr = np.nan


    if spln_kind2 != 0:
        spln_lwr = interp1d(peak2_indx, peak2, kind=spln_kind2, fill_value=np.nan, bounds_error=False)
        x_new_lwr = np.arange(data_lngth)
        envlp_lwr = spln_lwr(x_new_lwr)
        envlp_lwr = interpolate_nan(envlp_lwr)
    else:
        envlp_lwr = np.nan
    
    return envlp_upr, envlp_lwr


def get_nearest_value(array, query):
    """
    配列から検索値に最も近い値を検索し，そのインデックスを返却する．

    Parameters
    ---------------
    array : 
        検索対象配列
    query : float
        検索値

    Returns
    ---------------
    indx : int
        検索値に最も近い値が格納されているインデックス

    """
    
    indx = np.abs(np.asarray(array) - query).argmin()
    value = array[indx]
    
    return indx


def interpolate_nan(data):
    """
    nanを近傍データで補間 (1次元 or 2次元)

    Parameters
    ---------------
    data : 1D [データ長] or 2D [データ数, データ長] / データ長軸で補間
        補間対象データ

    Returns
    ---------------
    data_inpl : 1D or 2D
        nanを補間したデータ

    """
    
    x = np.arange(data.shape[0])
    nan_indx = np.isnan(data)
    not_nan_indx = np.isfinite(data)
    
    # 始端，終端に1つもしくは連続してnanがあれば，左右近傍ではなく，片側近傍で補間
    count_nan_lft = 0
    count = 0
    while True:
        if not_nan_indx[count] == False:
            count_nan_lft += 1
            not_nan_indx[count] = True
        else:
            break
        count += 1
        
    count_nan_rgt = 0
    count = 1
    while True:
        if not_nan_indx[-count] == False:
            count_nan_rgt += 1
            not_nan_indx[-count] = True
        else:
            break
        count += 1
    
    if count_nan_lft > 0:
        data[:count_nan_lft] = np.nanmean(data[count_nan_lft+1:count_nan_lft+5])
    if count_nan_rgt > 0:
        data[-count_nan_rgt:] = np.nanmean(data[-count_nan_rgt-5:-count_nan_rgt-1])
    
    func_inpl = interp1d(x[not_nan_indx], data[not_nan_indx])
    data[nan_indx] = func_inpl(x[nan_indx])
        
    return data


