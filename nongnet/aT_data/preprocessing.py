# data preprocessing
# written by C.J.H - AIFactory Team DS
# 실행하시면 train/test raw data를 제공된 set의 형태로 변환해줍니다
# 형태 가공 시에 참고 바랍니다


# 모듈 가져오기
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from pandasql import sqldf
import os


# 경고 끄기
warnings.filterwarnings(action='ignore')



# 전처리 예시 Class
class preprocessing_data(object):

    """
    도매, 소매, 수입수출, 도매경락, 주산지 데이터 전처리용 class
    중간결과물 저장 check parameter을 통해 지정, 중간결과물 저장 없이 사용은 check = 0
    """

    def __init__(self,dir):
        """
        전체 데이터에서 해당하는 domae,imexport,pummok,somae,weather 별 분리
        """
        self.data_list = glob(dir)
        self.domae = []
        self.imexport = []
        self.pummok = []
        self.somae = []
        self.weather = []


        for i in self.data_list:
            if 'domae' in i:
                self.domae.append(i)
            if 'imexport' in i:
                self.imexport.append(i)
            if 'pummok' in i.split('\\')[1]:
                self.pummok.append(i)
            if 'somae' in i:
                self.somae.append(i)
            if 'weather' in i:
                self.weather.append(i)


    def add_pummock(self,check=0):

        """
        check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        pummock의 데이터를 가져와 '해당일자_전체거래물량', '하위가격 평균가', '상위가격 평균가', '하위가격 거래물량', '상위가격 거래물량' 의 파생변수를 생성하는 단계
        """

        for num in tqdm(self.pummok):
            ddf = pd.read_csv(num)  # pummock의 csv 읽어오기
            name = num.split('\\')[1] # 전체 정제한 데이터를 담을 변수 이름

            sep2 = sqldf(f"select *, sum(거래량) as '해당일자_전체거래물량(kg)' from ddf group by datadate")
            # sql 문법을 이용해 '해당일자_전체거래물량' 계산

            height_set = []
            low_set = []
            height_volume_set = []
            low_volume_set = []

            for i in sep2['datadate']:

                """
                sep2는 group by를 통해 각 일자가 합쳐진 상태 예를 들어 '201703' 이 5개 이렇게 있을때 sep2는 group 시켜서 '해당일자_전체거래물량'을 계산
                이후 sep2 기준 20170101 and 20220630 사이의 날짜들에 해당하는 각 '201703' 마다 '해당일자_전체평균가격' 보다 큰지 아니면 작은지 판단
                위 과정을 통해 '하위가격 평균가', '상위가격 평균가', '하위가격 거래물량', '상위가격 거래물량' 변수 생성
                """

                new_list = ddf.loc[[d for d, x in enumerate(ddf['datadate']) if x == i]]
                set_price = sep2.loc[list(sep2['datadate']).index(i)]['해당일자_전체평균가격(원)']

                sum_he_as = sum(new_list['거래대금(원)'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z >= set_price)
                sum_he_vo = sum(new_list['거래량'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z >= set_price)

                sum_lo_as = sum(new_list['거래대금(원)'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z < set_price)
                sum_lo_vo = sum(new_list['거래량'].iloc[n] for n, z in enumerate(new_list['단가(원)']) if z < set_price)

                if sum_lo_vo != 0:
                    low_set.append(sum_lo_as / sum_lo_vo)
                    low_volume_set.append(sum_lo_vo)
                else:
                    low_set.append(np.nan)
                    low_volume_set.append(np.nan)

                if sum_he_vo != 0:
                    height_set.append(sum_he_as / sum_he_vo)
                    height_volume_set.append(sum_he_vo)
                else:
                    height_set.append(np.nan)
                    height_volume_set.append(np.nan)

            sep2['하위가격 평균가(원)'] = low_set
            sep2['상위가격 평균가(원)'] = height_set

            sep2['하위가격 거래물량(kg)'] = low_volume_set
            sep2['상위가격 거래물량(kg)'] = height_volume_set


            globals()[f'df_{name.split("_")[1].split(".")[0]}'] = sep2.copy()


            # 중간 산출물 저장
            if check != 0:
                if os.path.exists(f'./data') == False:
                    os.mkdir(f'./data')

                if os.path.exists(f'./data/품목') == False:
                    os.mkdir(f'./data/품목')

                sep2.to_csv(f'./data/품목/{name}', index=False)




    def add_dosomae(self, option=1, check=0):

        """
        check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        domae, somae 데이터를 가져와서 정제하는 단계
        option parameter을 통한 도매, 소매 선택
        """

        if option == 1:
            df = self.domae
            text = '도매'
        else:
            df = self.somae
            text = '소매'

        for i in tqdm(df):
            test = pd.read_csv(i)
            name = i.split('\\')[1]

            sep = test.loc[(test['등급명'] == '상품') | (test['등급명'] == 'S과')]  # 모든 상품에 대해서 수행하지 않고 GRAD_NM이 '상품', 'S과' 만 해당하는 품목 가져옴
            sep = sep[['datadate', '등급명', '조사단위(kg)', '가격(원)']]

            sep.rename(columns={"가격(원)": "가격"}, inplace=True)

            sep2 = sqldf(
                f"select datadate, max(가격) as '일자별_{text}가격_최대(원)', avg(가격) as '일자별_{text}가격_평균(원)', min(가격) as '일자별_{text}가격_최소(원)' from sep group by datadate")

            globals()[f'df_{name.split("_")[1].split(".")[0]}'] = globals()[f'df_{name.split("_")[1].split(".")[0]}'].merge(sep2, how='left')

            # 중간 산출물 저장
            if check != 0:
                if os.path.exists(f'./data') == False:
                    os.mkdir(f'./data')

                if os.path.exists(f'./data/{text}') == False:
                    os.mkdir(f'./data/{text}')

                sep2.to_csv(f'./data/{text}/{name}', index=False)

    def add_imexport(self,check=0):
        """
        check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        imexport 데이터 관련 정제, imexport 데이터는 월별 수입수출 데이터임으로 해당 월에 같은 값을 넣어주고 없는 것에는 np.nan
        해당 품목에 대한 imexport 데이터가 없는 경우 np.nan으로 대체, 모든 품목의 데이터가 동일한 컬럼수를 가지기 위해 수행
        """

        imex_cd = [i.split('_')[-1].split('.')[0] for i in self.imexport]

        for i in tqdm(range(len(self.pummok))):

            cd_number = self.pummok[i].split('_')[-1].split('.')[0]
            file_name = 'imexport_' + self.pummok[i].split('pummok_')[1]


            if cd_number in imex_cd:
                test4 = pd.read_csv(self.imexport[imex_cd.index(cd_number)])

                new_exim1 = []
                new_exim2 = []
                new_exim3 = []
                new_exim4 = []
                new_exim5 = []

                for j in globals()[f'df_{cd_number}']['datadate']:
                    target = j//100

                    try:
                        number = list(test4['datadate']).index(target)


                        new_exim1.append(test4['수출중량(kg)'].iloc[number])
                        new_exim2.append(test4['수출금액(달러)'].iloc[number])
                        new_exim3.append(test4['수입중량(kg)'].iloc[number])
                        new_exim4.append(test4['수입금액(달러)'].iloc[number])
                        new_exim5.append(test4['무역수지(달러)'].iloc[number])
                    except:
                        new_exim1.append(np.nan)
                        new_exim2.append(np.nan)
                        new_exim3.append(np.nan)
                        new_exim4.append(np.nan)
                        new_exim5.append(np.nan)

                df2 = pd.DataFrame()
                df2['수출중량(kg)'] = new_exim1
                df2['수출금액(달러)'] = new_exim2
                df2['수입중량(kg)'] = new_exim3
                df2['수입금액(달러)'] = new_exim4
                df2['무역수지(달러)'] = new_exim5

                globals()[f'df_{cd_number}'] = pd.concat([globals()[f'df_{cd_number}'], df2],axis=1)

            else:
                df2 = pd.DataFrame()
                df2['수출중량(kg)'] = np.nan
                df2['수출금액(달러)'] = np.nan
                df2['수입중량(kg)'] = np.nan
                df2['수입금액(달러)'] = np.nan
                df2['무역수지(달러)'] = np.nan

                globals()[f'df_{cd_number}'] = pd.concat([globals()[f'df_{cd_number}'], df2], axis=1)


            if check != 0:
                if os.path.exists(f'./data') == False:
                    os.mkdir(f'./data')

                if os.path.exists(f'./data/수출입') == False:
                    os.mkdir(f'./data/수출입')

                df2.to_csv(f'./data/수출입/{file_name}', index=False)

    def add_weather(self, check=0):

        """
        check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        weather 품목별 주산지 데이터를 가져와 합치는 함수, 일부 품목의 주산지가 3개가 아닌 것에 대해서는 np.nan 값으로 합쳐줌
        """

        for i in tqdm(self.pummok):
            name = i.split('_')[-1].split('.')[0]
            check_file = [j for j in self.weather if j.split('_')[-2] == name]


            df = pd.DataFrame()
            for d, j in enumerate(check_file):
                weather_df = pd.read_csv(j)
                new_exim1, new_exim2, new_exim3, new_exim4, new_exim5, new_exim6 = [], [], [], [], [], []


                for k in globals()[f'df_{name}']['datadate']:
                    try:
                        number = list(weather_df['datadate']).index(k)

                        new_exim1.append(weather_df['초기온도(℃)'].iloc[number])
                        new_exim2.append(weather_df['최대온도(℃)'].iloc[number])
                        new_exim3.append(weather_df['최저온도(℃)'].iloc[number])
                        new_exim4.append(weather_df['평균온도(℃)'].iloc[number])
                        new_exim5.append(weather_df['강수량(ml)'].iloc[number])
                        new_exim6.append(weather_df['습도(%)'].iloc[number])
                    except:
                        new_exim1.append(np.nan)
                        new_exim2.append(np.nan)
                        new_exim3.append(np.nan)
                        new_exim4.append(np.nan)
                        new_exim5.append(np.nan)
                        new_exim6.append(np.nan)


                df[f'주산지_{d}_초기온도(℃)'] = new_exim1
                df[f'주산지_{d}_최대온도(℃)'] = new_exim2
                df[f'주산지_{d}_최저온도(℃)'] = new_exim3
                df[f'주산지_{d}_평균온도(℃)'] = new_exim4
                df[f'주산지_{d}_강수량(ml)'] = new_exim5
                df[f'주산지_{d}_습도(%)'] = new_exim6

            if len(check_file) < 3:
                df[f'주산지_2_초기온도(℃)'] = np.nan
                df[f'주산지_2_최대온도(℃)'] = np.nan
                df[f'주산지_2_최저온도(℃)'] = np.nan
                df[f'주산지_2_평균온도(℃)'] = np.nan
                df[f'주산지_2_강수량(ml)'] = np.nan
                df[f'주산지_2_습도(%)'] = np.nan

            globals()[f'df_{name}'] = pd.concat([globals()[f'df_{name}'], df], axis=1)

            if check !=0:
                if os.path.exists(f'./data') == False:
                    os.mkdir(f'./data')

                if os.path.exists(f'./data/주산지') == False:
                    os.mkdir(f'./data/주산지')

                df.to_csv(f'./data/주산지/weather_{name}.csv', index=False)

    def add_categorical(self, out_dir, data_type="train", check=0):

        """
        check = 중간 산출물을 저장하고 싶다면 check 을 0 이외의 숫자로
        일자별 정보를 넣어주는 함수, 월별, 상순, 하순, 중순 을 원핫 인코딩을 통해 데이터로 넣어주는 함수
        모델이 각 행마다의 정보에서 몇월인지 상순인지 하순인지 파악하며 훈련시키기 위한 변수
        """

        for i in tqdm(self.pummok):
            name = i.split('_')[-1].split('.')[0]

            day_set = []
            month_set = []

            for k in globals()[f'df_{name}']['datadate']:
                day = k % 100
                month = k % 10000 // 100

                if day <= 10:
                    day_set.append('초순')
                elif (day > 10) and (day <= 20):
                    day_set.append('중순')
                else:
                    day_set.append('하순')

                month_set.append(f'{month}월')

            globals()[f'df_{name}']['일자구분'] = day_set
            globals()[f'df_{name}']['월구분'] = month_set

            globals()[f'df_{name}'] = pd.get_dummies(globals()[f'df_{name}'], columns=['일자구분', '월구분'])

            if check !=0:
                if os.path.exists(f'./data') == False:
                    os.mkdir(f'./data')

                if data_type != "train":
                    if os.path.exists(f'./data/{data_type}') == False:
                        os.mkdir("./data/{data_type}")
                    if os.path.exists(f'./data/{data_type}/{out_dir}') == False:
                        os.mkdir(f'./data/{data_type}/{out_dir}')
                    globals()[f'df_{name}'].to_csv(f'./data/{data_type}/{out_dir}/{data_type}_{name}.csv', index=False)
                else:
                    if os.path.exists(f'./data/{out_dir}') == False:
                        os.mkdir(f'./data/{out_dir}')
                    globals()[f'df_{name}'].to_csv(f'./data/{out_dir}/{data_type}_{name}.csv', index=False)






# 데이터 전처리 및 저장 (중간저장 X, 최종저장 O) - train
data = preprocessing_data('./aT_train_raw/*.csv')
data.add_pummock()
data.add_dosomae()
data.add_dosomae(option=2)
data.add_imexport()
data.add_weather()
data.add_categorical('train', data_type="train" ,check=1)


# 검증 데이터셋 전처리 및 저장 (중간저장 X, 최종저장 O) - test
for i in range(10):
    data = preprocessing_data(f'./aT_test_raw/sep_{i}/*.csv')
    data.add_pummock()
    data.add_dosomae()
    data.add_dosomae(option=2)
    data.add_imexport()
    data.add_weather()
    data.add_categorical(f'set_{i}', data_type="test", check=1)