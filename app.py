from flask import Flask, render_template
from bs4 import BeautifulSoup
import urllib.request as req


def recommend():

    import pandas as pd

    df_init = pd.read_csv('C:\\Users\\waryo\\OneDrive\\바탕 화면\\핀테크_인턴쉽\\핀테크_인턴쉽\\데이터전처리_2\\대출(대부)_init.csv', encoding="utf-8")

    user_bth = 1964  # 유저 년생 입력
    user_bth_str = str(user_bth)
    user_bth_list = list(user_bth_str)
    user_bth_list.pop()
    user_bth_list.append('0')
    user_bth_str = "".join(user_bth_list)
    user_bth = int(user_bth_str)
    print(user_bth)

    user_rate = 0  # 유저 이자 입력
    if user_rate != 0:
        user_rate = user_rate * 1000
    print(user_rate)

    user_AMT = '2,500,000'  # 유저 대출금액 입력
    user_AMT_list = list(user_AMT)
    while ',' in user_AMT_list:
        user_AMT_list.remove(',')
    user_AMT_str = "".join(user_AMT_list)
    user_AMT = int(user_AMT_str)
    user_AMT = int(user_AMT / 1000)
    print(user_AMT)

    # 조건에 따른 축출
    df_new = df_init[(df_init['BTH_YR'] < user_bth + 10) & (df_init['BTH_YR'] >= user_bth) &  # 1997년 => 1990년대
                     (df_init['RATE'] <= user_rate + (user_rate * 0.5)) & (
                                 df_init['RATE'] >= user_rate - (user_rate * 0.5)) &  # 이자 6000 => +50%, -50% 사이 이자
                     (df_init['LN_AMT'] <= user_AMT + (user_AMT * 0.2)) & (
                                 df_init['LN_AMT'] >= user_AMT - (user_AMT * 0.2))]  # 대출금액 5000 => +20%, -20% 사이의 금액

    df_new = df_new.reset_index()

    SCTR_CD_list = list(df_new['SCTR_CD'])
    LN_CD_1_list = list(df_new['LN_CD_1'])
    LN_CD_2_list = list(df_new['LN_CD_2'])
    LN_CD_3_list = list(df_new['LN_CD_3'])

    df_ready = []
    for i in range(0, len(df_new), 1):
        df_ready.append(
            str(SCTR_CD_list[i]) + '/' + str(LN_CD_1_list[i]) + '/' + str(LN_CD_2_list[i]) + '/' + str(LN_CD_3_list[i]))

    df_ready3 = pd.DataFrame(data=df_ready, columns=['total'])
    df_ready4 = df_new[['JOIN_SN']]

    df = pd.concat([df_ready4, df_ready3], axis=1)

    df["JOIN_SN"] = df["JOIN_SN"].apply(str)
    join_sn_df = list(df['JOIN_SN'])

    join_sn_df_result = []
    for value in join_sn_df:
        if value not in join_sn_df_result:
            join_sn_df_result.append(value)

    df["JOIN_SN"] = df["JOIN_SN"].apply(str)

    df_count = {}
    for i in range(0, len(join_sn_df_result), 1):
        df_yr = df[df['JOIN_SN'].str.contains(str(join_sn_df_result[i]), na=False)]
        df_count[join_sn_df_result[i]] = df_yr['total'].value_counts().head(5)

    df_count2 = {}
    for i in range(0, len(join_sn_df_result), 1):
        new_list = []
        for data in df_count[join_sn_df_result[i]].index:
            new_list.append(data)
        df_count2[join_sn_df_result[i]] = new_list

    total = list(df['total'])

    total_result = []
    for value in total:
        if value not in total_result:
            total_result.append(value)

    df_count3 = {}
    for key in df_count2.keys():
        df_count2_score = {}
        count = 5
        for j in range(0, len(df_count2[key]), 1):
            df_count2_score[df_count2[key][j]] = count
            count = count - 1
        df_count3[key] = df_count2_score

    result_list_2 = []
    for i in range(0, len(total_result), 1):
        result_list_1 = []
        for j in range(0, len(join_sn_df_result), 1):
            # print("1.",total_result[i],end="")
            # print("=>",join_sn_df_result[j], ":",df_count3[join_sn_df_result[j]],end="")
            if total_result[i] in df_count3[join_sn_df_result[j]]:
                # print("=>",df_count3[join_sn_df_result[j]][total_result[i]],)
                result_list_1.append(df_count3[join_sn_df_result[j]][total_result[i]])
                # print("=>",result_list_1)
            else:
                # print("=>","0")
                result_list_1.append(0)
                # print("=>",result_list_1)
        result_list_2.append(result_list_1)

    import numpy as np
    result_array = np.array(result_list_2)

    result_array_T = result_array.T

    final_df = pd.DataFrame(result_array_T, columns=total_result, index=join_sn_df_result)

    # 유저에게 상품을 추천하기 위해 행 추가
    user_input = []
    for i in range(0, len(final_df.columns), 1):
        user_input.append(0)

    new_row = pd.DataFrame([user_input], columns=final_df.columns)

    new_final_df = pd.concat([final_df.iloc[:0], new_row, final_df.iloc[0:]], ignore_index=True)

    SGD_array = np.array(new_final_df)

    import numpy as np
    from tqdm import tqdm_notebook as tqdm

    import numpy as np

    # Base code : https://yamalab.tistory.com/92
    class MatrixFactorization():
        def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
            """
            :param R: rating matrix
            :param k: latent parameter
            :param learning_rate: alpha on weight update
            :param reg_param: beta on weight update
            :param epochs: training epochs
            :param verbose: print status
            """

            self._R = R
            self._num_users, self._num_items = R.shape
            self._k = k
            self._learning_rate = learning_rate
            self._reg_param = reg_param
            self._epochs = epochs
            self._verbose = verbose

        def fit(self):
            """
            training Matrix Factorization : Update matrix latent weight and bias

            참고: self._b에 대한 설명
            - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
            - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

            :return: training_process
            """

            # init latent features
            self._P = np.random.normal(size=(self._num_users, self._k))
            self._Q = np.random.normal(size=(self._num_items, self._k))

            # init biases
            self._b_P = np.zeros(self._num_users)
            self._b_Q = np.zeros(self._num_items)
            self._b = np.mean(self._R[np.where(self._R != 0)])

            # train while epochs
            self._training_process = []
            for epoch in range(self._epochs):
                # rating이 존재하는 index를 기준으로 training
                xi, yi = self._R.nonzero()
                for i, j in zip(xi, yi):
                    self.gradient_descent(i, j, self._R[i, j])
                cost = self.cost()
                self._training_process.append((epoch, cost))

                # print status
                if self._verbose == True and ((epoch + 1) % 10 == 0):
                    print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))

        def cost(self):
            """
            compute root mean square error
            :return: rmse cost
            """

            # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
            # 참고: http://codepractice.tistory.com/90
            xi, yi = self._R.nonzero()

            # predicted = self.get_complete_matrix()
            cost = 0
            for x, y in zip(xi, yi):
                cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
            return np.sqrt(cost / len(xi))

        def gradient(self, error, i, j):
            """
            gradient of latent feature for GD

            :param error: rating - prediction error
            :param i: user index
            :param j: item index
            :return: gradient of latent feature tuple
            """

            dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
            dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
            return dp, dq

        def gradient_descent(self, i, j, rating):
            """
            graident descent function

            :param i: user index of matrix
            :param j: item index of matrix
            :param rating: rating of (i,j)
            """

            # get error
            prediction = self.get_prediction(i, j)
            error = rating - prediction

            # update biases
            self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
            self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

            # update latent feature
            dp, dq = self.gradient(error, i, j)
            self._P[i, :] += self._learning_rate * dp
            self._Q[j, :] += self._learning_rate * dq

        def get_prediction(self, i, j):
            """
            get predicted rating: user_i, item_j
            :return: prediction of r_ij
            """
            return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

        def get_complete_matrix(self):
            """
            computer complete matrix PXQ + P.bias + Q.bias + global bias

            - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
            - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
            - b를 더하는 것은 각 element마다 bias를 더해주는 것

            - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

            :return: complete matrix R^
            """
            return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)

    # run example
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000, debug=False)
        # rating matrix - User X Item : (7 X 5)
        R = SGD_array = np.array(new_final_df)

    ##########################################3
    factorizer = MatrixFactorization(R, k=20, learning_rate=0.01, reg_param=0.01, epochs=350, verbose=True)
    factorizer.fit()

    good_array = factorizer.get_complete_matrix()

    good_list = list(good_array[0])
    good_list2 = good_list

    good_list2.sort(reverse=True)
    print(good_list2)

    good_list2 = good_list2[0:5]

    good_list3 = []
    for i in range(0, len(good_list2), 1):
        good_list3.append(good_list.index(good_list2[i]))

    final_good = []
    for i in range(0, len(good_list3), 1):
        final_good.append(total_result[good_list3[i]])

    final_result = []
    for i in range(0, len(final_good), 1):
        final_result.append(final_good[i].split('/'))

    df_output = []
    for i in range(0, len(final_result), 1):
        df_output.append(df_new[(df_new['SCTR_CD'] == int(final_result[i][0])) &
                                (df_new['LN_CD_1'] == int(final_result[i][1])) &
                                (df_new['LN_CD_2'] == int(final_result[i][2])) &
                                (df_new['LN_CD_3'] == int(final_result[i][3]))])

    df_output3 = []
    for i in range(0, len(final_result), 1):
        df_output2 = df_output[i][['COM_SN', 'SCTR_CD', 'LN_CD_1', 'LN_CD_2', 'LN_CD_3', 'RATE']]
        df_output3.append(df_output2.values.tolist())

    df_output4 = []
    for i in range(0, len(df_output3), 1):
        my_list = df_output3[i]
        new_df_output3_list = []
        for v in my_list:
            if v not in new_df_output3_list:
                new_df_output3_list.append(v)
        df_output4.append(new_df_output3_list)

    SCTR_CD_dic = {1: '국내은행',
                   2: '외국은행',
                   3: '신용협동기구',
                   5: '신용카드사',
                   6: '손해보험사',
                   8: '생명보험사',
                   10: '투신사',
                   12: '기타',
                   13: '신기술사·창투사·벤쳐캐피탈',
                   15: '증권사·종금사',
                   16: '리스사',
                   17: '할부금융사',
                   21: '상호저축은행',
                   24: '대부업권'}

    LN_CD_1_dic = {31: '개인대출', 37: '장기카드대출(카드론)', 41: '단기카드대출(현금서비스)'}

    LN_CD_2_dic = {0: '카드대출',
                   100: '신용대출 > 신용대출',
                   150: '신용대출 > 학자금대출',
                   170: '신용대출 > 전세자금대출',
                   200: '담보대출 > 예적금담보대출',
                   210: '담보대출 > 유가증권담보대출',
                   220: '담보대출 > 주택담보대출',
                   230: '담보대출 > 주택외부동산(토지,상가등)담보대출',
                   240: '담보대출 > 지급보증(보증서)담보대출',
                   245: '담보대출 > 보금자리론',
                   250: '담보대출 > 학자금지급보증대출',
                   260: '담보대출 > 주택연금대출',
                   270: '담보대출 > 전세자금(보증서, 질권 등)대출',
                   271: '담보대출 > 전세보증금담보대출',
                   290: '담보대출 > 기타담보대출',
                   400: '보험계약대출거래사실',
                   500: '할부금융 > 신차할부',
                   510: '할부금융 > 중고차할부',
                   590: '할부금융 > 기타할부',
                   700: '리스 > 금융리스',
                   710: '리스 > 운용리스'}

    LN_CD_3_dic = {0: '(서민금융아님)',
                   100: '새희망홀씨',
                   150: '햇살론15',
                   170: '햇살론17',
                   180: '햇살론youth',
                   190: '햇살론뱅크',
                   200: '햇살론',
                   300: '바꿔드림론',
                   350: '안전망대출',
                   360: '안전망대출Ⅱ',
                   900: '기타'}

    for i in range(0, len(df_output4), 1):
        for j in range(0, len(df_output4[i]), 1):
            df_output4[i][j][1] = SCTR_CD_dic[df_output4[i][j][1]]
            df_output4[i][j][2] = LN_CD_1_dic[df_output4[i][j][2]]
            df_output4[i][j][3] = LN_CD_2_dic[df_output4[i][j][3]]
            df_output4[i][j][4] = LN_CD_3_dic[df_output4[i][j][4]]
            df_output4[i][j][5] = round(df_output4[i][j][5] * 0.001, 1)

    return df_output4


app = Flask(__name__)


@app.route('/community.html')
def community():
    return render_template('community..html')

@app.route('/goal.html')
def goal():
    return render_template('goal.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/indexcopy.html')
def indexcopy():
    return render_template('indexcopy.html')

@app.route('/information.html')
def information():
    return render_template('information.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/main.html')
def main():
    return render_template('main.html')

@app.route('/management.html')
def management():
    return render_template('management.html', datas=recommend())

@app.route('/map.html')
def map():
    return render_template('map.html')

@app.route('/map3.html')
def map3():
    return render_template('map3.html')

@app.route('/maps.html')
def maps():
    return render_template('maps.html')

@app.route('/maptest.html')
def maptest():
    return render_template('maptest.html')

@app.route('/mpa2.html')
def mpa2():
    return render_template('mpa2.html')


if __name__ == "__main__":
    app.run()