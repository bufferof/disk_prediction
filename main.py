import os
import sys
from colorama import Fore, Back
import numpy as np
import pandas as pd

from datetime import timedelta
from sklearn.linear_model import LinearRegression

print("디스크 사이즈 입력 : ",end='')
TOTAL_DISK_SIZE = int(input())

print("디스크 사용량 로그 파일(.CSV) 위치 입력 : ",end='')
CSV_FILE_PATH = input()

if not os.path.exists(CSV_FILE_PATH) or os.path.splitext(CSV_FILE_PATH)[1] != ".csv":
    print(Fore.RED+f"File Path {CSV_FILE_PATH} is not found or is not csv file")
    sys.exit(0)

def predict():
    df = pd.read_csv(CSV_FILE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 첫 측정일로부터 경과일
    X = (
        df["date"] - df["date"].min()
    ).dt.total_seconds().to_numpy().reshape(-1,1) / 86400

    #디스크 사용량
    y = df["used_gb"].to_numpy()

    prediction_model = LinearRegression()
    prediction_model.fit(X,y)

    #하루 증가량의 평균 계산
    daily_growing = prediction_model.coef_[0]

    current_usage = y[-1]
    current_date = df["date"].iloc[-1]


    print(Fore.BLUE + "============= [INFO] =============")
    print(Fore.WHITE + f"전체 용량 : {TOTAL_DISK_SIZE:.2f} GB")
    print(f"현재 사용량 : {current_usage:.2f} GB")
    print(f"하루 증가량 : {daily_growing:.2f} GB")

    if current_usage >= TOTAL_DISK_SIZE:
        print(Fore.RED + "디스크가 이미 가득 찼습니다.")
        return

    if daily_growing <= 0:
        print(Fore.RED + "디스크 사용량이 증가하는 추세가 아닙니다.")
        return

    # 회귀선 기준으로 고갈 시점 예측
    deadline = (
        TOTAL_DISK_SIZE - prediction_model.intercept_
    ) / daily_growing

    last_day = X[-1][0]
    remaining_day = max(0,deadline-last_day)

    deadline = current_date + timedelta(days=remaining_day)

    print(Fore.GREEN + "============= [PREDICTION] =============")
    print(Fore.WHITE + f"예상 잔여 기간: {remaining_day:.1f}일")
    print(f"예상 고갈 날짜: {deadline:%Y-%m-%d}")
    print(f"모델 신뢰도: {prediction_model.score(X, y):.4f}")
    print(Fore.WHITE)

if __name__ == "__main__":
    predict()