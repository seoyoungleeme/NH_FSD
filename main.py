import pandas as pd
import datetime as dt
from risk_engine import Transaction, UserProfileStats, RiskEngine

def main():
    """
    메인 실행 함수
    1. 데이터 로드 및 전처리
    2. 사용자별 통계 프로필 생성
    3. 특정 거래에 대한 위험도 분석 및 출력
    """
    
    # --- 1) 데이터 로드 및 전처리 ---
    try:
        df = pd.read_csv('transaction_data.csv') 
    except FileNotFoundError:
        print("오류: 'transaction_data.csv' 파일을 찾을 수 없습니다.")
        return

    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
    print("1단계: 데이터 로드 및 전처리 완료.")
    print("-" * 30)

    # --- 2) 사용자 프로필 생성 ---
    user_profiles = {}
    for user_id in df['AccountID'].unique():
        user_txs = df[df['AccountID'] == user_id].sort_values(by='TransactionDate')

        if not user_txs.empty:
            mean_amount = user_txs['TransactionAmount'].mean()
            std_amount = user_txs['TransactionAmount'].std()

            if pd.isna(std_amount) or std_amount == 0:
                std_amount = mean_amount * 0.1 if mean_amount > 0 else 1.0
            
            typical_hours = user_txs['TransactionDate'].dt.hour.value_counts().nlargest(5).index.tolist()
            typical_ips = set(user_txs['IP Address'].value_counts().nlargest(5).index)
            typical_devices = set(user_txs['DeviceID'].value_counts().nlargest(5).index)
            typical_counterparties = set(user_txs['MerchantID'].value_counts().nlargest(5).index)

            user_profiles[user_id] = UserProfileStats(
                mean_amount=mean_amount,
                std_amount=std_amount,
                typical_hours=typical_hours,
                typical_ips=typical_ips,
                typical_devices=typical_devices,
                typical_counterparties=typical_counterparties,
                small_amount_threshold=200.0,
                repetition_window_minutes=30,
                repetition_count_threshold=3
            )
    
    print(f"2단계: 총 {len(user_profiles)}명의 사용자 프로필 생성 완료.")
    print("-" * 30)

    # --- 3) 특정 거래 위험도 평가 ---
    engine = RiskEngine()
    
    # 분석할 거래의 인덱스 저장
    sample_index = 150 
    if sample_index >= len(df):
        print(f"오류: 샘플 인덱스 {sample_index}가 데이터셋 범위를 벗어났습니다.")
        return
        
    current_tx_row = df.iloc[sample_index]
    user_id = current_tx_row['AccountID']
    
    print(f"3단계: 사용자 '{user_id}'의 {sample_index}번째 거래 위험도 분석 시작...")
    print("-" * 30)

    tx_to_score = Transaction(
        user_id=user_id,
        amount=current_tx_row['TransactionAmount'],
        timestamp=current_tx_row['TransactionDate'],
        ip=current_tx_row['IP Address'],
        device_id=current_tx_row['DeviceID'],
        counterparty_id=current_tx_row['MerchantID'],
        purpose_text="온라인 쇼핑몰 결제", # 임의 값
        is_new_account_opening=False,
        is_new_loan_application=False
    )

    user_stats = user_profiles.get(user_id)
    
    if not user_stats:
        print(f"경고: 사용자 {user_id}의 프로필을 찾을 수 없습니다.")
        return

    recent_txs_df = df[(df['AccountID'] == user_id) & (df['TransactionDate'] < tx_to_score.timestamp)]
    
    recent_txs = []
    for _, row in recent_txs_df.tail(20).iterrows():
        recent_txs.append(Transaction(
            user_id=row['AccountID'],
            amount=row['TransactionAmount'],
            timestamp=row['TransactionDate'],
            ip=row['IP Address'],
            device_id=row['DeviceID'],
            counterparty_id=row['MerchantID'],
            purpose_text=""
        ))

    risk_result = engine.score_transaction(
        tx=tx_to_score,
        stats=user_stats,
        recent_txs=recent_txs
    )
    
    # --- 최종 결과 출력 ---
    print("분석 완료! 최종 결과:")
    print("=" * 30)
    print(risk_result.summary)
    print("=" * 30)


if __name__ == "__main__":
    main()