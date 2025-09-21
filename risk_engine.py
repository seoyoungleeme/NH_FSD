from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import statistics
import datetime as dt

# 텍스트 모델 고도화 시도
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# 데이터 구조
@dataclass
class Transaction:
    user_id: str
    amount: float
    timestamp: dt.datetime
    ip: str
    device_id: str
    counterparty_id: str
    purpose_text: str  # 거래 목적 입력
    is_new_account_opening: bool = False
    is_new_loan_application: bool = False


@dataclass
class UserProfileStats:
    # 사용자 평소 패턴 통계(롤링 30~90일 등): 평균 금액, 표준편차, 주 사용 시간대, 주 사용 지역/ASN 등
    mean_amount: float
    std_amount: float
    typical_hours: List[int]              
    typical_counterparties: set          
    typical_devices: set
    typical_ips: set
    small_amount_threshold: float = 20000.0  
    repetition_window_minutes: int = 30     
    repetition_count_threshold: int = 3      


@dataclass
class RiskWeights:
    # 가중치 부여
    rule_ip_device: float = 0.25
    rule_time: float = 0.10
    rule_new_account_loan: float = 0.15
    rule_small_repetition: float = 0.15
    anomaly_amount: float = 0.15
    anomaly_velocity: float = 0.10
    nlp_text: float = 0.10

    def normalize(self) -> "RiskWeights":
        s = sum(self.__dict__.values())
        if s == 0:
            return self
        scaled = {k: v / s for k, v in self.__dict__.items()}
        return RiskWeights(**scaled)


@dataclass
class RiskFactorResult:
    # 요인별 점수과 근거
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class RiskResult:
    total_score: float                  
    summary: str                        
    breakdown: Dict[str, RiskFactorResult]  
    triggered_rules: List[str]        
    timestamp: dt.datetime



# 리스크 유틸 함수
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def z_score(x: float, mean: float, std: float, cap: float = 5.0) -> float:
    if std <= 1e-9:
        return 0.0
    z = abs((x - mean) / std)
    return min(z, cap)

def hour_of(ts: dt.datetime) -> int:
    return ts.hour


# NLP: 키워드 베이스라인 + 선택적 학습
PHISHING_KEYWORDS = [
    "대출", "급전", "본인인증", "보증금", "세금", "환불", "상품권", "코인", "투자",
    "수수료", "검찰", "경찰", "계좌동결", "미납", "납부", "지연", "압류", "해제", "해킹",
    "인증번호", "링크", "절차", "원금", "재단", "후원", "기부", "지원금"
]

class TextRiskModel:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.clf: Optional[LogisticRegression] = None

    def keyword_score(self, text: str) -> float:
        # 키워드 존재 비율 기반 스코어
        if not text:
            return 0.0
        tokens = set(text.lower())
        hits = 0
        for kw in PHISHING_KEYWORDS:
            if kw in text:
                hits += 1
        # 단순히 개수 기반 스케일
        return clamp01(hits / 5.0)

    def predict_proba(self, text: str) -> float:
        # 모델 확률 or 키워드 스코어 반영
        if self.vectorizer and self.clf:
            X = self.vectorizer.transform([text])
            p = self.clf.predict_proba(X)[0, 1]
            return float(p)
        return self.keyword_score(text)

    def train_text_model(self, texts: List[str], labels: List[int]) -> None:
        # 라벨이 모이면 간단 학습 시도
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn이 필요합니다. pip install scikit-learn")
        self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, labels)


# 메인 엔진
class RiskEngine:
    def __init__(
        self,
        weights: RiskWeights = RiskWeights().normalize(),
        text_model: Optional[TextRiskModel] = None,
        anomaly_model: Optional[IsolationForest] = None
    ):
        self.w = weights.normalize()
        self.text_model = text_model or TextRiskModel()
        self.anomaly_model = anomaly_model  


    # 룰 기반 점수
    def rule_ip_device(self, tx: Transaction, stats: UserProfileStats) -> RiskFactorResult:
        reasons = []
        score = 0.0

        if tx.ip not in stats.typical_ips:
            reasons.append(f"평소와 다른 IP 사용: {tx.ip}")
            score += 0.20  

        if tx.device_id not in stats.typical_devices:
            reasons.append(f"평소와 다른 디바이스 사용: {tx.device_id}")
            score += 0.15

        if tx.counterparty_id not in stats.typical_counterparties:
            reasons.append(f"처음 거래하는 상대: {tx.counterparty_id}")
            score += 0.10

        # 스케일 고정(0~1) -> 과도 누적 방지
        score = clamp01(score)
        return RiskFactorResult(score=score, reasons=reasons)

    def rule_time(self, tx: Transaction, stats: UserProfileStats) -> RiskFactorResult:
        h = hour_of(tx.timestamp)
        if h not in stats.typical_hours:
            return RiskFactorResult(score=0.30, reasons=[f"이례적 시간대 거래: {h}시"])
        return RiskFactorResult(score=0.0, reasons=[])

    def rule_new_account_loan(self, tx: Transaction) -> RiskFactorResult:
        reasons = []
        score = 0.0
        if tx.is_new_account_opening:
            reasons.append("이상신호: 신규 계좌 개설")
            score += 0.30
        if tx.is_new_loan_application:
            reasons.append("이상신호: 신규 대출 신청")
            score += 0.30
        return RiskFactorResult(score=clamp01(score), reasons=reasons)

    def rule_small_repetition(
        self,
        tx_window: List[Transaction],
        stats: UserProfileStats
    ) -> RiskFactorResult:
        # tx_window: 최근 시간순 거래 리스트, 윈도우는 호출 측에서 보장
        # 소액 반복 송금 탐지
        smalls = [t for t in tx_window if t.amount <= stats.small_amount_threshold]
        # N회 이상 시 트리거
        if len(smalls) >= stats.repetition_count_threshold:
            return RiskFactorResult(
                score=0.35,
                reasons=[f"소액 반복 송금 감지: {len(smalls)}회/{stats.repetition_window_minutes}분"]
            )
        return RiskFactorResult(score=0.0, reasons=[])

    # 이상치/통계 기반 점수
    def anomaly_amount(self, tx: Transaction, stats: UserProfileStats) -> RiskFactorResult:
        # 금액의 z-score 기반 0~1 점수
        z = z_score(tx.amount, stats.mean_amount, stats.std_amount, cap=5.0)
        # z=0 → 0점, z>=5 → 1점
        score = clamp01(z / 5.0)
        reasons = []
        if score > 0:
            reasons.append(f"금액 이상치(z={z:.2f}, mean={stats.mean_amount:.0f}, std={stats.std_amount:.0f})")
        return RiskFactorResult(score=score, reasons=reasons)

    def anomaly_velocity(self, recent_amounts: List[float]) -> RiskFactorResult:
        # 단기간 내 급격한 송금 빈도/금액 변동
        # 지표: 최근 N개 금액의 변동성. 표준편차가 일정 이상이면 리스크 가중시킴
        if len(recent_amounts) < 4:
            return RiskFactorResult(score=0.0, reasons=[])
        stdv = statistics.pstdev(recent_amounts)
        median = statistics.median(recent_amounts)
        if median <= 0:
            return RiskFactorResult(score=0.0, reasons=[])
        vol = stdv / (median + 1e-6)
        # 임의 매핑
        if vol <= 0.5:
            score = 0.2
        elif vol <= 1.0:
            score = 0.5
        elif vol <= 2.0:
            score = 0.8
        else:
            score = 1.0
        return RiskFactorResult(score=score, reasons=[f"단기 변동성 증가(vol={vol:.2f})"])

    # NLP 점수
    def nlp_text(self, text: str) -> RiskFactorResult:
        p = self.text_model.predict_proba(text or "")
        reasons = []
        if p > 0:
            reasons.append(f"목적문 위험 단서 확률 p={p:.2f}")
        return RiskFactorResult(score=p, reasons=reasons)

    # 최종 스코어 계산
    def score_transaction(
        self,
        tx: Transaction,
        stats: UserProfileStats,
        recent_txs: List[Transaction],  
    ) -> RiskResult:
        # 소액 반복 탐지
        if recent_txs:
            latest_ts = tx.timestamp
            window = [t for t in recent_txs if (latest_ts - t.timestamp).total_seconds() <= stats.repetition_window_minutes * 60]
            window.append(tx)
        else:
            window = [tx]

        recent_amounts = [t.amount for t in recent_txs[-8:]] + [tx.amount]

        f_ipdev = self.rule_ip_device(tx, stats)
        f_time = self.rule_time(tx, stats)
        f_new = self.rule_new_account_loan(tx)
        f_smallrep = self.rule_small_repetition(window, stats)
        f_anom_amount = self.anomaly_amount(tx, stats)
        f_anom_vel = self.anomaly_velocity(recent_amounts)
        f_nlp = self.nlp_text(tx.purpose_text)

        breakdown = {
            "rule_ip_device": f_ipdev,
            "rule_time": f_time,
            "rule_new_account_loan": f_new,
            "rule_small_repetition": f_smallrep,
            "anomaly_amount": f_anom_amount,
            "anomaly_velocity": f_anom_vel,
            "nlp_text": f_nlp,
        }

        # 가중합
        weighted = (
            self.w.rule_ip_device * f_ipdev.score +
            self.w.rule_time * f_time.score +
            self.w.rule_new_account_loan * f_new.score +
            self.w.rule_small_repetition * f_smallrep.score +
            self.w.anomaly_amount * f_anom_amount.score +
            self.w.anomaly_velocity * f_anom_vel.score +
            self.w.nlp_text * f_nlp.score
        )
        total = round(clamp01(weighted) * 100.0, 2)

        # 트리거된 룰 라벨 & 요약 생성
        triggered = []
        for k, v in breakdown.items():
            if v.score > 0 and v.reasons:
                triggered.append(k)

        summary = self._build_summary(tx, breakdown, total)

        return RiskResult(
            total_score=total,
            summary=summary,
            breakdown=breakdown,
            triggered_rules=triggered,
            timestamp=dt.datetime.utcnow()
        )

    def _build_summary(self, tx: Transaction, br: Dict[str, RiskFactorResult], total: float) -> str:
        bullets: List[str] = []
        for k in [
            "rule_new_account_loan",
            "rule_ip_device",
            "rule_small_repetition",
            "anomaly_amount",
            "anomaly_velocity",
            "rule_time",
            "nlp_text",
        ]:
            rf = br[k]
            for r in rf.reasons:
                bullets.append(f"- {r}")

        reason_text = "\n".join(bullets) if bullets else "- 특이 사유 없음"

        return (
            f"[위험도 {total:.2f}/100]\n"
            f"거래 금액: {tx.amount:.0f}, 시간: {tx.timestamp.isoformat()}, 상대: {tx.counterparty_id}\n"
            f"주요 사유:\n{reason_text}"
        )