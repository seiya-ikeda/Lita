"""
Proactive AI Friend - Research Logger
研究用のデータ収集とログ管理
"""

import os
import json
import csv
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

import config


@dataclass
class ConversationLog:
    """会話ログエントリ"""
    timestamp: str
    session_id: str
    user_id: str
    event_type: str  # "user_message", "ai_response", "proactive_intervention"
    content: str
    metadata: dict


@dataclass
class ThoughtLog:
    """思考ログエントリ"""
    timestamp: str
    session_id: str
    user_id: str
    thought_content: str
    trigger_reason: str
    motivation_score: float
    evaluation_details: dict
    was_expressed: bool
    response_if_expressed: Optional[str]


@dataclass
class InteractionMetrics:
    """インタラクション指標"""
    session_id: str
    user_id: str
    start_time: str
    end_time: str
    total_turns: int
    user_messages: int
    ai_reactive_responses: int
    ai_proactive_interventions: int
    avg_user_response_time: float
    avg_ai_response_time: float
    intervention_acceptance_rate: float  # 介入後にユーザーが返答した割合


class ResearchLogger:
    """
    研究用ログ収集クラス
    
    収集するデータ:
    1. 全会話ログ
    2. 思考生成・評価ログ
    3. インタラクション指標
    4. セッション統計
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        
        # ログディレクトリ作成
        self.log_dir = config.LOG_DIRECTORY
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/conversations", exist_ok=True)
        os.makedirs(f"{self.log_dir}/thoughts", exist_ok=True)
        os.makedirs(f"{self.log_dir}/metrics", exist_ok=True)
        
        # セッション内のログ
        self.conversation_logs: list[ConversationLog] = []
        self.thought_logs: list[ThoughtLog] = []
        
        # 指標計算用
        self.user_message_times: list[datetime] = []
        self.ai_response_times: list[datetime] = []
        self.proactive_interventions: int = 0
        self.interventions_with_response: int = 0
        self.last_was_intervention: bool = False
    
    # =========================================================================
    # 会話ログ
    # =========================================================================
    
    def log_user_message(self, user_id: str, content: str, metadata: dict = None):
        """ユーザーメッセージをログ"""
        now = datetime.now()
        
        # 介入後の返答かチェック
        if self.last_was_intervention:
            self.interventions_with_response += 1
            self.last_was_intervention = False
        
        self.user_message_times.append(now)
        
        log = ConversationLog(
            timestamp=now.isoformat(),
            session_id=self.session_id,
            user_id=user_id,
            event_type="user_message",
            content=content,
            metadata=metadata or {}
        )
        self.conversation_logs.append(log)
        self._append_to_csv("conversations", log)
    
    def log_ai_response(self, user_id: str, content: str, 
                        is_proactive: bool, metadata: dict = None):
        """AI応答をログ"""
        now = datetime.now()
        self.ai_response_times.append(now)
        
        event_type = "proactive_intervention" if is_proactive else "ai_response"
        
        if is_proactive:
            self.proactive_interventions += 1
            self.last_was_intervention = True
        
        log = ConversationLog(
            timestamp=now.isoformat(),
            session_id=self.session_id,
            user_id=user_id,
            event_type=event_type,
            content=content,
            metadata=metadata or {}
        )
        self.conversation_logs.append(log)
        self._append_to_csv("conversations", log)
    
    # =========================================================================
    # 思考ログ
    # =========================================================================
    
    def log_thought(
        self,
        user_id: str,
        thought_content: str,
        trigger_reason: str,
        motivation_score: float,
        evaluation_details: dict,
        was_expressed: bool,
        response_if_expressed: Optional[str] = None
    ):
        """思考をログ"""
        if not config.LOG_THOUGHTS:
            return
        
        log = ThoughtLog(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            user_id=user_id,
            thought_content=thought_content,
            trigger_reason=trigger_reason,
            motivation_score=motivation_score,
            evaluation_details=evaluation_details,
            was_expressed=was_expressed,
            response_if_expressed=response_if_expressed
        )
        self.thought_logs.append(log)
        self._append_to_csv("thoughts", log)
    
    # =========================================================================
    # 指標計算
    # =========================================================================
    
    def calculate_metrics(self, user_id: str) -> InteractionMetrics:
        """セッションの指標を計算"""
        now = datetime.now()
        
        # 平均応答時間計算
        avg_user_time = 0
        avg_ai_time = 0
        
        if len(self.user_message_times) > 1:
            intervals = [
                (self.user_message_times[i+1] - self.user_message_times[i]).total_seconds()
                for i in range(len(self.user_message_times) - 1)
            ]
            avg_user_time = sum(intervals) / len(intervals) if intervals else 0
        
        # 介入受容率
        acceptance_rate = 0
        if self.proactive_interventions > 0:
            acceptance_rate = self.interventions_with_response / self.proactive_interventions
        
        # カウント
        user_msgs = sum(1 for log in self.conversation_logs if log.event_type == "user_message")
        reactive = sum(1 for log in self.conversation_logs if log.event_type == "ai_response")
        proactive = sum(1 for log in self.conversation_logs if log.event_type == "proactive_intervention")
        
        return InteractionMetrics(
            session_id=self.session_id,
            user_id=user_id,
            start_time=self.start_time.isoformat(),
            end_time=now.isoformat(),
            total_turns=len(self.conversation_logs),
            user_messages=user_msgs,
            ai_reactive_responses=reactive,
            ai_proactive_interventions=proactive,
            avg_user_response_time=avg_user_time,
            avg_ai_response_time=avg_ai_time,
            intervention_acceptance_rate=acceptance_rate
        )
    
    def save_session_metrics(self, user_id: str):
        """セッション終了時に指標を保存"""
        metrics = self.calculate_metrics(user_id)
        
        metrics_file = f"{self.log_dir}/metrics/sessions.csv"
        file_exists = os.path.exists(metrics_file)
        
        with open(metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(metrics).keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metrics))
    
    # =========================================================================
    # 統計サマリー
    # =========================================================================
    
    def get_thought_statistics(self) -> dict:
        """思考の統計を取得"""
        if not self.thought_logs:
            return {}
        
        scores = [log.motivation_score for log in self.thought_logs]
        expressed = sum(1 for log in self.thought_logs if log.was_expressed)
        
        return {
            "total_thoughts": len(self.thought_logs),
            "expressed_thoughts": expressed,
            "expression_rate": expressed / len(self.thought_logs),
            "avg_motivation_score": sum(scores) / len(scores),
            "max_motivation_score": max(scores),
            "min_motivation_score": min(scores),
            "triggers": self._count_triggers()
        }
    
    def _count_triggers(self) -> dict:
        """トリガー種別のカウント"""
        triggers = {}
        for log in self.thought_logs:
            trigger = log.trigger_reason.split()[0]  # 最初の単語
            triggers[trigger] = triggers.get(trigger, 0) + 1
        return triggers
    
    # =========================================================================
    # ファイル出力
    # =========================================================================
    
    def _append_to_csv(self, log_type: str, log_entry):
        """CSVにログを追記"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{self.log_dir}/{log_type}/{date_str}_{self.session_id}.csv"
        
        file_exists = os.path.exists(filename)
        data = asdict(log_entry)
        
        # dictフィールドをJSON文字列に変換
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = json.dumps(value, ensure_ascii=False)
        
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    
    def export_session_summary(self) -> dict:
        """セッションのサマリーをエクスポート"""
        return {
            "session_id": self.session_id,
            "experiment_condition": config.EXPERIMENT_CONDITION,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "config": {
                "motivation_threshold": config.MOTIVATION_THRESHOLD,
                "silence_timeout": config.SILENCE_TIMEOUT,
                "thought_generation_interval": config.THOUGHT_GENERATION_INTERVAL,
                "max_consecutive_interventions": config.MAX_CONSECUTIVE_INTERVENTIONS
            },
            "thought_statistics": self.get_thought_statistics(),
            "total_logs": len(self.conversation_logs)
        }
    
    def save_session_summary(self):
        """セッションサマリーをJSONで保存"""
        summary = self.export_session_summary()
        filename = f"{self.log_dir}/metrics/summary_{self.session_id}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
