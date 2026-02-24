"""
Proactive AI Friend - Memory System
短期記憶、長期記憶、思考リザーバーの管理
"""

import json
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from collections import deque

import config


@dataclass
class Message:
    """会話メッセージ"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    user_id: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Thought:
    """AIの内なる思考"""
    content: str
    motivation_score: float
    reasoning: str
    timestamp: str
    triggered_by: str  # 何がトリガーになったか
    expressed: bool = False  # 発言されたかどうか
    
    def to_dict(self):
        return asdict(self)


@dataclass
class LongTermMemory:
    """長期記憶エントリ"""
    user_id: str
    key: str  # 記憶のカテゴリ（例: "好きなもの", "仕事", "悩み"）
    content: str
    importance: float  # 重要度 1-5
    created_at: str
    last_accessed: str
    access_count: int = 1
    
    def to_dict(self):
        return asdict(self)


class MemoryManager:
    """
    記憶管理クラス
    - 短期記憶: 直近の会話履歴
    - 長期記憶: ユーザーについての重要な情報
    - 思考リザーバー: 保留中の思考
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # 短期記憶（会話履歴）
        self.short_term: deque[Message] = deque(
            maxlen=config.SHORT_TERM_MEMORY_SIZE
        )
        
        # 長期記憶
        self.long_term: list[LongTermMemory] = []
        
        # 思考リザーバー（保留中の思考）
        self.thought_reservoir: deque[Thought] = deque(
            maxlen=config.THOUGHT_RESERVOIR_SIZE
        )
        
        # 最後の発言時刻
        self.last_user_message_time: Optional[datetime] = None
        self.last_ai_message_time: Optional[datetime] = None
        
        # 連続AI発言カウント
        self.consecutive_ai_messages = 0
        
        # 永続化用のパス
        self.storage_path = f"memory_store/{user_id}"
        
        # 既存データの読み込み
        self._load_from_disk()
    
    # =========================================================================
    # 短期記憶操作
    # =========================================================================
    
    def add_message(self, role: str, content: str):
        """メッセージを短期記憶に追加"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            user_id=self.user_id
        )
        self.short_term.append(message)
        
        # 発言時刻の更新
        if role == "user":
            self.last_user_message_time = datetime.now()
            self.consecutive_ai_messages = 0
        else:
            self.last_ai_message_time = datetime.now()
            self.consecutive_ai_messages += 1
        
        return message
    
    def get_conversation_history(self, n: Optional[int] = None) -> list[dict]:
        """会話履歴を取得"""
        messages = list(self.short_term)
        if n:
            messages = messages[-n:]
        
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
    
    def get_context_summary(self) -> str:
        """会話の要約を取得（思考生成用）"""
        if not self.short_term:
            return "まだ会話が始まっていません。"
        
        recent = list(self.short_term)[-5:]
        summary_parts = []
        
        for msg in recent:
            role_name = "ユーザー" if msg.role == "user" else config.AI_NAME
            summary_parts.append(f"{role_name}: {msg.content[:200]}...")
        
        return "\n".join(summary_parts)
    
    # =========================================================================
    # 長期記憶操作
    # =========================================================================
    
    def add_long_term_memory(self, key: str, content: str, importance: float = 3.0):
        """長期記憶に追加"""
        # 同じキーの既存記憶を更新
        for mem in self.long_term:
            if mem.key == key and mem.user_id == self.user_id:
                mem.content = content
                mem.importance = importance
                mem.last_accessed = datetime.now().isoformat()
                mem.access_count += 1
                self._save_to_disk()
                return mem
        
        # 新規追加
        memory = LongTermMemory(
            user_id=self.user_id,
            key=key,
            content=content,
            importance=importance,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat()
        )
        self.long_term.append(memory)
        
        # サイズ制限
        if len(self.long_term) > config.LONG_TERM_MEMORY_SIZE:
            # 重要度とアクセス頻度が低いものを削除
            self.long_term.sort(
                key=lambda x: x.importance * x.access_count,
                reverse=True
            )
            self.long_term = self.long_term[:config.LONG_TERM_MEMORY_SIZE]
        
        self._save_to_disk()
        return memory
    
    def get_relevant_memories(self, query: str, top_k: int = 5) -> list[LongTermMemory]:
        """関連する長期記憶を取得（簡易的なキーワードマッチング）"""
        # TODO: 将来的にはembeddingベースの検索に置き換え
        
        scored_memories = []
        query_lower = query.lower()
        
        for mem in self.long_term:
            score = 0
            # キーワードマッチング
            if mem.key.lower() in query_lower:
                score += 3
            if any(word in query_lower for word in mem.content.lower().split()):
                score += 1
            # 重要度とアクセス頻度も考慮
            score += mem.importance * 0.5
            score += min(mem.access_count * 0.1, 1)
            
            if score > 0:
                scored_memories.append((score, mem))
        
        # スコア順でソート
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [mem for _, mem in scored_memories[:top_k]]
    
    def get_all_memories_summary(self) -> str:
        """すべての長期記憶の要約"""
        if not self.long_term:
            return "まだユーザーについての情報がありません。"
        
        summaries = []
        for mem in sorted(self.long_term, key=lambda x: x.importance, reverse=True):
            summaries.append(f"- {mem.key}: {mem.content}")
        
        return "\n".join(summaries[:10])
    
    # =========================================================================
    # 思考リザーバー操作
    # =========================================================================
    
    def add_thought(self, content: str, motivation_score: float, 
                    reasoning: str, triggered_by: str) -> Thought:
        """思考をリザーバーに追加"""
        thought = Thought(
            content=content,
            motivation_score=motivation_score,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            triggered_by=triggered_by
        )
        self.thought_reservoir.append(thought)
        return thought
    
    def get_pending_thoughts(self, min_score: float = 0) -> list[Thought]:
        """未発言の思考を取得"""
        return [
            t for t in self.thought_reservoir
            if not t.expressed and t.motivation_score >= min_score
        ]

    def get_expressed_thoughts(self) -> list[Thought]:
        """発言済みの思考を取得"""
        return [
            t for t in self.thought_reservoir
            if t.expressed
        ]
    
    def mark_thought_expressed(self, thought: Thought):
        """思考を発言済みにマーク"""
        thought.expressed = True
    
    def get_highest_motivation_thought(self) -> Optional[Thought]:
        """最も動機づけスコアが高い未発言の思考を取得"""
        pending = self.get_pending_thoughts()
        if not pending:
            return None
        return max(pending, key=lambda t: t.motivation_score)
    
    # =========================================================================
    # 状態チェック
    # =========================================================================
    
    def get_silence_duration(self) -> float:
        """ユーザーの沈黙時間（秒）"""
        if not self.last_user_message_time:
            return 0
        return (datetime.now() - self.last_user_message_time).total_seconds()
    
    def can_intervene(self) -> bool:
        """AIが自発的に発言できるかチェック"""
        # 連続発言制限
        if self.consecutive_ai_messages >= config.MAX_CONSECUTIVE_INTERVENTIONS:
            return False
        
        # 最小間隔チェック
        if self.last_ai_message_time:
            elapsed = (datetime.now() - self.last_ai_message_time).total_seconds()
            if elapsed < config.MIN_INTERVENTION_INTERVAL:
                return False
        
        return True
    
    # =========================================================================
    # 永続化
    # =========================================================================
    
    def _save_to_disk(self):
        """記憶をディスクに保存"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 長期記憶の保存
        long_term_data = [m.to_dict() for m in self.long_term]
        with open(f"{self.storage_path}/long_term.json", "w", encoding="utf-8") as f:
            json.dump(long_term_data, f, ensure_ascii=False, indent=2)
    
    def _load_from_disk(self):
        """記憶をディスクから読み込み"""
        long_term_path = f"{self.storage_path}/long_term.json"
        
        if os.path.exists(long_term_path):
            with open(long_term_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.long_term = [LongTermMemory(**m) for m in data]
