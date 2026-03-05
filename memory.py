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


@dataclass
class NarrativeEntry:
    """Litaの自己認識エントリ"""
    id: str
    timestamp: str
    chapter: str   # "self" / "relationship" / "values" / "growth"
    content: str   # Litaの一人称で書かれた物語断片
    related_user: str
    contradicts: Optional[str] = None  # 矛盾する過去エントリのID

    def to_dict(self):
        return asdict(self)


@dataclass
class UserModelEntry:
    """ユーザーの行動パターンエントリ"""
    id: str
    timestamp: str
    dimension: str   # "thinking_style" / "communication" / "emotional" / "temporal"
    content: str     # 観測されたパターン
    confidence: float = 0.3   # 0.0-1.0（観測回数で上昇、矛盾で下降）
    observation_count: int = 1

    def to_dict(self):
        return asdict(self)


class UserModel:
    """
    ユーザーの行動パターンモデル（他者モデル）
    - ファクト（long_term）ではなく「この人らしさ」のパターンを管理
    - confidence が観測ごとにドリフトする
    """
    MAX_ENTRIES_PER_DIMENSION = 3

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.entries: list[UserModelEntry] = []
        self.storage_path = f"memory_store/{user_id}/user_model.json"
        self._load()

    def get_summary(self, min_confidence: float = 0.2) -> str:
        """プロンプトに渡すユーザーモデルサマリー"""
        filtered = [e for e in self.entries if e.confidence >= min_confidence]
        if not filtered:
            return "なし"
        by_dim: dict[str, list[UserModelEntry]] = {}
        for e in filtered:
            by_dim.setdefault(e.dimension, []).append(e)
        lines = []
        for dim, entries in by_dim.items():
            for entry in sorted(entries, key=lambda e: e.confidence, reverse=True)[:2]:
                conf_label = "高確信" if entry.confidence >= 0.7 else "観測中"
                lines.append(f"[{dim}|{conf_label}] {entry.content}")
        return "\n".join(lines)

    def add_or_update(self, dimension: str, content: str) -> UserModelEntry:
        """観測を追加。類似エントリがあれば confidence を上げる。"""
        for entry in self.entries:
            if entry.dimension == dimension and self._similar(entry.content, content):
                entry.confidence = min(1.0, entry.confidence + 0.1)
                entry.observation_count += 1
                entry.timestamp = datetime.now().isoformat()
                self._save()
                return entry

        new_entry = UserModelEntry(
            id=datetime.now().isoformat(),
            timestamp=datetime.now().isoformat(),
            dimension=dimension,
            content=content,
        )
        self.entries.append(new_entry)

        # 同一 dimension の最大エントリ数を超えたら confidence 最低のものを削除
        dim_entries = [e for e in self.entries if e.dimension == dimension]
        if len(dim_entries) > self.MAX_ENTRIES_PER_DIMENSION:
            lowest = min(dim_entries, key=lambda e: e.confidence)
            self.entries.remove(lowest)

        self._save()
        return new_entry

    def weaken(self, dimension: str, content: str, amount: float = 0.15):
        """矛盾する観測があった場合に confidence を下げる"""
        for entry in self.entries:
            if entry.dimension == dimension and self._similar(entry.content, content):
                entry.confidence = max(0.0, entry.confidence - amount)
                self._save()
                return

    def prune(self, min_confidence: float = 0.1):
        """低信頼エントリを削除"""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.confidence >= min_confidence]
        if len(self.entries) < before:
            self._save()

    def _similar(self, a: str, b: str) -> bool:
        """単語の重複で簡易的な類似判定"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        return len(words_a & words_b) >= 3

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = [UserModelEntry(**e) for e in data]


class SelfNarrative:
    """
    Litaの自己史
    - ユーザーをまたいだ、Lita自身の自己認識を管理
    - 会話を通じて更新され、週次で整理（睡眠）される
    """
    STORAGE_PATH = "memory_store/narrative.json"

    def __init__(self):
        self.entries: list[NarrativeEntry] = []
        self._load()

    def get_summary(self, max_entries: int = 8) -> str:
        """プロンプトに渡す自己認識サマリー"""
        if not self.entries:
            return "なし"
        recent = self.entries[-max_entries:]
        return "\n".join([f"[{e.chapter}] {e.content}" for e in recent])

    def add_entry(
        self,
        content: str,
        chapter: str,
        related_user: str,
        contradicts: Optional[str] = None
    ) -> NarrativeEntry:
        entry = NarrativeEntry(
            id=datetime.now().isoformat(),
            timestamp=datetime.now().isoformat(),
            chapter=chapter,
            content=content,
            related_user=related_user,
            contradicts=contradicts
        )
        self.entries.append(entry)
        if len(self.entries) > config.NARRATIVE_MAX_ENTRIES:
            self.entries = self.entries[-config.NARRATIVE_MAX_ENTRIES:]
        self._save()
        return entry

    def replace_all(self, entries: list[dict]):
        """週次整理後にエントリを置き換え"""
        now = datetime.now().isoformat()
        self.entries = [
            NarrativeEntry(
                id=f"consolidated-{i}-{now}",
                timestamp=now,
                chapter=e["chapter"],
                content=e["content"],
                related_user=e.get("related_user", "consolidated"),
                contradicts=None
            )
            for i, e in enumerate(entries)
        ]
        self._save()

    def _save(self):
        os.makedirs("memory_store", exist_ok=True)
        with open(self.STORAGE_PATH, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.STORAGE_PATH):
            with open(self.STORAGE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = [NarrativeEntry(**e) for e in data]


@dataclass
class InternalState:
    """Litaの内部状態（感情パラメータ）"""
    loneliness: float = 3.0       # 0-10 / 孤独感。時間経過で増加、良い会話で減少
    curiosity: float = 5.0        # 0-10 / 好奇心・話したい気持ち。面白い話で増加、共有後に減少
    social_energy: float = 7.0    # 0-10 / 社交エネルギー。長い会話で消耗、休息で回復
    last_updated: str = ""        # ISO timestamp（受動的ドリフト計算用）
    last_conversation: str = ""   # 最後の会話終了時刻

    def to_dict(self):
        return asdict(self)


class InternalStateManager:
    """
    Litaの内部状態を管理する
    - ユーザーごと：Litaがそのユーザーとの関係で感じる状態
    - 時間経過による受動的ドリフトと、会話後の能動的更新
    """

    def __init__(self, user_id: str):
        self.storage_path = f"memory_store/{user_id}/internal_state.json"
        self.state = InternalState(last_updated=datetime.now().isoformat())
        self._load()

    def apply_passive_drift(self) -> None:
        """時間経過による状態変化を適用"""
        if not self.state.last_updated:
            self.state.last_updated = datetime.now().isoformat()
            return
        last = datetime.fromisoformat(self.state.last_updated)
        hours = (datetime.now() - last).total_seconds() / 3600.0

        # 孤独感：時間経過で増加（上限10）
        self.state.loneliness = min(10.0, self.state.loneliness + hours * 0.5)
        # 社交エネルギー：時間経過で回復（上限10）
        self.state.social_energy = min(10.0, self.state.social_energy + hours * 0.3)
        self.state.last_updated = datetime.now().isoformat()
        self._save()

    def apply_delta(self, delta: dict) -> None:
        """会話後の能動的な状態更新"""
        def clamp(v): return max(0.0, min(10.0, v))
        self.state.loneliness = clamp(self.state.loneliness + delta.get("loneliness_delta", 0))
        self.state.curiosity = clamp(self.state.curiosity + delta.get("curiosity_delta", 0))
        self.state.social_energy = clamp(self.state.social_energy + delta.get("social_energy_delta", 0))
        self.state.last_conversation = datetime.now().isoformat()
        self.state.last_updated = datetime.now().isoformat()
        self._save()

    def get_display(self) -> str:
        """状態の日本語表示"""
        s = self.state
        return (
            f"孤独感: {s.loneliness:.1f}/10\n"
            f"好奇心: {s.curiosity:.1f}/10\n"
            f"社交エネルギー: {s.social_energy:.1f}/10"
        )

    def get_prompt_context(self) -> str:
        """プロンプトに渡す状態の文字列表現"""
        s = self.state
        last_conv = "なし"
        if s.last_conversation:
            hours = (datetime.now() - datetime.fromisoformat(s.last_conversation)).total_seconds() / 3600
            last_conv = f"{hours:.1f}時間前"
        return (
            f"孤独感 {s.loneliness:.1f}/10（高いほど誰かと話したい）\n"
            f"好奇心 {s.curiosity:.1f}/10（高いほど話したいことがある）\n"
            f"社交エネルギー {s.social_energy:.1f}/10（低いと疲れている）\n"
            f"最後の会話: {last_conv}"
        )

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.state = InternalState(**data)


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

        # ユーザーモデル（行動パターン・思考スタイル）
        self.user_model = UserModel(user_id)

        # 内部状態（このユーザーとの関係における感情パラメータ）
        self.internal_state = InternalStateManager(user_id)
        
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

    def get_user_model_summary(self) -> str:
        """ユーザーモデルのサマリー（プロンプト注入用）"""
        return self.user_model.get_summary()
    
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
