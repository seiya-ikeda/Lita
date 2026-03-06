"""
Proactive AI Friend - Inner Thoughts Engine
思考生成、動機づけ評価、記憶抽出のコアエンジン
"""

import json
import re
from datetime import datetime
from typing import Optional
from openai import AsyncOpenAI

import config
import prompts
from memory import MemoryManager, SelfNarrative, NarrativeEntry, Thought, InternalStateManager


class InnerThoughtsEngine:
    """
    Inner Thoughtsフレームワークの実装
    
    5つのステージ:
    1. Trigger - 思考生成のトリガー検出
    2. Retrieval - 関連記憶の取得
    3. Thought Formation - 思考の生成
    4. Evaluation - 動機づけ評価
    5. Participation - 発言の決定と実行
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.OPENAI_API_KEY or "no-key"
        )
    
    # =========================================================================
    # ステージ1: Trigger（トリガー検出）
    # =========================================================================
    
    # =========================================================================
    # ステージ2 & 3: Retrieval + Thought Formation（記憶取得 + 思考生成）
    # =========================================================================

    async def generate_thought(self, memory: MemoryManager, trigger_reason: str, trigger_type: str = "conversation") -> Optional[Thought]:
        """
        内なる思考を生成
        """
        # コンテキスト準備
        conversation_context = memory.get_context_summary()
        user_memories = memory.get_all_memories_summary()
        
        # 保留中の思考
        pending = memory.get_pending_thoughts()
        pending_thoughts = "\n".join([
            f"- {t.content} (スコア: {t.motivation_score})"
            for t in pending
        ]) if pending else "なし"

        # 発言済みの思考（thought_reservoir + short_termのassistant発言を合算）
        expressed = memory.get_expressed_thoughts()
        expressed_lines = [f"- {t.content}" for t in expressed]
        # reactive responseはthought_reservoirに入らないため、short_termからも補完
        recent_assistant = [
            f"- {m.content}"
            for m in list(memory.short_term)[-10:]
            if m.role == "assistant"
        ]
        all_expressed = expressed_lines + [l for l in recent_assistant if l not in expressed_lines]
        expressed_thoughts = "\n".join(all_expressed) if all_expressed else "なし"

        # プロンプト生成
        prompt = prompts.format_thought_generation_prompt(
            conversation_context=conversation_context,
            user_memories=user_memories,
            pending_thoughts=pending_thoughts,
            expressed_thoughts=expressed_thoughts,
            silence_duration=memory.get_silence_duration(),
            internal_state=memory.internal_state.get_prompt_context(),
            trigger_type=trigger_type
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # JSON抽出
            result = self._extract_json(response.choices[0].message.content)
            if not result:
                return None
            
            # 思考オブジェクト作成（まだ評価前なのでスコアは0）
            thought_content = result.get("thought", "")
            potential_response = result.get("potential_response", thought_content)
            
            return Thought(
                content=thought_content,
                motivation_score=0,  # 評価で更新
                reasoning="",
                timestamp="",
                triggered_by=trigger_reason
            ), potential_response
            
        except Exception as e:
            print(f"Thought generation error: {e}")
            return None
    
    # =========================================================================
    # ステージ4: Evaluation（動機づけ評価）
    # =========================================================================
    
    async def evaluate_motivation(
        self,
        thought_content: str,
        memory: MemoryManager,
        potential_response: str = ""
    ) -> dict:
        """
        思考の動機づけスコアを評価
        """
        prompt = prompts.format_motivation_evaluation_prompt(
            thought=thought_content,
            potential_response=potential_response,
            conversation_context=memory.get_context_summary(),
            silence_duration=memory.get_silence_duration(),
            consecutive_ai_messages=memory.consecutive_ai_messages,
            total_turns=len(memory.short_term)
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # デバッグ: レスポンス全体を確認
            choice = response.choices[0]
            raw_content = choice.message.content or ""
            if not raw_content:
                print(f"[DEBUG] Empty motivation response! finish_reason={choice.finish_reason}, tool_calls={choice.message.tool_calls}")
            
            result = self._extract_json(raw_content)
            if not result:
                # デバッグ: 何が返ってきたか確認
                print(f"[DEBUG] Motivation eval parse failed. Raw ({len(raw_content)} chars): {raw_content[:300]}...")
                return {"overall_score": 0, "should_speak": False, "reasoning": "Parse error"}
            
            return result
            
        except Exception as e:
            print(f"Motivation evaluation error: {e}")
            return {"overall_score": 0, "should_speak": False, "reasoning": str(e)}
    
    # =========================================================================
    # ステージ5: Participation（発言）
    # =========================================================================
    
    async def generate_proactive_response(
        self,
        thought: Thought,
        potential_response: str,
        memory: MemoryManager,
        trigger_reason: str,
        narrative: Optional[SelfNarrative] = None
    ) -> str:
        """
        自発的な発言を生成
        """
        system_prompt = prompts.format_proactive_system_prompt(
            thought=potential_response,
            user_memories=memory.get_all_memories_summary(),
            silence_duration=memory.get_silence_duration(),
            trigger_reason=trigger_reason,
            self_narrative=narrative.get_summary() if narrative else "なし",
            user_model=memory.get_user_model_summary()
        )

        # 会話履歴をチャット形式で渡す（Litaの既発言を正しく認識させる）
        messages = memory.get_conversation_history()

        try:
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=openai_messages
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            print(f"Proactive response error: {e}")
            return ""
    
    # =========================================================================
    # 反応的応答（従来型）
    # =========================================================================
    
    async def generate_reactive_response(self, memory: MemoryManager, narrative: Optional[SelfNarrative] = None) -> str:
        """
        ユーザーのメッセージに対する通常の応答
        """
        system_prompt = prompts.format_system_prompt(
            user_memories=memory.get_all_memories_summary(),
            self_narrative=narrative.get_summary() if narrative else "なし",
            user_model=memory.get_user_model_summary(),
            current_time=datetime.now().strftime("%Y年%m月%d日 %H:%M (%A)"),
            internal_state=memory.internal_state.get_prompt_context()
        )
        
        messages = memory.get_conversation_history()
        
        try:
            # OpenAI形式: systemはmessages内の最初の要素として渡す
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=openai_messages
            )
            
            content = response.choices[0].message.content
            # デバッグ: contentが空の場合、response全体を確認
            if not content:
                print(f"[DEBUG] Empty content! Full response: {response.choices[0]}")
            return content.strip() if content else "ごめん、ちょっと調子悪いみたい..."
            
        except Exception as e:
            import traceback
            print(f"Reactive response error: {e}")
            traceback.print_exc()
            return "ごめんね、ちょっと調子悪いみたい..."
    
    # =========================================================================
    # 記憶抽出
    # =========================================================================
    
    async def extract_memories(self, memory: MemoryManager) -> list[dict]:
        """
        会話から記憶すべき情報を抽出
        """
        # 直近の会話
        recent = memory.get_conversation_history(n=10)
        conversation = "\n".join([
            f"{'ユーザー' if m['role'] == 'user' else config.AI_NAME}: {m['content']}"
            for m in recent
        ])
        
        prompt = prompts.format_memory_extraction_prompt(
            conversation=conversation,
            existing_memories=memory.get_all_memories_summary()
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_content = response.choices[0].message.content or ""
            print(f"[DEBUG] Memory extraction raw response: {raw_content[:500]}...")
            
            result = self._extract_json(raw_content)
            if isinstance(result, list):
                return result
            print(f"[DEBUG] Memory extraction: result is not list, got {type(result)}: {result}")
            return []
            
        except Exception as e:
            print(f"Memory extraction error: {e}")
            return []
    
    # =========================================================================
    # 自己ナラティブ
    # =========================================================================

    async def update_self_narrative(
        self,
        memory: MemoryManager,
        narrative: SelfNarrative,
    ) -> Optional[NarrativeEntry]:
        """
        会話後にLitaの自己認識を更新する。
        発見がなければ None を返す（多くの会話では更新なしが正常）。
        """
        conversation = "\n".join([
            f"{'ユーザー' if m['role'] == 'user' else config.AI_NAME}: {m['content']}"
            for m in memory.get_conversation_history(n=15)
        ])

        prompt = prompts.format_narrative_update_prompt(
            user_id=memory.user_id,
            conversation=conversation,
            existing_narrative=narrative.get_summary()
        )

        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            result = self._extract_json(response.choices[0].message.content or "")
            if not result:
                return None

            content = result.get("content", "").strip()
            if not content:
                return None

            chapter = result.get("chapter", "self")
            is_contradiction = result.get("is_contradiction", False)
            contradicts_content = result.get("contradicts_content", "").strip()

            # 矛盾する既存エントリのconfidenceを下げる
            if is_contradiction and contradicts_content:
                narrative.weaken(chapter, contradicts_content)
                print(f"[Narrative] Weakened [{chapter}]: {contradicts_content[:60]}")

            entry = narrative.add_entry(
                content=content,
                chapter=chapter,
                related_user=memory.user_id,
                contradicts=contradicts_content if is_contradiction else None
            )
            print(f"[Narrative] New entry [{chapter}]: {content[:80]}")
            return entry

        except Exception as e:
            print(f"Narrative update error: {e}")
            return None

    async def update_user_model(self, memory: MemoryManager) -> list[dict]:
        """
        会話からユーザーの行動パターンを抽出してモデルを更新する。
        更新したエントリの情報リストを返す（research_log用）。
        """
        conversation = "\n".join([
            f"{'ユーザー' if m['role'] == 'user' else config.AI_NAME}: {m['content']}"
            for m in memory.get_conversation_history(n=15)
        ])

        prompt = prompts.format_user_model_update_prompt(
            user_id=memory.user_id,
            conversation=conversation,
            existing_model=memory.get_user_model_summary()
        )

        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            result = self._extract_json(response.choices[0].message.content or "")
            if not isinstance(result, list):
                return []

            updated = []
            for item in result:
                dimension = item.get("dimension", "communication")
                content = item.get("content", "").strip()
                is_contradiction = item.get("is_contradiction", False)
                if not content:
                    continue
                if is_contradiction:
                    memory.user_model.weaken(dimension, content)
                    # weaken後のconfidenceを取得
                    conf = next(
                        (e.confidence for e in memory.user_model.entries
                         if e.dimension == dimension and memory.user_model._similar(e.content, content)),
                        0.0
                    )
                else:
                    entry = memory.user_model.add_or_update(dimension, content)
                    conf = entry.confidence
                updated.append({
                    "dimension": dimension,
                    "content": content,
                    "is_contradiction": is_contradiction,
                    "confidence_after": conf,
                })

            if updated:
                print(f"[UserModel] {len(updated)} entries updated for {memory.user_id}")
            return updated

        except Exception as e:
            print(f"User model update error: {e}")
            return []

    async def summarize_session(self, memory: MemoryManager) -> bool:
        """
        睡眠時: 今日の会話をサマリーしてlong_termに書き込む（LD-Agent式）。
        short_termが空の場合はスキップ。
        """
        history = memory.get_conversation_history()
        if not history:
            return False

        conversation = "\n".join([
            f"{'ユーザー' if m['role'] == 'user' else config.AI_NAME}: {m['content']}"
            for m in history
        ])
        prompt = prompts.format_session_summary_prompt(
            user_id=memory.user_id,
            conversation=conversation
        )

        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            result = self._extract_json(response.choices[0].message.content or "")
            summary = result.get("summary", "").strip() if isinstance(result, dict) else ""
            if not summary:
                return False

            date_str = datetime.now().strftime("%Y-%m-%d")
            memory.add_long_term_memory(
                key=f"session_summary_{date_str}",
                content=summary,
                importance=3.0
            )
            print(f"[Sleep] Session summary saved for {memory.user_id}: {summary[:60]}")
            return True

        except Exception as e:
            print(f"Session summary error: {e}")
            return False

    # =========================================================================
    # 内部状態管理
    # =========================================================================

    async def update_internal_state(
        self,
        memory: MemoryManager,
        internal_state: InternalStateManager
    ) -> None:
        """会話後の内部状態更新（LLMが状態変化を評価）"""
        recent = list(memory.short_term)[-10:]
        if not recent:
            return

        conversation = "\n".join([
            f"{'ユーザー' if m.role == 'user' else 'Lita'}: {m.content}"
            for m in recent
        ])

        prompt = prompts.format_internal_state_update_prompt(
            current_state=internal_state.get_prompt_context(),
            conversation=conversation
        )

        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            result = self._extract_json(response.choices[0].message.content or "")
            if isinstance(result, dict):
                internal_state.apply_delta(result)
                print(f"[InternalState] Updated: loneliness={internal_state.state.loneliness:.1f} "
                      f"curiosity={internal_state.state.curiosity:.1f} "
                      f"energy={internal_state.state.social_energy:.1f} "
                      f"({result.get('reasoning', '')})")
        except Exception as e:
            print(f"InternalState update error: {e}")

    # =========================================================================
    # メインの処理フロー
    # =========================================================================

    async def process_proactive_cycle(self, memory: MemoryManager, narrative: Optional[SelfNarrative] = None, trigger_type: str = "conversation") -> Optional[dict]:
        """
        Proactiveサイクルの実行：思考生成 → ブレーキ評価 → 発話

        Returns:
            {"response": str, "thought_content": str, "trigger_reason": str,
             "motivation_score": float, "evaluation": dict, "was_expressed": bool}
            または None（介入不可の場合）
        """
        # スパム防止チェック
        if not memory.can_intervene():
            return None

        silence = memory.get_silence_duration()
        trigger_reason = f"{trigger_type} ({int(silence)}s since last message)"

        # 思考生成（沈黙時間・内部状態をコンテキストとして渡す）
        result = await self.generate_thought(memory, trigger_reason, trigger_type=trigger_type)
        if not result:
            return None

        thought, potential_response = result

        # 動機づけ評価（potential_responseも渡して焼き直し検出を正確にする）
        evaluation = await self.evaluate_motivation(thought.content, memory, potential_response)

        # 思考をリザーバーに保存
        thought.motivation_score = evaluation.get("overall_score", 0)
        thought.reasoning = evaluation.get("reasoning", "")
        reservoir_thought = memory.add_thought(
            content=thought.content,
            motivation_score=thought.motivation_score,
            reasoning=thought.reasoning,
            triggered_by=trigger_reason
        )

        # 閾値チェック
        was_expressed = False
        response = None
        if thought.motivation_score >= config.MOTIVATION_THRESHOLD:
            if evaluation.get("should_speak", False):
                # 発言生成
                response = await self.generate_proactive_response(
                    thought, potential_response, memory, trigger_reason, narrative
                )
                if response:
                    reservoir_thought.expressed = True
                    was_expressed = True

        # 思考データは発言の有無に関わらず返す（ログ用）
        return {
            "response": response,
            "thought_content": thought.content,
            "trigger_reason": trigger_reason,
            "trigger_type": trigger_type,
            "motivation_score": thought.motivation_score,
            "evaluation": evaluation,
            "was_expressed": was_expressed,
        }
    
    # =========================================================================
    # ユーティリティ
    # =========================================================================
    
    def _extract_json(self, text: str) -> Optional[dict | list]:
        """テキストからJSONを抽出"""
        # コードブロック内のJSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 直接JSON
        try:
            # { または [ で始まる部分を探す
            start_idx = min(
                text.find('{') if text.find('{') != -1 else len(text),
                text.find('[') if text.find('[') != -1 else len(text)
            )
            if start_idx < len(text):
                # 対応する閉じ括弧を探す
                bracket_count = 0
                end_idx = start_idx
                open_bracket = text[start_idx]
                close_bracket = '}' if open_bracket == '{' else ']'
                
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == open_bracket:
                        bracket_count += 1
                    elif char == close_bracket:
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                
                return json.loads(text[start_idx:end_idx])
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
