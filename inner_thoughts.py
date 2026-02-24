"""
Proactive AI Friend - Inner Thoughts Engine
思考生成、動機づけ評価、記憶抽出のコアエンジン
"""

import json
import re
from typing import Optional
from openai import AsyncOpenAI

import config
import prompts
from memory import MemoryManager, Thought


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
    
    def should_trigger_thought(self, memory: MemoryManager) -> tuple[bool, str]:
        """
        思考生成をトリガーすべきか判定
        
        Returns:
            (should_trigger, reason)
        """
        # 沈黙タイムアウト
        silence = memory.get_silence_duration()
        if silence > config.SILENCE_TIMEOUT:
            return True, f"silence_timeout ({int(silence)}s)"
        
        # 定期的な思考生成（会話中）
        if memory.last_user_message_time:
            if silence > config.THOUGHT_GENERATION_INTERVAL:
                return True, f"periodic ({int(silence)}s since last message)"
        
        # ユーザーの新しい発言があった
        if memory.short_term and memory.short_term[-1].role == "user":
            return True, "new_user_message"
        
        return False, ""
    
    # =========================================================================
    # ステージ2 & 3: Retrieval + Thought Formation（記憶取得 + 思考生成）
    # =========================================================================
    
    async def generate_thought(self, memory: MemoryManager, trigger_reason: str) -> Optional[Thought]:
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

        # 発言済みの思考
        expressed = memory.get_expressed_thoughts()
        expressed_thoughts = "\n".join([
            f"- {t.content}"
            for t in expressed
        ]) if expressed else "なし"

        # プロンプト生成
        prompt = prompts.format_thought_generation_prompt(
            conversation_context=conversation_context,
            user_memories=user_memories,
            pending_thoughts=pending_thoughts,
            expressed_thoughts=expressed_thoughts
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
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
        memory: MemoryManager
    ) -> dict:
        """
        思考の動機づけスコアを評価
        """
        prompt = prompts.format_motivation_evaluation_prompt(
            thought=thought_content,
            conversation_context=memory.get_context_summary(),
            silence_duration=memory.get_silence_duration(),
            consecutive_ai_messages=memory.consecutive_ai_messages,
            total_turns=len(memory.short_term)
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
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
        trigger_reason: str
    ) -> str:
        """
        自発的な発言を生成
        """
        system_prompt = prompts.format_proactive_system_prompt(
            thought=potential_response,
            user_memories=memory.get_all_memories_summary(),
            silence_duration=memory.get_silence_duration(),
            trigger_reason=trigger_reason
        )

        # 会話履歴をチャット形式で渡す（Litaの既発言を正しく認識させる）
        messages = memory.get_conversation_history()

        try:
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=openai_messages
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            print(f"Proactive response error: {e}")
            return ""
    
    async def generate_silence_break(self, memory: MemoryManager) -> str:
        """
        沈黙を破る発言を生成

        注意: 現在は未使用。沈黙タイムアウトも通常の思考生成→動機づけ評価パイプラインを
        通すように変更したため、この専用メソッドは呼ばれない。
        沈黙破り専用の生成ロジックが必要になった場合に備えて残している。
        """
        system_prompt = prompts.format_silence_break_system_prompt(
            user_memories=memory.get_all_memories_summary(),
            silence_duration=memory.get_silence_duration()
        )

        # 会話履歴をチャット形式で渡す
        messages = memory.get_conversation_history()

        try:
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=openai_messages
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            print(f"Silence break error: {e}")
            return ""
    
    # =========================================================================
    # 反応的応答（従来型）
    # =========================================================================
    
    async def generate_reactive_response(self, memory: MemoryManager) -> str:
        """
        ユーザーのメッセージに対する通常の応答
        """
        system_prompt = prompts.format_system_prompt(
            user_memories=memory.get_all_memories_summary()
        )
        
        messages = memory.get_conversation_history()
        
        try:
            # OpenAI形式: systemはmessages内の最初の要素として渡す
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
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
    # メインの処理フロー
    # =========================================================================
    
    async def process_proactive_cycle(self, memory: MemoryManager) -> Optional[dict]:
        """
        Proactiveサイクルの実行

        Returns:
            {"response": str, "thought": dict, "evaluation": dict, "trigger": str}
            または None
        """
        # 介入可能かチェック
        if not memory.can_intervene():
            return None

        # トリガーチェック
        should_trigger, trigger_reason = self.should_trigger_thought(memory)
        if not should_trigger:
            return None

        # 思考生成
        result = await self.generate_thought(memory, trigger_reason)
        if not result:
            return None

        thought, potential_response = result

        # 動機づけ評価
        evaluation = await self.evaluate_motivation(thought.content, memory)

        # 思考をリザーバーに保存
        thought.motivation_score = evaluation.get("overall_score", 0)
        thought.reasoning = evaluation.get("reasoning", "")
        memory.add_thought(
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
                    thought, potential_response, memory, trigger_reason
                )
                if response:
                    thought.expressed = True
                    was_expressed = True

        # 思考データは発言の有無に関わらず返す（ログ用）
        return {
            "response": response,
            "thought_content": thought.content,
            "trigger_reason": trigger_reason,
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
