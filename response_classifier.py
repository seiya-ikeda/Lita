"""
Proactive AI Friend - Response Classifier
メッセージに対してリアクション/返信/無視を判定する
"""

from openai import AsyncOpenAI
from typing import Optional
from dataclasses import dataclass
import json
import re

import config


@dataclass
class ResponseDecision:
    """応答の判定結果"""
    action: str  # "reply", "react", "none"
    reaction: Optional[str]  # リアクションの場合のSlack絵文字名
    reason: str  # 判定理由


class ResponseClassifier:
    """
    メッセージに対する応答タイプを判定するクラス
    
    判定タイプ:
    - reply: 返信が必要（質問、相談、話題提供など）
    - react: リアクションで十分（了解、ありがとう、相槌など）
    - none: 何もしなくていい（独り言、誤送信っぽいなど）
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.OPENAI_API_KEY or "no-key"
        )
    
    async def classify(
        self, 
        message: str, 
        conversation_context: str = ""
    ) -> ResponseDecision:
        """
        メッセージの応答タイプを判定
        """
        prompt = f"""
あなたは友達とのチャットで、相手のメッセージにどう反応するか判断します。

## 相手のメッセージ
「{message}」

## 最近の会話の流れ
{conversation_context if conversation_context else "（なし）"}

## 判定基準

### reply（返信すべき）
- 質問されている
- 相談や悩みを話している
- 新しい話題・近況・出来事を話している
- 意見や感想を求めている
- 何かを伝えようとしている（長短問わず）
- 挨拶・報告（「おはよう」「仕事終わり！」「疲れた」など）← 会話の流れに関わらず常にreply

### react（リアクションで十分）
こちらの発言に対する短い反応で、やり取りをそれ以上広げる意図がないもの：
- 「おk」「了解」「りょ」などの短い受け取り
- 「ありがとう」「さんきゅー」などのお礼
- 「なるほど」「たしかに」「だね」「せやな」など同意・相槌で話を閉じる発言
- 「w」「草」「笑」などの笑いだけのメッセージ
- 「おやすみ」「またね」「じゃあね」などの別れの挨拶
- 話題が一段落した後の短い締めの一言

### none（何もしなくていい）
- 明らかな誤送信
- 自分に向けられていないメッセージ

## リアクション絵文字について
actionがreactの場合、そのメッセージの感情・文脈に一番フィットするSlack絵文字名を選ぶ。
thumbsupなどに偏らず、状況に応じて多様な絵文字を使うこと。
（Slackで使える絵文字名なら何でもOK）

## 出力形式（JSON）
{{
    "action": "reply" or "react" or "none",
    "reaction": "Slack絵文字名（actionがreactの場合のみ）",
    "reason": "判定理由（1文）"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.choices[0].message.content
            result = self._extract_json(text)
            
            if not result:
                # パース失敗時はデフォルトで返信
                return ResponseDecision(
                    action="reply",
                    reaction=None,
                    reason="Parse error - defaulting to reply"
                )
            
            action = result.get("action", "reply")
            reaction = result.get("reaction")
            
            return ResponseDecision(
                action=action,
                reaction=reaction,
                reason=result.get("reason", "")
            )
            
        except Exception as e:
            print(f"Classification error: {e}")
            # エラー時はデフォルトで返信
            return ResponseDecision(
                action="reply",
                reaction=None,
                reason=f"Error: {str(e)}"
            )
    
    def _extract_json(self, text: str) -> Optional[dict]:
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
            start_idx = text.find('{')
            if start_idx != -1:
                bracket_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            return json.loads(text[start_idx:i+1])
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
