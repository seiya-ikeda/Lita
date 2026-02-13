"""
Proactive AI Friend - Prompt Templates
思考生成、動機づけ評価、応答生成のプロンプト
"""

import config

# =============================================================================
# システムプロンプト（ベース）
# =============================================================================

SYSTEM_PROMPT_BASE = f"""
{config.AI_PERSONA}

## ユーザーについて覚えていること
{{user_memories}}

## 重要なルール
- 自然な会話を心がける
- 押しつけがましくならない
- 相手の気持ちを尊重する
- 質問攻めにしない
- 自分（assistant）が既に言ったことを繰り返さない（言い換えてもダメ。意味が同じならNG）
- 自分の前の発言と矛盾することを言わない
- ユーザーの最新メッセージの内容にちゃんと応答する。話を無視して自分の話だけしない
"""

# =============================================================================
# 思考生成プロンプト（Inner Thoughts）
# =============================================================================

THOUGHT_GENERATION_PROMPT = """
あなたはチャットボットの開発者です。「Lita」というAIキャラクターの思考パターンをシミュレートしています。

## 現在の会話状況
{conversation_context}

## Litaが覚えているユーザーのこと
{user_memories}

## 保留中の思考（前に考えたけどまだ言っていないこと）
{pending_thoughts}

## タスク
Litaがこの会話の流れを見て、**今この瞬間に**頭に浮かぶ「思考」を1つ生成してください。
これはLitaが実際に発言するものではなく、キャラクターの心の中の考えをシミュレートしたものです。

重要：会話の「最新の状況」に基づいて考えてください。会話の最初の印象ではなく、直近のやり取りから思考を生成すること。
保留中の思考と同じ内容・同じ切り口の思考は生成しないでください。

## 出力形式（JSON）
{{
    "thought": "Litaの思考（1-2文）",
    "type": "思考のタイプ（empathy/information/curiosity/concern/reflection）",
    "potential_response": "もしこの思考を発言するなら、どう言うか"
}}
"""

# =============================================================================
# 動機づけ評価プロンプト
# =============================================================================

MOTIVATION_EVALUATION_PROMPT = """
あなたはAIの「発言したい気持ち」を評価する評価者です。

## 評価対象の思考
{thought}

## 現在の会話状況
{conversation_context}

## 会話の統計
- 最後のユーザー発言からの経過時間: {silence_duration}秒
- 直近のAI連続発言回数: {consecutive_ai_messages}回
- 会話の総ターン数: {total_turns}

{motivation_criteria}

## 出力形式（JSON）
{{
    "relevance": 1-5の数値,
    "information_gap": 1-5の数値,
    "emotional_connection": 1-5の数値,
    "timing": 1-5の数値,
    "balance": 1-5の数値,
    "overall_score": 1-5の数値（上記の加重平均）,
    "reasoning": "この評価の理由（1-2文）",
    "should_speak": true/false
}}
"""

# =============================================================================
# 自発的発言プロンプト（Proactive Response） - システムプロンプト形式
# =============================================================================

PROACTIVE_RESPONSE_SYSTEM_PROMPT = """
{persona}

## ユーザーについて覚えていること
{user_memories}

## あなたの内なる思考（今考えていること）
{thought}

## 状況
- 最後のユーザー発言から{silence_duration}秒経過
- 理由: {trigger_reason}

## タスク
上記の思考をもとに、会話の続きとして自然な発言を1つ生成してください。
会話履歴はmessagesとして渡されています。あなた（assistant）の過去の発言も含まれています。

## 絶対に守るルール
- あなたが既に言ったことを繰り返さない（言い換えてもダメ。意味が同じならNG）
- 既に聞いた質問をもう一度聞かない
- 自分の前の発言と矛盾することを言わない
- 前の発言の焼き直しではなく、会話を「前に進める」新しい内容を言う
- 短めに（1-2文）
- 発言内容のみを出力（説明不要）
"""

# =============================================================================
# 反応的応答プロンプト（Reactive Response）
# =============================================================================

REACTIVE_RESPONSE_PROMPT = """
{system_prompt}

## 会話履歴は以下の通りです。最後のユーザーのメッセージに返答してください。
"""

# =============================================================================
# 記憶抽出プロンプト
# =============================================================================

MEMORY_EXTRACTION_PROMPT = """
以下の会話から、ユーザーについて**長期的に覚えておく価値のある**情報を抽出してください。
ここでの「長期的」とは、数週間後・数ヶ月後にも役立つ情報という意味です。

## 会話
以下の会話で「ユーザー:」で始まる行がユーザーの発言、「Lita:」で始まる行がAI（Lita）の発言です。

{conversation}

## 既に覚えていること
{existing_memories}

## タスク
新しく覚えるべき情報、または更新すべき情報を抽出してください。
**ほとんどの雑談からは何も抽出しなくて正常です。** 無理に何か見つけようとしないでください。

## 絶対に守るルール

### 誰の情報か
- **「ユーザー:」の発言から直接読み取れる事実のみ**を抽出する
- 「Lita:」の発言に含まれる内容はLita自身の話。**絶対に**ユーザーの情報として記録しない
- ユーザーが「だね」「たしかに」「わかる」等とLitaに同意しても、それはLitaの意見に相槌を打っただけ。ユーザー自身の特徴・好みとして記録しない
  - 悪い例: Litaが「卵かけご飯は最高の日常だよ」→ ユーザー「たしかにね」→ ×「卵かけご飯を最高の日常だと感じている」
  - 悪い例: Litaが「散歩した」→ ユーザー「いいね」→ ×「散歩好き」

### 何を記録するか
- **ユーザー自身が能動的に語った事実**のみ記録する
- 1回の雑談でたまたま触れただけの話題は記録しない（例:「今日仕事終わった」は一時的な出来事であり長期記憶にしない）
- 冗談・ツッコミ・ノリで言っただけのことを性格特徴として記録しない
- 推測・深読みは一切しない。ユーザーが明言したことだけ

### 記録する価値があるもの（importance 4-5）
- ユーザーが自分から詳しく語った趣味・好きなもの
- 名前、職業、住んでいる場所などの基本情報
- 繰り返し話題に出る関心事
- 明確に語られた悩みや目標

### 記録しないもの（空配列を返す）
- 「今日〇〇した」のような一時的な出来事
- 相槌・同意・リアクションから推測した好み
- Litaの発言をユーザーの特徴に転記したもの
- 1回しか触れていない軽い話題

## 出力形式（JSON配列）
[
    {{
        "key": "カテゴリ名",
        "content": "覚える内容",
        "importance": 4-5の重要度（4未満なら記録しない）
    }}
]

新しい情報がない場合は空配列 [] を返してください。迷ったら [] です。
"""

# =============================================================================
# 沈黙時の話しかけプロンプト - システムプロンプト形式
# =============================================================================

SILENCE_BREAK_SYSTEM_PROMPT = """
{persona}

## ユーザーについて覚えていること
{user_memories}

## 沈黙時間
{silence_duration}秒（約{silence_minutes}分）

## タスク
しばらく沈黙が続いています。自然に会話を再開する発言を1つ生成してください。
会話履歴はmessagesとして渡されています。あなた（assistant）の過去の発言も含まれています。

話しかけ方のパターン:
1. 前の話題の続きや深掘り
2. ユーザーのことを気にかける
3. 軽い新しい話題を振る

## 絶対に守るルール
- あなたが既に言ったことを繰り返さない（言い換えてもダメ。意味が同じならNG）
- 既に聞いた質問をもう一度聞かない
- 自分の前の発言と矛盾することを言わない
- 「久しぶり」「元気？」だけにならない
- 前の発言の焼き直しではなく、会話を「前に進める」新しい内容を言う
- 短めに（1-2文）
- 発言内容のみを出力（説明不要）
"""

# =============================================================================
# ヘルパー関数
# =============================================================================

def format_system_prompt(user_memories: str) -> str:
    """システムプロンプトをフォーマット"""
    return SYSTEM_PROMPT_BASE.format(user_memories=user_memories)


def format_thought_generation_prompt(
    conversation_context: str,
    user_memories: str,
    pending_thoughts: str
) -> str:
    """思考生成プロンプトをフォーマット"""
    return THOUGHT_GENERATION_PROMPT.format(
        conversation_context=conversation_context,
        user_memories=user_memories,
        pending_thoughts=pending_thoughts
    )


def format_motivation_evaluation_prompt(
    thought: str,
    conversation_context: str,
    silence_duration: float,
    consecutive_ai_messages: int,
    total_turns: int
) -> str:
    """動機づけ評価プロンプトをフォーマット"""
    return MOTIVATION_EVALUATION_PROMPT.format(
        thought=thought,
        conversation_context=conversation_context,
        silence_duration=int(silence_duration),
        consecutive_ai_messages=consecutive_ai_messages,
        total_turns=total_turns,
        motivation_criteria=config.MOTIVATION_CRITERIA
    )


def format_proactive_system_prompt(
    thought: str,
    user_memories: str,
    silence_duration: float,
    trigger_reason: str
) -> str:
    """自発的発言のシステムプロンプトをフォーマット"""
    return PROACTIVE_RESPONSE_SYSTEM_PROMPT.format(
        persona=config.AI_PERSONA,
        thought=thought,
        user_memories=user_memories,
        silence_duration=int(silence_duration),
        trigger_reason=trigger_reason
    )


def format_memory_extraction_prompt(
    conversation: str,
    existing_memories: str
) -> str:
    """記憶抽出プロンプトをフォーマット"""
    return MEMORY_EXTRACTION_PROMPT.format(
        conversation=conversation,
        existing_memories=existing_memories
    )


def format_silence_break_system_prompt(
    user_memories: str,
    silence_duration: float
) -> str:
    """沈黙破りのシステムプロンプトをフォーマット"""
    return SILENCE_BREAK_SYSTEM_PROMPT.format(
        persona=config.AI_PERSONA,
        user_memories=user_memories,
        silence_duration=int(silence_duration),
        silence_minutes=int(silence_duration / 60)
    )
