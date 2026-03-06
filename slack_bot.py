"""
Proactive AI Friend - Slack Bot
Inner ThoughtsフレームワークをSlackに移植
"""

import asyncio
import random
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

import config
from memory import MemoryManager, SelfNarrative
from inner_thoughts import InnerThoughtsEngine
from research_logger import ResearchLogger
from information_gatherer import InformationGatherer
from response_classifier import ResponseClassifier


class ProactiveAISlackBot:
    """
    Proactive AI Friend Slack Bot

    機能:
    - ユーザーメッセージへの反応的応答
    - Inner Thoughtsによる自発的発言
    - 長期記憶によるユーザー理解
    - 研究用ログ収集
    """

    def __init__(self):
        self.app = AsyncApp(token=config.SLACK_BOT_TOKEN)

        # エンジン
        self.engine = InnerThoughtsEngine()
        self.info_gatherer = InformationGatherer()
        self.classifier = ResponseClassifier()

        # ユーザーごとの記憶管理
        self.memories: dict[str, MemoryManager] = {}

        # Lita自己史（ユーザーをまたいで共有）
        self.narrative = SelfNarrative()

        # 最終ナラティブ整理時刻
        self.last_narrative_consolidation: Optional[datetime] = None

        # セッション管理
        self.session_id = str(uuid.uuid4())[:8]
        self.logger = ResearchLogger(self.session_id)

        # ユーザーごとの会話チャンネル（DM channel ID）
        self.user_channels: dict[str, str] = {}

        # 処理中フラグ（二重応答防止）
        self.processing: set[str] = set()

        # Bot自身のユーザーID（起動時に取得）
        self.bot_user_id: str = ""

        # ワークスペースのカスタム絵文字一覧（起動時に取得）
        self.custom_emojis: list[str] = []

        # イベントハンドラとコマンドを登録
        self._register_handlers()

    # =========================================================================
    # ハンドラ登録
    # =========================================================================

    def _register_handlers(self):
        """イベントハンドラとスラッシュコマンドを登録"""

        # メッセージイベント
        @self.app.event("message")
        async def handle_message(event, say, client):
            await self._on_message(event, say, client)

        # スラッシュコマンド
        @self.app.command("/lita-status")
        async def handle_status(ack, command, client):
            await ack()
            await self._cmd_status(command, client)

        @self.app.command("/lita-memories")
        async def handle_memories(ack, command, client):
            await ack()
            await self._cmd_memories(command, client)

        @self.app.command("/lita-forget")
        async def handle_forget(ack, command, client):
            await ack()
            await self._cmd_forget(command, client)

        @self.app.command("/lita-thoughts")
        async def handle_thoughts(ack, command, client):
            await ack()
            await self._cmd_thoughts(command, client)

        @self.app.command("/lita-config")
        async def handle_config(ack, command, client):
            await ack()
            await self._cmd_config(command, client)

        @self.app.command("/lita-interests")
        async def handle_interests(ack, command, client):
            await ack()
            await self._cmd_interests(command, client)

        @self.app.command("/lita-search")
        async def handle_search(ack, command, client):
            await ack()
            await self._cmd_search(command, client)

        @self.app.command("/lita-export")
        async def handle_export(ack, command, client):
            await ack()
            await self._cmd_export(command, client)

        @self.app.command("/lita-narrative")
        async def handle_narrative(ack, command, client):
            await ack()
            await self._cmd_narrative(command, client)

        @self.app.command("/lita-usermodel")
        async def handle_usermodel(ack, command, client):
            await ack()
            await self._cmd_usermodel(command, client)

    # =========================================================================
    # メッセージ処理
    # =========================================================================

    async def _on_message(self, event, say, client):
        """メッセージ受信時の処理"""
        # Bot自身のメッセージ、サブタイプ付き（編集・削除等）は無視
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user", "")
        text = event.get("text", "")
        channel = event.get("channel", "")
        channel_type = event.get("channel_type", "")
        ts = event.get("ts", "")

        # 古すぎるイベントは無視（再起動時の再送対策）
        if ts:
            import time
            event_age = time.time() - float(ts)
            if event_age > 120:
                return

        # DMまたはメンションされた場合のみ反応
        is_dm = channel_type == "im"
        is_mentioned = f"<@{self.bot_user_id}>" in text

        if not (is_dm or is_mentioned):
            return

        # メンションを除去
        content = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        if not content:
            return

        # 二重処理防止
        process_key = f"{user_id}:{ts}"
        if process_key in self.processing:
            return
        self.processing.add(process_key)

        try:
            # 記憶マネージャー取得または作成
            memory = self._get_memory(user_id)

            # ユーザーとチャンネルの対応を保存
            self.user_channels[user_id] = channel

            # 応答タイプを判定（add_message前にコンテキストを取得し、現在のメッセージが混入しないようにする）
            context = memory.get_context_summary()
            decision = await self.classifier.classify(content, context, custom_emojis=self.custom_emojis)

            # ユーザーメッセージを記録
            memory.add_message("user", content)
            self.logger.log_user_message(user_id, content)

            if decision.action == "react" and decision.reaction:
                # リアクションで十分な場合
                try:
                    await client.reactions_add(
                        channel=channel,
                        timestamp=ts,
                        name=self._sanitize_reaction_name(decision.reaction)
                    )
                except SlackApiError as e:
                    err = e.response["error"]
                    if err == "message_not_found":
                        print(f"[Reaction] Message already deleted, skipping: ts={ts}")
                        return
                    if err == "invalid_name":
                        # LLMが存在しない絵文字名を返した場合はthumbsupで再試行
                        try:
                            await client.reactions_add(channel=channel, timestamp=ts, name="thumbsup")
                        except SlackApiError:
                            pass
                        return
                    raise
                # short_termにも記録（proactiveループの「last==user」スキップを解除するため）
                memory.add_message("assistant", f"[reaction: {decision.reaction}]", is_reaction=True)
                self.logger.log_ai_response(
                    user_id,
                    f"[reaction: {decision.reaction}]",
                    is_proactive=False,
                    metadata={"type": "reaction", "reason": decision.reason}
                )
                # 思考ログにもリアクション判断を記録
                if config.LOG_THOUGHTS:
                    self.logger.log_thought(
                        user_id=user_id,
                        thought_content=f"「{content[:50]}」に対してリアクションで返す: {decision.reason}",
                        trigger_reason="user_message (classifier: react)",
                        motivation_score=5,
                        evaluation_details={"action": "react", "reaction": decision.reaction, "reason": decision.reason},
                        was_expressed=True,
                        response_if_expressed=f"[reaction: {decision.reaction}]",
                    )

            elif decision.action == "reply":
                # 返信を生成
                response = await self.engine.generate_reactive_response(memory, self.narrative)

                is_fallback = "調子悪いみたい..." in response
                if not is_fallback:
                    memory.add_message("assistant", response)
                self.logger.log_ai_response(user_id, response, is_proactive=False)

                await client.chat_postMessage(
                    channel=channel,
                    text=response
                )

            # 内部状態更新（5ターンごと）
            if len(memory.short_term) >= config.INTERNAL_STATE_UPDATE_INTERVAL and \
               len(memory.short_term) % config.INTERNAL_STATE_UPDATE_INTERVAL <= 1:
                await self._update_internal_state(memory)

        finally:
            self.processing.discard(process_key)

    # =========================================================================
    # Proactiveサイクル
    # =========================================================================

    async def _proactive_loop(self):
        """定期的に実行されるProactiveサイクル（思考生成 → ブレーキ評価 → 発話）"""
        while True:
            await asyncio.sleep(config.THOUGHT_GENERATION_INTERVAL)

            for user_id, memory in list(self.memories.items()):
                try:
                    # 時間経過による内部状態の変化（発言可否に関わらず常に適用）
                    memory.internal_state.apply_passive_drift()
                    s = memory.internal_state.state
                    self.logger.log_internal_state(
                        user_id=user_id,
                        loneliness=s.loneliness,
                        curiosity=s.curiosity,
                        social_energy=s.social_energy,
                        trigger="passive_drift",
                    )

                    # Reactiveが未応答のユーザーメッセージがある場合はスキップ（二重応答防止）
                    if memory.short_term and memory.short_term[-1].role == "user":
                        continue

                    # トリガータイプをランダムに選択（多様性確保）
                    # conversation:40% / memory_recall:35% / self_thought:25%
                    trigger_type = random.choices(
                        ["conversation", "memory_recall", "self_thought"],
                        weights=[40, 35, 25]
                    )[0]
                    result = await self.engine.process_proactive_cycle(memory, self.narrative, trigger_type=trigger_type)
                    if result is None:
                        continue

                    # 思考ログを記録（発言の有無に関わらず）
                    if result.get("thought_content"):
                        self.logger.log_thought(
                            user_id=user_id,
                            thought_content=result["thought_content"],
                            trigger_reason=result["trigger_reason"],
                            motivation_score=result["motivation_score"],
                            evaluation_details=result["evaluation"],
                            was_expressed=result["was_expressed"],
                            response_if_expressed=result.get("response"),
                        )

                    # 発言がある場合のみ送信
                    response = result.get("response")
                    if response:
                        # 重複チェック: 直近のassistant発言と同一なら送らない
                        recent_ai = [m.content for m in list(memory.short_term)[-5:] if m.role == "assistant"]
                        if response in recent_ai:
                            continue
                        channel = self.user_channels.get(user_id)
                        if channel:
                            memory.add_message("assistant", response)
                            self.logger.log_ai_response(
                                user_id, response, is_proactive=True,
                                metadata={
                                    "trigger": result["trigger_reason"],
                                    "motivation_score": result["motivation_score"],
                                }
                            )
                            await self.app.client.chat_postMessage(
                                channel=channel,
                                text=response
                            )

                except Exception as e:
                    print(f"Proactive cycle error for {user_id}: {e}")

    async def _info_gather_loop(self):
        """定期的にユーザーの興味に合った情報を収集し、自然に共有する"""
        while True:
            await asyncio.sleep(config.SEARCH_INTERVAL)

            for user_id, memory in list(self.memories.items()):
                if not config.ENABLE_INFORMATION_GATHERING:
                    break
                try:
                    result = await self.info_gatherer.find_shareable_article(memory)
                    if not result:
                        continue

                    article, message = result
                    channel = self.user_channels.get(user_id)
                    if not channel:
                        continue

                    # 発言可否チェック（最小間隔）
                    if not memory.can_intervene():
                        continue

                    memory.add_message("assistant", message)
                    self.logger.log_ai_response(
                        user_id, message, is_proactive=True,
                        metadata={"trigger": "info_share", "article_url": article.url}
                    )
                    await self.app.client.chat_postMessage(channel=channel, text=message)
                    print(f"[InfoGather] Shared to {user_id}: {article.title[:50]}")

                except Exception as e:
                    print(f"Info gather loop error for {user_id}: {e}")

    async def _daily_sleep_loop(self):
        """毎日0時: 睡眠サイクル（narrative更新・session summary・short_termリセット）"""
        while True:
            now = datetime.now()
            # 次の0時までの秒数を計算
            next_midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            await asyncio.sleep((next_midnight - now).total_seconds())

            print("[Sleep] Starting daily sleep cycle...")
            try:
                for user_id, memory in self.memories.items():
                    if not memory.short_term:
                        continue
                    # セッションサマリーをlong_termへ
                    await self.engine.summarize_session(memory)
                    # 長期記憶の抽出（セッション全体を対象）
                    await self._extract_and_save_memories(memory)
                    # ナラティブ・ユーザーモデル更新
                    entry = await self.engine.update_self_narrative(memory, self.narrative)
                    if entry:
                        self.logger.log_narrative_update(
                            user_id=user_id,
                            chapter=entry.chapter,
                            content=entry.content,
                            contradicts=entry.contradicts,
                            total_entries=len(self.narrative.entries),
                        )
                    user_model_updates = await self.engine.update_user_model(memory)
                    for upd in user_model_updates:
                        self.logger.log_user_model_update(
                            user_id=user_id,
                            dimension=upd["dimension"],
                            content=upd["content"],
                            is_contradiction=upd["is_contradiction"],
                            confidence_after=upd["confidence_after"],
                        )
                    # short_termをクリア
                    memory.short_term.clear()
                    print(f"[Sleep] Done for {user_id}")

                # session_idを更新
                self.session_id = str(uuid.uuid4())[:8]
                self.logger = ResearchLogger(self.session_id)
                print(f"[Sleep] New session: {self.session_id}")

            except Exception as e:
                print(f"Daily sleep loop error: {e}")


    # =========================================================================
    # スラッシュコマンド
    # =========================================================================

    async def _cmd_status(self, command, client):
        """現在のステータスを表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        stats = self.logger.get_thought_statistics()
        metrics = self.logger.calculate_metrics(user_id)

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{config.AI_NAME} Status"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Session ID:* `{self.session_id}`"},
                    {"type": "mrkdwn", "text": f"*Condition:* `{config.EXPERIMENT_CONDITION}`"},
                ]
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total turns:* {metrics.total_turns}"},
                    {"type": "mrkdwn", "text": f"*Your messages:* {metrics.user_messages}"},
                    {"type": "mrkdwn", "text": f"*My replies:* {metrics.ai_reactive_responses}"},
                    {"type": "mrkdwn", "text": f"*My initiatives:* {metrics.ai_proactive_interventions}"},
                ]
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Short-term memory:* {len(memory.short_term)} turns"},
                    {"type": "mrkdwn", "text": f"*Long-term memory:* {len(memory.long_term)} items"},
                    {"type": "mrkdwn", "text": f"*Pending thoughts:* {len(memory.thought_reservoir)}"},
                ]
            },
        ]

        if stats:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Generated thoughts:* {stats.get('total_thoughts', 0)}"},
                    {"type": "mrkdwn", "text": f"*Expressed thoughts:* {stats.get('expressed_thoughts', 0)}"},
                    {"type": "mrkdwn", "text": f"*Avg motivation:* {stats.get('avg_motivation_score', 0):.2f}"},
                ]
            })

        # 内部状態（このユーザーとの関係）
        memory.internal_state.apply_passive_drift()
        s = memory.internal_state.state
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": (
                f"*Internal State:*\n"
                f"孤独感 {s.loneliness:.1f}/10  好奇心 {s.curiosity:.1f}/10  社交エネルギー {s.social_energy:.1f}/10"
            )}
        })

        await client.chat_postEphemeral(
            channel=channel, user=user_id, blocks=blocks,
            text=f"{config.AI_NAME} Status"
        )

    async def _cmd_memories(self, command, client):
        """覚えていることを表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        memories_text = memory.get_all_memories_summary()

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text=f"*{config.AI_NAME} が覚えていること:*\n{memories_text if memories_text else 'まだ何も覚えていません'}"
        )

    async def _cmd_forget(self, command, client):
        """記憶をリセット"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        if user_id in self.memories:
            del self.memories[user_id]

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text="記憶をリセットしました。また一から仲良くなりましょう！"
        )

    async def _cmd_thoughts(self, command, client):
        """保留中の思考を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        pending = memory.get_pending_thoughts()

        if not pending:
            await client.chat_postEphemeral(
                channel=channel, user=user_id,
                text="今のところ、特に言いたいことはないかな"
            )
            return

        threshold = config.MOTIVATION_THRESHOLD
        lines = []
        for i, thought in enumerate(pending[:5], 1):
            passed = "passed" if thought.motivation_score >= threshold else "below"
            line = f"*Thought {i}* (score: {thought.motivation_score:.1f})\n"
            line += f"> {thought.content[:100]}\n"
            if thought.reasoning:
                line += f"> Reason: {thought.reasoning[:100]}\n"
            line += f"> Threshold({threshold}): {passed}\n"
            lines.append(line)

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text="\n".join(lines)
        )

    async def _cmd_config(self, command, client):
        """設定を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        text = (
            f"*Proactive Settings*\n"
            f"- Motivation threshold: {config.MOTIVATION_THRESHOLD}\n"
            f"- Thought interval: {config.THOUGHT_GENERATION_INTERVAL}s\n"
            f"- Min intervention interval: {config.MIN_INTERVENTION_INTERVAL}s\n\n"
            f"*Information Gathering*\n"
            f"- Enabled: {config.ENABLE_INFORMATION_GATHERING}\n"
            f"- Search interval: {config.SEARCH_INTERVAL}s\n"
            f"- Max daily shares: {config.MAX_DAILY_SHARES}"
        )

        await client.chat_postEphemeral(
            channel=channel, user=user_id, text=text
        )

    async def _cmd_interests(self, command, client):
        """推測された興味を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        interests = await self.info_gatherer.extract_interests(memory)

        if interests:
            text = "*あなたの興味（推測）:*\n" + "\n".join([f"- {i}" for i in interests])
        else:
            text = "まだ興味を把握できていません。もっとお話ししましょう！"

        await client.chat_postEphemeral(
            channel=channel, user=user_id, text=text
        )

    async def _cmd_search(self, command, client):
        """今すぐ情報を探して共有"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        if not config.BRAVE_SEARCH_API_KEY:
            await client.chat_postEphemeral(
                channel=channel, user=user_id,
                text="検索機能が設定されていません（BRAVE_SEARCH_API_KEY が必要です）"
            )
            return

        memory = self._get_memory(user_id)

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text="あなたが興味ありそうな情報を探しています..."
        )

        result = await self.info_gatherer.find_shareable_article(memory)

        if result:
            article, message = result

            memory.add_message("assistant", message)
            self.logger.log_ai_response(
                user_id, message, is_proactive=True,
                metadata={
                    "trigger": "manual_search",
                    "article_title": article.title,
                    "relevance_score": article.relevance_score
                }
            )

            await client.chat_postMessage(channel=channel, text=message)
        else:
            await client.chat_postEphemeral(
                channel=channel, user=user_id,
                text="今回は特に良さそうな情報が見つからなかったかも...また後で探してみるね！"
            )

    async def _cmd_narrative(self, command, client):
        """Litaの自己認識を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        summary = self.narrative.get_summary(max_entries=10)
        count = len(self.narrative.entries)

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text=f"*Litaの自己認識（{count}エントリ）:*\n{summary if summary != 'なし' else 'まだ形成されていません'}"
        )

    async def _cmd_usermodel(self, command, client):
        """ユーザーモデルを表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        summary = memory.get_user_model_summary()
        count = len(memory.user_model.entries)

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text=f"*Litaが観測したあなたのパターン（{count}エントリ）:*\n{summary if summary != 'なし' else 'まだ観測されていません'}"
        )

    async def _cmd_export(self, command, client):
        """ログをエクスポート"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        self.logger.save_session_summary()
        summary = self.logger.export_session_summary()

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text=f"*Log exported*\nSaved to: `{config.LOG_DIRECTORY}/`\n```{str(summary)[:500]}...```"
        )

    # =========================================================================
    # ヘルパー関数
    # =========================================================================

    def _get_memory(self, user_id: str) -> MemoryManager:
        """ユーザーの記憶マネージャーを取得または作成"""
        if user_id not in self.memories:
            self.memories[user_id] = MemoryManager(user_id)
        return self.memories[user_id]

    @staticmethod
    def _sanitize_reaction_name(name: str) -> str:
        """LLMが返したリアクション名をサニタイズ"""
        import re
        # コロンや余分な空白を除去（:thumbsup: → thumbsup）
        name = name.strip().strip(":").strip()
        # スペースをアンダースコアに変換
        name = name.replace(" ", "_")
        # 小文字に変換
        name = name.lower()
        # Slack絵文字名に使えない文字を除去（英数字・アンダースコア・ハイフンのみ許可）
        name = re.sub(r"[^a-z0-9_\-]", "", name)
        # 空になったらデフォルトに
        return name if name else "thumbsup"

    async def _extract_and_save_memories(self, memory: MemoryManager):
        """記憶を抽出して保存（importance 4以上のみ）"""
        try:
            new_memories = await self.engine.extract_memories(memory)
            for mem in new_memories:
                importance = mem.get("importance", 0)
                if importance >= 4:
                    memory.add_long_term_memory(
                        key=mem.get("key", "その他"),
                        content=mem.get("content", ""),
                        importance=importance
                    )
                else:
                    print(f"[DEBUG] Memory skipped (importance {importance} < 4): {mem.get('content', '')[:50]}")
        except Exception as e:
            print(f"Memory extraction error: {e}")


    async def _update_internal_state(self, memory: MemoryManager):
        """内部状態を更新"""
        try:
            await self.engine.update_internal_state(memory, memory.internal_state)
            s = memory.internal_state.state
            self.logger.log_internal_state(
                user_id=memory.user_id,
                loneliness=s.loneliness,
                curiosity=s.curiosity,
                social_energy=s.social_energy,
                trigger="conversation_update",
            )
        except Exception as e:
            print(f"InternalState update error: {e}")

    # =========================================================================
    # 起動
    # =========================================================================

    async def start(self):
        """Botを起動"""
        # Bot自身のユーザーIDを取得
        auth = await self.app.client.auth_test()
        self.bot_user_id = auth["user_id"]

        # カスタム絵文字を取得（emoji:read スコープが必要）
        try:
            emoji_resp = await self.app.client.emoji_list()
            self.custom_emojis = list(emoji_resp.get("emoji", {}).keys())
            print(f"Custom emojis loaded: {len(self.custom_emojis)}")
        except Exception as e:
            print(f"[Emoji] Failed to load custom emojis: {e}")

        print(f"{config.AI_NAME} がオンラインになりました！ (Slack)")
        print(f"Session ID: {self.session_id}")
        print(f"Condition: {config.EXPERIMENT_CONDITION}")
        print(f"Bot User ID: {self.bot_user_id}")
        print("-" * 50)

        # バックグラウンドタスクを起動
        asyncio.create_task(self._daily_sleep_loop())
        if config.EXPERIMENT_CONDITION == "proactive":
            asyncio.create_task(self._proactive_loop())
            asyncio.create_task(self._info_gather_loop())


        # Socket Modeで接続
        handler = AsyncSocketModeHandler(self.app, config.SLACK_APP_TOKEN)
        await handler.start_async()


# =============================================================================
# メイン
# =============================================================================

def main():
    """Botを起動"""
    if not config.SLACK_BOT_TOKEN:
        print("SLACK_BOT_TOKEN が設定されていません")
        print("  .env ファイルに SLACK_BOT_TOKEN=xoxb-... を追加してください")
        return

    if not config.SLACK_APP_TOKEN:
        print("SLACK_APP_TOKEN が設定されていません")
        print("  .env ファイルに SLACK_APP_TOKEN=xapp-... を追加してください")
        return

    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY が設定されていません")
        return

    bot = ProactiveAISlackBot()
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
