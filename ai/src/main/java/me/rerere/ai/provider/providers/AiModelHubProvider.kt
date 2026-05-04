package me.rerere.ai.provider.providers

import android.content.Context
import com.ai_model_hub.sdk.AiHubClient
import com.ai_model_hub.sdk.ConnectionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import me.rerere.ai.core.MessageRole
import me.rerere.ai.core.TokenUsage
import me.rerere.ai.provider.ImageGenerationParams
import me.rerere.ai.provider.Model
import me.rerere.ai.provider.Provider
import me.rerere.ai.provider.ProviderSetting
import me.rerere.ai.provider.TextGenerationParams
import me.rerere.ai.ui.ImageGenerationResult
import me.rerere.ai.ui.MessageChunk
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessageChoice
import me.rerere.ai.ui.UIMessagePart
import java.util.concurrent.CountDownLatch
import kotlin.uuid.Uuid

/**
 * Provider implementation that delegates to the AiModelHub on-device inference service
 * via AIDL binding. The AiModelHub app (com.ai_model_hub) must be installed on the device.
 *
 * Models run entirely on-device using Google's LiteRT LM engine — no network required.
 */
class AiModelHubProvider(private val context: Context) : Provider<ProviderSetting.AiModelHub> {

    private val client by lazy { AiHubClient(context) }

    override suspend fun listModels(providerSetting: ProviderSetting.AiModelHub): List<Model> {
        return providerSetting.models
    }

    override suspend fun generateText(
        providerSetting: ProviderSetting.AiModelHub,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): MessageChunk {
        var result = ""
        streamText(providerSetting, messages, params).collect { chunk ->
            val text = chunk.choices.firstOrNull()?.delta?.parts
                ?.filterIsInstance<UIMessagePart.Text>()
                ?.joinToString("") { it.text } ?: ""
            result += text
        }
        val responseMsg = UIMessage(
            role = MessageRole.ASSISTANT,
            parts = listOf(UIMessagePart.Text(result)),
        )
        return MessageChunk(
            id = Uuid.random().toString(),
            model = params.model.modelId,
            choices = listOf(
                UIMessageChoice(
                    index = 0,
                    delta = null,
                    message = responseMsg,
                    finishReason = "stop",
                )
            ),
            usage = TokenUsage(),
        )
    }

    override suspend fun streamText(
        providerSetting: ProviderSetting.AiModelHub,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): Flow<MessageChunk> {
        val modelName = params.model.modelId

        // Connect on main thread (service connection callbacks are delivered on main looper)
        withContext(Dispatchers.Main) { client.connect() }

        // Wait for Connected or Error state (up to 15 seconds)
        val connState = withTimeout(15_000L) {
            client.connectionState.first { it is ConnectionState.Connected || it is ConnectionState.Error }
        }
        if (connState is ConnectionState.Error) error(connState.message)

        // Load model if not already loaded, then reset session
        withContext(Dispatchers.IO) {
            if (!client.isModelLoaded(modelName)) {
                var loadError: String? = null
                val latch = CountDownLatch(1)
                client.loadModel(modelName) { err ->
                    loadError = err.takeIf { it.isNotEmpty() }
                    latch.countDown()
                }
                latch.await()
                loadError?.let { error("Failed to load model '$modelName': $it") }
            }
            client.resetSession(modelName)
        }

        val prompt = formatPrompt(messages)

        return client.sendMessage(modelName, prompt).map { token ->
            MessageChunk(
                id = Uuid.random().toString(),
                model = modelName,
                choices = listOf(
                    UIMessageChoice(
                        index = 0,
                        delta = UIMessage(
                            role = MessageRole.ASSISTANT,
                            parts = listOf(UIMessagePart.Text(token)),
                        ),
                        message = null,
                        finishReason = null,
                    )
                ),
            )
        }.flowOn(Dispatchers.IO)
    }

    override suspend fun generateImage(
        providerSetting: ProviderSetting,
        params: ImageGenerationParams,
    ): ImageGenerationResult {
        error("Image generation is not supported by AiModelHub")
    }

    /**
     * Formats the conversation history into a single prompt string suitable for
     * AiModelHub's single-turn sendMessage API.
     */
    private fun formatPrompt(messages: List<UIMessage>): String {
        val sb = StringBuilder()
        messages.forEach { msg ->
            val textContent = msg.parts
                .filterIsInstance<UIMessagePart.Text>()
                .joinToString("") { it.text }
                .trim()
            if (textContent.isEmpty()) return@forEach

            when (msg.role) {
                MessageRole.SYSTEM -> sb.appendLine("[System: $textContent]")
                MessageRole.USER -> sb.appendLine("Human: $textContent")
                MessageRole.ASSISTANT -> sb.appendLine("Assistant: $textContent")
                MessageRole.TOOL -> { /* skip tool messages */ }
            }
        }
        sb.append("Assistant:")
        return sb.toString()
    }
}
