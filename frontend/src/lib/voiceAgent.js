/**
 * Browser client for Backend/websocket voice interviews.
 *
 * Protocol (see Backend/tests/test_live_microphone.py):
 *   - Mic → binary PCM16 LE mono @ 24 kHz
 *   - Server → turn / tts_start / PCM chunks / tts_end / turn_playback_end
 *   - Client → {"type":"playback_done"} after the full turn has played
 */

const TARGET_MIC_RATE = 24000
const MIC_BUFFER_SIZE = 4096

function defaultWsUrl() {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/voice`
}

function floatToPcm16(float32) {
  const out = new Int16Array(float32.length)

  for (let i = 0; i < float32.length; i += 1) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }

  return out
}

function resampleLinear(input, fromRate, toRate) {
  if (fromRate === toRate) {
    return input
  }

  const ratio = fromRate / toRate
  const newLength = Math.max(1, Math.round(input.length / ratio))
  const output = new Float32Array(newLength)

  for (let i = 0; i < newLength; i += 1) {
    const srcIndex = i * ratio
    const i0 = Math.floor(srcIndex)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const t = srcIndex - i0
    output[i] = input[i0] * (1 - t) + input[i1] * t
  }

  return output
}

function pcm16ToFloat32(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const samples = new Float32Array(bytes.byteLength / 2)

  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = view.getInt16(i * 2, true) / 32768
  }

  return samples
}

function rmsLevel(samples) {
  if (!samples.length) return 0

  let sum = 0
  for (let i = 0; i < samples.length; i += 1) {
    sum += samples[i] * samples[i]
  }

  return Math.min(1, Math.sqrt(sum / samples.length) * 4)
}

export class VoiceAgentSession {
  /**
   * @param {object} handlers
   * @param {(level: number) => void} [handlers.onLevel]
   * @param {(speaking: boolean) => void} [handlers.onBotSpeaking]
   * @param {(text: string) => void} [handlers.onTranscript]
   * @param {(info: {mode?: string, questionId?: string|null}) => void} [handlers.onTurn]
   * @param {(payload: {candidate: object, endedEarly: boolean}) => void} [handlers.onComplete]
   * @param {(message: string, stage?: string) => void} [handlers.onError]
   * @param {() => void} [handlers.onDisconnected]
   */
  constructor(handlers = {}) {
    this.handlers = handlers
    this.ws = null
    this.mediaStream = null
    this.audioContext = null
    this.processor = null
    this.sourceNode = null
    this.silentGain = null

    this.botSpeaking = false
    this.ttsSampleRate = TARGET_MIC_RATE
    this.ttsChunks = []
    this.playQueue = Promise.resolve()
    this.closed = false
  }

  async start() {
    this.closed = false

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    })

    this.audioContext = new AudioContext()
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume()
    }

    await this.#openSocket()
    this.#startMic()
  }

  stop() {
    this.closed = true
    this.#setBotSpeaking(false)

    if (this.processor) {
      this.processor.onaudioprocess = null
      try {
        this.processor.disconnect()
      } catch {
        /* ignore */
      }
      this.processor = null
    }

    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect()
      } catch {
        /* ignore */
      }
      this.sourceNode = null
    }

    if (this.silentGain) {
      try {
        this.silentGain.disconnect()
      } catch {
        /* ignore */
      }
      this.silentGain = null
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop())
      this.mediaStream = null
    }

    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      try {
        this.ws.close()
      } catch {
        /* ignore */
      }
    }
    this.ws = null

    if (this.audioContext) {
      this.audioContext.close().catch(() => {})
      this.audioContext = null
    }
  }

  #openSocket() {
    const url = defaultWsUrl()

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url)
      ws.binaryType = 'arraybuffer'
      this.ws = ws

      const onOpen = () => {
        cleanup()
        resolve()
      }

      const onError = () => {
        cleanup()
        reject(new Error(`Unable to connect to voice agent at ${url}`))
      }

      const cleanup = () => {
        ws.removeEventListener('open', onOpen)
        ws.removeEventListener('error', onError)
      }

      ws.addEventListener('open', onOpen)
      ws.addEventListener('error', onError)
      ws.addEventListener('message', (event) => this.#onMessage(event))
      ws.addEventListener('close', () => {
        if (!this.closed) {
          this.handlers.onDisconnected?.()
        }
      })
    })
  }

  #startMic() {
    const ctx = this.audioContext
    this.sourceNode = ctx.createMediaStreamSource(this.mediaStream)
    this.processor = ctx.createScriptProcessor(MIC_BUFFER_SIZE, 1, 1)
    this.silentGain = ctx.createGain()
    this.silentGain.gain.value = 0

    this.processor.onaudioprocess = (event) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return

      const input = event.inputBuffer.getChannelData(0)
      this.handlers.onLevel?.(rmsLevel(input))

      // Keep sending while the bot speaks — the server drops muted audio
      // and draining the socket keeps the connection healthy.
      const resampled = resampleLinear(
        input,
        ctx.sampleRate,
        TARGET_MIC_RATE,
      )
      const pcm = floatToPcm16(resampled)
      this.ws.send(pcm.buffer)
    }

    this.sourceNode.connect(this.processor)
    this.processor.connect(this.silentGain)
    this.silentGain.connect(ctx.destination)
  }

  #setBotSpeaking(value) {
    this.botSpeaking = value
    this.handlers.onBotSpeaking?.(value)
  }

  #onMessage(event) {
    if (typeof event.data !== 'string') {
      if (this.ttsChunks) {
        this.ttsChunks.push(new Uint8Array(event.data))
      }
      return
    }

    let data
    try {
      data = JSON.parse(event.data)
    } catch {
      return
    }

    switch (data.type) {
      case 'turn':
        this.handlers.onTurn?.({
          mode: data.mode,
          questionId: data.question_id ?? null,
        })
        break

      case 'tts_start':
        this.#setBotSpeaking(true)
        this.ttsSampleRate = data.sample_rate || TARGET_MIC_RATE
        this.ttsChunks = []
        break

      case 'tts_end':
        this.#enqueuePlayback(this.ttsChunks, this.ttsSampleRate)
        this.ttsChunks = []
        break

      case 'turn_playback_end':
        this.playQueue = this.playQueue.then(async () => {
          this.#sendJson({ type: 'playback_done' })
          this.#setBotSpeaking(false)
        })
        break

      case 'transcript':
        if (data.text) {
          this.handlers.onTranscript?.(data.text)
        }
        break

      case 'speech_started':
      case 'speech_ended':
        break

      case 'interview_complete':
        this.handlers.onComplete?.({
          candidate: data.candidate || {},
          endedEarly: Boolean(data.ended_early),
        })
        break

      case 'error':
        this.handlers.onError?.(
          data.message || 'Unknown agent error',
          data.stage,
        )
        break

      default:
        break
    }
  }

  #enqueuePlayback(chunks, sampleRate) {
    if (!chunks.length) return

    const total = chunks.reduce((n, c) => n + c.byteLength, 0)
    const merged = new Uint8Array(total)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.byteLength
    }

    this.playQueue = this.playQueue
      .then(() => this.#playPcm16(merged, sampleRate))
      .catch((err) => {
        console.error('[TTS] Playback failed', err)
      })
  }

  async #playPcm16(bytes, sampleRate) {
    if (!this.audioContext || this.closed) return

    const ctx = this.audioContext
    if (ctx.state === 'suspended') {
      await ctx.resume()
    }

    const samples = pcm16ToFloat32(bytes)
    if (!samples.length) return

    const buffer = ctx.createBuffer(1, samples.length, sampleRate)
    buffer.copyToChannel(samples, 0)

    await new Promise((resolve) => {
      const source = ctx.createBufferSource()
      source.buffer = buffer
      source.connect(ctx.destination)
      source.onended = () => resolve()
      source.start()
    })
  }

  #sendJson(payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload))
    }
  }
}

export { defaultWsUrl, TARGET_MIC_RATE }
