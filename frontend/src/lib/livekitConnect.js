import { Room } from 'livekit-client'

/**
 * Connect to a LiveKit room for inbound voice calls.
 *
 * This is intentionally a safe stub:
 * - If `VITE_LIVEKIT_URL` and `VITE_LIVEKIT_TOKEN` are not set, it returns a mock room.
 * - If they are set, it shows the intended LiveKit connection flow.
 *
 * Replace the stub logic with your real backend-provided token/room name.
 */
export async function connectToCall() {
  const livekitUrl = import.meta.env.VITE_LIVEKIT_URL
  const token = import.meta.env.VITE_LIVEKIT_TOKEN

  if (!livekitUrl || !token) {
    // Mock room that matches the minimal interface we use in the UI.
    return {
      disconnect: () => {},
    }
  }

  // Real connection path (shown for integration; adjust as needed for your backend).
  const room = new Room()
  // TODO: Replace with actual connect parameters (room name / websocket URL / token).
  // Example (may differ by LiveKit SDK version):
  // await room.connect(livekitUrl, token)
  // return room

  // If you enable env vars without updating the connect parameters, this will throw.
  // Keep it as a clearly-marked placeholder.
  throw new Error(
    'LiveKit connection stub enabled. Set VITE_LIVEKIT_URL and VITE_LIVEKIT_TOKEN AND update connect parameters in src/lib/livekitConnect.js.'
  )
}

