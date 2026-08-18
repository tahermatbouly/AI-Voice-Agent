import axios from 'axios'
import { getMockCallById, getMockCalls } from '../mocks/mockCalls'

const client = axios.create({
  // Keep default baseURL so the app can be served behind a proxy.
  timeout: 8000,
})

export async function fetchCalls() {
  try {
    const res = await client.get('/api/calls')
    // Backend is expected to return an array of call objects.
    return res.data
  } catch (err) {
    // In local dev, the API may not be running yet.
    return getMockCalls()
  }
}

export async function fetchCallById(callId) {
  try {
    const res = await client.get(`/api/calls/${callId}`)
    // Backend is expected to return a call object.
    return res.data
  } catch (err) {
    return getMockCallById(callId)
  }
}

