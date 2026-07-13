<script setup>
import {onMounted, onUnmounted, ref} from 'vue'

const devices = ref([])
const error = ref('')
let refreshTimer = null

const refresh = async () => {
  try {
    const response = await fetch('./api/ar-glasses', {credentials: 'include'})
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const body = await response.json()
    devices.value = body.devices || []
    error.value = ''
  } catch (reason) {
    error.value = `Could not read AR display status: ${reason}`
  }
}

const decide = async (device, decision) => {
  try {
    const response = await fetch('./api/ar-glasses', {
      credentials: 'include',
      headers: {'Content-Type': 'application/json'},
      method: 'POST',
      body: JSON.stringify({id: device.id, decision}),
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    await refresh()
  } catch (reason) {
    error.value = `Could not save AR display decision: ${reason}`
  }
}

onMounted(() => {
  refresh()
  refreshTimer = setInterval(refresh, 1000)
})

onUnmounted(() => {
  if (refreshTimer !== null) clearInterval(refreshTimer)
})
</script>

<template>
  <div id="ar-glasses" class="config-page">
    <h2>AR displays</h2>
    <p class="text-body-secondary">
      Apollo automatically recognizes known AR-glasses names. For any other newly seen monitor,
      choose whether Apollo may use it as a local AR presentation output. Decisions are saved by
      monitor model and reused after reconnecting or changing ports.
    </p>

    <div v-if="error" class="alert alert-danger">{{ error }}</div>
    <div v-if="devices.length === 0" class="alert alert-secondary">
      No monitors have been discovered yet.
    </div>

    <div v-for="device in devices" :key="device.id" class="card mb-3">
      <div class="card-body d-flex flex-wrap align-items-center justify-content-between gap-3">
        <div>
          <h5 class="card-title mb-1">{{ device.name }}</h5>
          <div class="font-monospace small">{{ device.id }}</div>
          <span class="badge mt-2" :class="device.connected ? 'text-bg-success' : 'text-bg-secondary'">
            {{ device.connected ? 'Connected' : 'Not connected' }}
          </span>
          <span v-if="device.autoDetected" class="badge text-bg-info ms-2">Recognized automatically</span>
        </div>
        <div class="btn-group" role="group" :aria-label="`AR display decision for ${device.name}`">
          <button
            type="button"
            class="btn"
            :class="device.decision === 'approved' ? 'btn-success' : 'btn-outline-success'"
            @click="decide(device, 'approved')">
            Use as AR display
          </button>
          <button
            type="button"
            class="btn"
            :class="device.decision === 'rejected' ? 'btn-secondary' : 'btn-outline-secondary'"
            @click="decide(device, 'rejected')">
            Not an AR display
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
