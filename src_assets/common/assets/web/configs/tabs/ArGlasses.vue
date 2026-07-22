<script setup>
import {onMounted, onUnmounted, ref} from 'vue'

const props = defineProps({
  embedded: {
    type: Boolean,
    default: false,
  },
})

const devices = ref([])
const error = ref('')
const savingId = ref('')
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
  savingId.value = device.id
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
  } finally {
    savingId.value = ''
  }
}

onMounted(() => {
  refresh()
  refreshTimer = setInterval(refresh, 3000)
})

onUnmounted(() => {
  if (refreshTimer !== null) clearInterval(refreshTimer)
})
</script>

<template>
  <section
    v-if="!props.embedded || devices.length > 0 || error"
    id="ar-glasses"
    class="config-page ar-glasses-section"
    :class="{'ar-glasses-embedded': props.embedded}"
    aria-labelledby="ar-glasses-heading">
    <header class="ar-glasses-heading">
      <div>
        <h2 id="ar-glasses-heading">Local AR glasses</h2>
        <p class="text-body-secondary mb-0">
          Apollo can present directly to glasses connected as a Windows display. Remote XR streaming
          temporarily takes priority over local presentation.
        </p>
      </div>
    </header>

    <div v-if="error" class="alert alert-danger">{{ error }}</div>
    <div v-if="!props.embedded && devices.length === 0" class="ar-empty-state">
      <i class="fa-solid fa-glasses"></i>
      No monitors have been discovered yet.
    </div>

    <div class="ar-device-grid">
      <article v-for="device in devices" :key="device.id" class="ar-display-card">
        <div class="ar-display-header">
          <div class="ar-display-icon"><i class="fa-solid fa-glasses"></i></div>
          <div>
            <h3>{{ device.name || 'Unknown display' }}</h3>
            <div class="ar-display-state">
              <span class="ar-status-dot" :class="{offline: !device.connected}"></span>
              {{ device.connected ? 'Connected' : 'Not connected' }}
            </div>
          </div>
          <span v-if="device.autoDetected" class="badge rounded-pill text-bg-info ms-auto">Recognized</span>
        </div>

        <div class="ar-mode-guide">
          <div>
            <strong>1920 × 1080</strong>
            <span>2D</span>
          </div>
          <div>
            <strong>3840 × 1080</strong>
            <span>AI 3D</span>
          </div>
        </div>

        <div class="ar-display-actions" role="group" :aria-label="`AR glasses decision for ${device.name}`">
          <button
            type="button"
            class="btn"
            :class="device.decision === 'approved' ? 'btn-success' : 'btn-outline-success'"
            :disabled="savingId === device.id"
            @click="decide(device, 'approved')">
            <span v-if="savingId === device.id" class="spinner-border spinner-border-sm" aria-hidden="true"></span>
            <i v-else class="fa-solid fa-check"></i>
            Use as AR glasses
          </button>
          <button
            type="button"
            class="btn"
            :class="device.decision === 'rejected' ? 'btn-secondary' : 'btn-outline-secondary'"
            :disabled="savingId === device.id"
            @click="decide(device, 'rejected')">
            Ignore
          </button>
        </div>

        <details class="ar-technical-details">
          <summary>Technical details</summary>
          <div class="font-monospace small text-body-secondary mt-2">{{ device.id }}</div>
          <div class="small text-body-secondary mt-1">
            {{ device.autoDetected ? 'Apollo recognized this model automatically.' : 'This choice is saved for this monitor model.' }}
          </div>
        </details>
      </article>
    </div>
  </section>
</template>

<style scoped>
.ar-glasses-section {
  --ar-accent: #f4c542;
}

.ar-glasses-embedded {
  border-top: 1px solid var(--bs-border-color);
  padding-top: 2.75rem;
}

.ar-glasses-heading {
  align-items: end;
  display: flex;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.ar-glasses-heading h2 {
  font-size: 1.4rem;
  font-weight: 680;
  letter-spacing: -.025em;
  margin: 0;
}

.ar-glasses-heading p {
  margin-top: .35rem;
  max-width: 46rem;
}

.ar-device-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.ar-display-card {
  background: color-mix(in srgb, var(--bs-body-bg) 86%, var(--bs-secondary-bg));
  border: 1px solid var(--bs-border-color);
  border-radius: 1.25rem;
  padding: 1.25rem;
}

.ar-display-header {
  align-items: flex-start;
  display: flex;
  gap: .9rem;
}

.ar-display-icon {
  align-items: center;
  background: color-mix(in srgb, var(--ar-accent) 16%, transparent);
  border: 1px solid color-mix(in srgb, var(--ar-accent) 30%, transparent);
  border-radius: .85rem;
  color: var(--ar-accent);
  display: inline-flex;
  flex: 0 0 auto;
  height: 2.75rem;
  justify-content: center;
  width: 2.75rem;
}

.ar-display-header h3 {
  font-size: 1.05rem;
  font-weight: 680;
  margin: .15rem 0 .35rem;
}

.ar-display-state {
  align-items: center;
  color: var(--bs-secondary-color);
  display: flex;
  font-size: .82rem;
  gap: .45rem;
}

.ar-status-dot {
  background: var(--bs-success);
  border-radius: 50%;
  box-shadow: 0 0 0 .24rem color-mix(in srgb, var(--bs-success) 18%, transparent);
  height: .55rem;
  width: .55rem;
}

.ar-status-dot.offline {
  background: var(--bs-secondary-color);
  box-shadow: none;
  opacity: .55;
}

.ar-mode-guide {
  border-bottom: 1px solid var(--bs-border-color);
  border-top: 1px solid var(--bs-border-color);
  display: grid;
  grid-template-columns: 1fr 1fr;
  margin: 1rem 0;
  padding: .85rem 0;
}

.ar-mode-guide > div {
  display: flex;
  flex-direction: column;
  gap: .1rem;
}

.ar-mode-guide > div + div {
  border-left: 1px solid var(--bs-border-color);
  padding-left: 1rem;
}

.ar-mode-guide strong {
  font-size: .88rem;
}

.ar-mode-guide span {
  color: var(--bs-secondary-color);
  font-size: .78rem;
}

.ar-display-actions {
  display: flex;
  flex-wrap: wrap;
  gap: .6rem;
}

.ar-display-actions .btn {
  align-items: center;
  border-radius: .75rem;
  display: inline-flex;
  gap: .45rem;
  min-height: 42px;
}

.ar-technical-details {
  border-top: 1px solid var(--bs-border-color);
  margin-top: 1rem;
  padding-top: .75rem;
}

.ar-technical-details summary {
  color: var(--bs-secondary-color);
  cursor: pointer;
  font-size: .82rem;
}

.ar-empty-state {
  align-items: center;
  border: 1px dashed var(--bs-border-color);
  border-radius: 1rem;
  color: var(--bs-secondary-color);
  display: flex;
  gap: .65rem;
  padding: 1.25rem;
}

@media (max-width: 768px) {
  .ar-device-grid {
    grid-template-columns: 1fr;
  }
}
</style>
