<script setup>
import {computed, inject, onMounted, onUnmounted, ref} from 'vue'
import { getJson, postJson } from '../../api'
import { startVisibilityPolling } from '../../visibility-polling'
import { settingEnabled } from '../settingEnabled'

const $t = inject('i18n').t

const props = defineProps({
  config: {
    type: Object,
    default: null,
  },
  embedded: {
    type: Boolean,
    default: false,
  },
})

const devices = ref([])
const error = ref('')
const savingId = ref('')
let stopPolling = null

const virtualDisplayOnlyEnabled = computed(() => {
  const value = props.config?.local_ar_virtual_display_only
  if (value === undefined || value === null || value === '') return true
  return settingEnabled(value)
})

const setVirtualDisplayOnly = event => {
  if (props.config) {
    props.config.local_ar_virtual_display_only = event.target.checked ? 'enabled' : 'disabled'
  }
}

const visibleDevices = computed(() => {
  if (!props.embedded) return devices.value
  return devices.value.filter(device => device.decision === 'approved')
})

const refresh = async () => {
  try {
    const body = await getJson('./api/ar-glasses')
    devices.value = body.devices || []
    error.value = ''
  } catch (reason) {
    error.value = $t('config.ar_status_read_failed', {reason: String(reason)})
  }
}

const decide = async (device, decision) => {
  savingId.value = device.id
  try {
    await postJson('./api/ar-glasses', {id: device.id, decision})
    await refresh()
  } catch (reason) {
    error.value = $t('config.ar_decision_save_failed', {reason: String(reason)})
  } finally {
    savingId.value = ''
  }
}

onMounted(() => {
  stopPolling = startVisibilityPolling(refresh, 3000)
})

onUnmounted(() => {
  stopPolling?.()
})
</script>

<template>
  <section
    v-if="!props.embedded || visibleDevices.length > 0 || error"
    id="ar-glasses"
    class="config-page ar-glasses-section"
    :class="{'ar-glasses-embedded': props.embedded}"
    aria-labelledby="ar-glasses-heading">
    <header class="ar-glasses-heading">
      <div>
        <h2 id="ar-glasses-heading">{{ $t('config.ar_title') }}</h2>
        <p class="text-body-secondary mb-0">{{ $t('config.ar_intro') }}</p>
      </div>
    </header>

    <div v-if="error" class="alert alert-danger">{{ error }}</div>
    <div v-if="!props.embedded && props.config" class="ar-setting-card">
      <div>
        <strong>{{ $t('config.local_ar_virtual_display_only') }}</strong>
        <span>{{ $t('config.local_ar_virtual_display_only_desc') }}</span>
      </div>
      <label class="form-switch" :aria-label="$t('config.local_ar_virtual_display_only')">
        <input
          class="form-check-input"
          type="checkbox"
          :checked="virtualDisplayOnlyEnabled"
          @change="setVirtualDisplayOnly"
        />
      </label>
    </div>
    <div v-if="!props.embedded && devices.length === 0" class="ar-empty-state">
      <i class="fa-solid fa-glasses"></i>
      {{ $t('config.ar_empty') }}
    </div>

    <div class="ar-device-grid">
      <article v-for="device in visibleDevices" :key="device.id" class="ar-display-card">
        <div class="ar-display-header">
          <div class="ar-display-icon"><i class="fa-solid fa-glasses"></i></div>
          <div>
            <h3>{{ device.name || $t('config.ar_unknown_display') }}</h3>
            <div class="ar-display-state">
              <span class="ar-status-dot" :class="{offline: !device.connected}"></span>
              {{ device.connected ? $t('config.ar_connected') : $t('config.ar_not_connected') }}
            </div>
          </div>
          <span v-if="device.autoDetected" class="badge rounded-pill text-bg-info ms-auto">{{ $t('config.ar_recognized') }}</span>
        </div>

        <div class="ar-mode-guide">
          <div>
            <strong>1920 × 1080</strong>
            <span>2D</span>
          </div>
          <div>
            <strong>3840 × 1080</strong>
            <span>{{ $t('config.ar_mode_ai_3d') }}</span>
          </div>
        </div>

        <div class="ar-display-actions" role="group" :aria-label="$t('config.ar_decision_group', {name: device.name})">
          <button
            type="button"
            class="btn"
            :class="device.decision === 'approved' ? 'btn-success' : 'btn-outline-success'"
            :disabled="savingId === device.id"
            @click="decide(device, 'approved')">
            <span v-if="savingId === device.id" class="spinner-border spinner-border-sm" aria-hidden="true"></span>
            <i v-else class="fa-solid fa-check"></i>
            {{ $t('config.ar_use') }}
          </button>
          <button
            type="button"
            class="btn"
            :class="device.decision === 'rejected' ? 'btn-secondary' : 'btn-outline-secondary'"
            :disabled="savingId === device.id"
            @click="decide(device, 'rejected')">
            {{ $t('config.ar_ignore') }}
          </button>
        </div>

        <details class="ar-technical-details">
          <summary>{{ $t('config.ar_technical_details') }}</summary>
          <div class="font-monospace small text-body-secondary mt-2">{{ device.id }}</div>
          <div class="small text-body-secondary mt-1">
            {{ device.autoDetected ? $t('config.ar_auto_detected') : $t('config.ar_choice_saved') }}
          </div>
        </details>
      </article>
    </div>
  </section>
</template>

<style scoped>
.ar-glasses-embedded {
  border-top: 1px solid var(--apollo-border);
  padding-top: calc(var(--apollo-space-xl) * 2);
}

.ar-glasses-heading {
  align-items: end;
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--apollo-space-lg);
}

.ar-glasses-heading h2 {
  font-size: var(--apollo-text-title);
  font-weight: 680;
  letter-spacing: -.025em;
  margin: 0;
}

.ar-glasses-heading p {
  margin-top: var(--apollo-space-sm);
  max-width: 46rem;
}

.ar-device-grid {
  display: grid;
  gap: var(--apollo-space-lg);
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.ar-setting-card {
  align-items: center;
  background: var(--apollo-surface);
  border: 1px solid var(--apollo-border);
  border-radius: var(--apollo-radius-card);
  display: flex;
  gap: var(--apollo-space-lg);
  justify-content: space-between;
  margin-bottom: var(--apollo-space-lg);
  padding: var(--apollo-space-lg);
}

.ar-setting-card > div {
  display: flex;
  flex-direction: column;
  gap: var(--apollo-space-xs);
}

.ar-setting-card span {
  color: var(--apollo-text-secondary);
  font-size: var(--apollo-text-caption);
  max-width: 52rem;
}

.ar-setting-card .form-switch {
  flex: 0 0 auto;
  margin: 0;
}

.ar-display-card {
  background: var(--apollo-surface);
  border: 1px solid var(--apollo-border);
  border-radius: var(--apollo-radius-card);
  padding: var(--apollo-space-lg);
}

.ar-display-header {
  align-items: flex-start;
  display: flex;
  gap: var(--apollo-space-md);
}

.ar-display-icon {
  align-items: center;
  background: color-mix(in srgb, var(--apollo-accent-deep) 48%, transparent);
  border: 1px solid color-mix(in srgb, var(--apollo-accent) 30%, transparent);
  border-radius: var(--apollo-radius-control);
  color: var(--apollo-accent);
  display: inline-flex;
  flex: 0 0 auto;
  height: var(--apollo-icon-tile);
  justify-content: center;
  width: var(--apollo-icon-tile);
}

.ar-display-header h3 {
  font-size: var(--apollo-text-emphasis);
  font-weight: 680;
  margin: var(--apollo-space-xs) 0 var(--apollo-space-sm);
}

.ar-display-state {
  align-items: center;
  color: var(--apollo-text-secondary);
  display: flex;
  font-size: var(--apollo-text-caption);
  gap: var(--apollo-space-sm);
}

.ar-status-dot {
  background: var(--apollo-status-ok);
  border-radius: 50%;
  box-shadow: 0 0 0 var(--apollo-space-xs) color-mix(in srgb, var(--apollo-status-ok) 18%, transparent);
  height: .55rem;
  width: .55rem;
}

.ar-status-dot.offline {
  background: var(--apollo-text-disabled);
  box-shadow: none;
  opacity: .55;
}

.ar-mode-guide {
  border-bottom: 1px solid var(--apollo-border);
  border-top: 1px solid var(--apollo-border);
  display: grid;
  grid-template-columns: 1fr 1fr;
  margin: var(--apollo-space-lg) 0;
  padding: var(--apollo-space-md) 0;
}

.ar-mode-guide > div {
  display: flex;
  flex-direction: column;
  gap: var(--apollo-space-xs);
}

.ar-mode-guide > div + div {
  border-left: 1px solid var(--apollo-border);
  padding-left: var(--apollo-space-lg);
}

.ar-mode-guide strong {
  font-size: var(--apollo-text-caption);
}

.ar-mode-guide span {
  color: var(--apollo-text-secondary);
  font-size: var(--apollo-text-caption);
}

.ar-display-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--apollo-space-sm);
}

.ar-display-actions .btn {
  align-items: center;
  border-radius: var(--apollo-radius-control);
  display: inline-flex;
  gap: var(--apollo-space-sm);
  min-height: var(--apollo-control-compact);
}

.ar-technical-details {
  border-top: 1px solid var(--apollo-border);
  margin-top: var(--apollo-space-lg);
  padding-top: var(--apollo-space-md);
}

.ar-technical-details summary {
  color: var(--apollo-text-secondary);
  cursor: pointer;
  font-size: var(--apollo-text-caption);
}

.ar-empty-state {
  align-items: center;
  border: 1px dashed var(--apollo-border-strong);
  border-radius: var(--apollo-radius-card);
  color: var(--apollo-text-secondary);
  display: flex;
  gap: var(--apollo-space-sm);
  padding: var(--apollo-space-lg);
}

@media (max-width: 768px) {
  .ar-device-grid {
    grid-template-columns: 1fr;
  }
}
</style>
