<script setup>
import { computed } from 'vue'

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  platform: {
    type: String,
    default: '',
  },
  vdisplay: {
    type: [String, Number],
    default: '1',
  },
})

const resumeWindows = [0, 30000, 60000, 300000]

const driverState = computed(() => {
  const states = {
    0: { label: 'Ready', tone: 'success', detail: 'Virtual display is available for XR sessions.' },
    1: { label: 'Checking', tone: 'neutral', detail: 'Apollo has not reported virtual display health yet.' },
    '-1': { label: 'Not initialized', tone: 'warning', detail: 'The virtual display driver is not initialized.' },
    '-2': { label: 'Update required', tone: 'danger', detail: 'The installed virtual display driver is incompatible.' },
    '-3': { label: 'Needs attention', tone: 'danger', detail: 'The virtual display watchdog is not responding.' },
  }

  return states[String(props.vdisplay)] || states[1]
})

const webAccess = computed(() => {
  const access = props.config.origin_web_ui_allowed || 'lan'
  if (access === 'pc') return 'This PC only'
  if (access === 'wan') return 'Local and remote networks'
  return 'Local network'
})

function isEnabled(key) {
  const value = props.config[key]
  return value === true || value === 1 || value === '1' || value === 'enabled' || value === 'true'
}

function setEnabled(key, event) {
  props.config[key] = event.target.checked ? 'enabled' : 'disabled'
}
</script>

<template>
  <div class="essentials-grid">
    <section class="settings-card settings-card-wide">
      <div class="settings-card-heading">
        <div class="settings-icon"><i class="fas fa-desktop"></i></div>
        <div>
          <p class="settings-eyebrow">Identity</p>
          <h2>This PC</h2>
          <p>How Apollo appears to your XR headset and other devices.</p>
        </div>
      </div>

      <label class="simple-field" for="host-name">
        <span>Computer name</span>
        <input id="host-name" v-model="config.sunshine_name" class="form-control" type="text" placeholder="Apollo" />
      </label>

      <div class="simple-toggle-row">
        <div>
          <strong>Show Apollo on the local network</strong>
          <span>Lets Artemis and other Moonlight clients find this PC automatically.</span>
        </div>
        <label class="form-switch" aria-label="Show Apollo on the local network">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('enable_discovery')"
            @change="setEnabled('enable_discovery', $event)"
          />
        </label>
      </div>

      <div class="simple-toggle-row">
        <div>
          <strong>Allow new devices to pair</strong>
          <span>Turn this off when you do not want Apollo to accept pairing requests.</span>
        </div>
        <label class="form-switch" aria-label="Allow new devices to pair">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('enable_pairing')"
            @change="setEnabled('enable_pairing', $event)"
          />
        </label>
      </div>
    </section>

    <section class="settings-card">
      <div class="settings-card-heading compact">
        <div class="settings-icon"><i class="fas fa-wave-square"></i></div>
        <div>
          <p class="settings-eyebrow">Streaming</p>
          <h2>Session behavior</h2>
        </div>
      </div>

      <div class="simple-toggle-row flush">
        <div>
          <strong>Stream computer audio</strong>
          <span>Send desktop and app audio to the active XR device.</span>
        </div>
        <label class="form-switch" aria-label="Stream computer audio">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('stream_audio')"
            @change="setEnabled('stream_audio', $event)"
          />
        </label>
      </div>

      <label class="simple-field" for="resume-window">
        <span>Reconnect window</span>
        <select id="resume-window" v-model.number="config.session_resume_grace" class="form-select">
          <option v-if="!resumeWindows.includes(Number(config.session_resume_grace))" :value="Number(config.session_resume_grace)">
            Custom ({{ Math.round(Number(config.session_resume_grace) / 1000) }} seconds)
          </option>
          <option :value="0">Do not wait</option>
          <option :value="30000">30 seconds</option>
          <option :value="60000">1 minute</option>
          <option :value="300000">5 minutes</option>
        </select>
        <small>Keeps the single active session ready while a headset reconnects.</small>
      </label>
    </section>

    <section class="settings-card">
      <div class="settings-card-heading compact">
        <div class="settings-icon"><i class="fas fa-shield-alt"></i></div>
        <div>
          <p class="settings-eyebrow">Access</p>
          <h2>Host controls</h2>
        </div>
      </div>

      <div class="status-summary">
        <div>
          <span>Web interface</span>
          <strong>{{ webAccess }}</strong>
        </div>
        <i class="fas fa-wifi" aria-hidden="true"></i>
      </div>
      <p class="card-note">No sign-in is required on this PC or your trusted local network. WAN access still uses credentials if enabled.</p>
      <a class="quiet-link" href="#network">Review network access <i class="fas fa-arrow-right"></i></a>
    </section>

    <section v-if="platform === 'windows'" class="settings-card settings-card-wide">
      <div class="settings-card-heading compact">
        <div class="settings-icon"><i class="fas fa-vr-cardboard"></i></div>
        <div>
          <p class="settings-eyebrow">XR readiness</p>
          <h2>Virtual display</h2>
        </div>
        <span class="status-pill" :class="`status-${driverState.tone}`">
          <span class="status-dot"></span>{{ driverState.label }}
        </span>
      </div>
      <p class="card-note driver-note">{{ driverState.detail }}</p>
      <a v-if="driverState.tone !== 'success'" class="quiet-link" href="./troubleshooting#dd_reset">
        Open diagnostics <i class="fas fa-arrow-right"></i>
      </a>
    </section>

    <section v-if="platform === 'windows'" class="settings-card settings-card-wide tray-setting">
      <div class="simple-toggle-row flush">
        <div>
          <strong>Keep Apollo in the system tray</strong>
          <span>Recommended for quick access and everyday background use.</span>
        </div>
        <label class="form-switch" aria-label="Keep Apollo in the system tray">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('system_tray')"
            @change="setEnabled('system_tray', $event)"
          />
        </label>
      </div>
    </section>
  </div>
</template>

<style scoped>
.essentials-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.settings-card {
  min-width: 0;
  padding: 1.25rem;
  border: 1px solid var(--apollo-border);
  border-radius: 1.1rem;
  background: var(--apollo-surface);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.settings-card-wide {
  grid-column: 1 / -1;
}

.settings-card-heading {
  display: flex;
  align-items: flex-start;
  gap: 0.85rem;
  margin-bottom: 1.2rem;
}

.settings-card-heading.compact {
  align-items: center;
}

.settings-card-heading h2 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
}

.settings-card-heading p:not(.settings-eyebrow) {
  margin: 0.25rem 0 0;
  color: var(--apollo-text-muted);
  font-size: 0.9rem;
}

.settings-icon {
  display: grid;
  flex: 0 0 auto;
  width: 2.35rem;
  height: 2.35rem;
  place-items: center;
  border-radius: 0.8rem;
  color: var(--apollo-accent-hover);
  background: color-mix(in srgb, var(--apollo-accent) 12%, transparent);
}

.settings-eyebrow {
  margin: 0 0 0.15rem;
  color: var(--apollo-text-muted);
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.simple-field {
  display: grid;
  gap: 0.45rem;
  margin-top: 1rem;
  font-weight: 650;
}

.simple-field input,
.simple-field select {
  max-width: 34rem;
}

.simple-field small,
.simple-toggle-row span,
.card-note {
  color: var(--apollo-text-muted);
  font-size: 0.84rem;
  font-weight: 400;
}

.simple-toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.25rem;
  padding-top: 1rem;
  margin-top: 1rem;
  border-top: 1px solid var(--apollo-border);
}

.simple-toggle-row.flush {
  padding-top: 0;
  margin-top: 0;
  border-top: 0;
}

.simple-toggle-row > div {
  display: grid;
  gap: 0.18rem;
}

.form-switch {
  flex: 0 0 auto;
  padding-left: 0;
}

.form-switch .form-check-input {
  width: 2.6rem;
  height: 1.4rem;
  margin: 0;
  cursor: pointer;
}

.status-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 1rem;
  border-radius: 0.9rem;
  background: var(--apollo-surface-muted);
}

.status-summary > div {
  display: grid;
  gap: 0.15rem;
}

.status-summary span {
  color: var(--apollo-text-muted);
  font-size: 0.78rem;
}

.status-summary i {
  color: var(--apollo-accent-hover);
}

.card-note {
  margin: 0.9rem 0;
}

.driver-note {
  margin-bottom: 0;
}

.quiet-link {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.86rem;
  font-weight: 700;
  text-decoration: none;
}

.quiet-link i {
  font-size: 0.72rem;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-left: auto;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 750;
  white-space: nowrap;
  background: var(--apollo-surface-muted);
}

.status-dot {
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: currentColor;
}

.status-success {
  color: #138a55;
  background: color-mix(in srgb, #16a365 13%, transparent);
}

.status-warning {
  color: #a96600;
  background: color-mix(in srgb, #e59a16 15%, transparent);
}

.status-danger {
  color: var(--apollo-danger);
  background: color-mix(in srgb, var(--apollo-danger) 12%, transparent);
}

.tray-setting {
  padding-top: 1rem;
  padding-bottom: 1rem;
}

@media (max-width: 760px) {
  .essentials-grid {
    grid-template-columns: 1fr;
  }

  .settings-card-wide {
    grid-column: auto;
  }

  .settings-card-heading.compact {
    flex-wrap: wrap;
  }

  .status-pill {
    margin-left: 3.2rem;
  }
}
</style>
