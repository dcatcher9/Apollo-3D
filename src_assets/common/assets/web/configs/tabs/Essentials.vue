<script setup>
import { computed, inject } from 'vue'

const $t = inject('i18n').t

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
    0: { label: 'Ready', tone: 'success', detail: 'Virtual display is available for private sessions.' },
    1: { label: 'Checking', tone: 'neutral', detail: 'Sunshine 3D has not reported virtual display health yet.' },
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
  return value === true || value === 1 || value === '1' || value === 'enabled' || value === 'on' || value === 'true'
}

function setEnabled(key, event) {
  props.config[key] = event.target.checked ? 'enabled' : 'disabled'
}

function setOnOff(key, event) {
  props.config[key] = event.target.checked ? 'on' : 'off'
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
          <p>How Sunshine 3D appears to paired devices.</p>
        </div>
      </div>

      <label class="simple-field" for="host-name">
        <span>Computer name</span>
        <input id="host-name" v-model="config.sunshine_name" class="form-control" type="text" placeholder="Sunshine 3D" />
      </label>

      <div class="simple-toggle-row">
        <div>
          <strong>Show Sunshine 3D on the local network</strong>
          <span>Lets Moonlight 3D find this PC automatically.</span>
        </div>
        <label class="form-switch" aria-label="Show Sunshine 3D on the local network">
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
          <span>Turn this off when you do not want Sunshine 3D to accept pairing requests.</span>
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
        <small>Keeps the single active session ready while a device reconnects.</small>
      </label>

      <label class="simple-field" for="host-3d-strength">
        <span>Host 3D strength</span>
        <input
          id="host-3d-strength"
          v-model.number="config.sbs_3d_pop_strength"
          class="form-control"
          type="number"
          min="0.25"
          max="2"
          step="0.05"
        />
        <small>Controls 3D separation in Host 3D. Higher values look deeper but may be harder to focus. Also sets the base strength for new offline conversions.</small>
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

    <section class="settings-card settings-card-wide">
      <div class="settings-card-heading compact">
        <div class="settings-icon"><i class="fas fa-vr-cardboard"></i></div>
        <div>
          <p class="settings-eyebrow">3D readiness</p>
          <h2>Virtual display</h2>
        </div>
        <span class="status-pill" :class="`status-${driverState.tone}`">
          <span class="status-dot"></span>{{ driverState.label }}
        </span>
      </div>
      <p class="card-note driver-note">{{ driverState.detail }}</p>
      <a v-if="driverState.tone !== 'success'" class="quiet-link" href="./troubleshooting">
        Open diagnostics <i class="fas fa-arrow-right"></i>
      </a>
    </section>

    <section class="settings-card settings-card-wide tray-setting">
      <div class="simple-toggle-row flush">
        <div>
          <strong>Keep Sunshine 3D in the system tray</strong>
          <span>Recommended for quick access and everyday background use.</span>
        </div>
        <label class="form-switch" aria-label="Keep Sunshine 3D in the system tray">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('system_tray')"
            @change="setEnabled('system_tray', $event)"
          />
        </label>
      </div>
      <div v-if="platform === 'windows'" class="simple-toggle-row tray-taskbar-repair">
        <div>
          <strong>{{ $t('config.virtual_display_restart_explorer') }}</strong>
          <span>{{ $t('config.virtual_display_restart_explorer_desc') }}</span>
        </div>
        <label class="form-switch" :aria-label="$t('config.virtual_display_restart_explorer')">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('virtual_display_restart_explorer')"
            @change="setOnOff('virtual_display_restart_explorer', $event)"
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
  gap: var(--apollo-space-lg);
}

.settings-card {
  min-width: 0;
  padding: var(--apollo-space-lg);
  border: 1px solid var(--apollo-border);
  border-radius: var(--apollo-radius-card);
  background: var(--apollo-surface);
  box-shadow: var(--apollo-shadow-card);
}

.settings-card-wide {
  grid-column: 1 / -1;
}

.settings-card-heading {
  display: flex;
  align-items: flex-start;
  gap: var(--apollo-space-md);
  margin-bottom: var(--apollo-space-lg);
}

.settings-card-heading.compact {
  align-items: center;
}

.settings-card-heading h2 {
  margin: 0;
  font-size: var(--apollo-text-emphasis);
  font-weight: 700;
}

.settings-card-heading p:not(.settings-eyebrow) {
  margin: var(--apollo-space-xs) 0 0;
  color: var(--apollo-text-secondary);
  font-size: var(--apollo-text-caption);
}

.settings-icon {
  display: grid;
  flex: 0 0 auto;
  width: var(--apollo-icon-inline);
  height: var(--apollo-icon-inline);
  place-items: center;
  border-radius: var(--apollo-radius-control);
  color: var(--apollo-accent);
  background: color-mix(in srgb, var(--apollo-accent-deep) 48%, transparent);
}

.settings-eyebrow {
  margin: 0 0 var(--apollo-space-xs);
  color: var(--apollo-text-disabled);
  font-size: var(--apollo-text-caption);
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.simple-field {
  display: grid;
  gap: var(--apollo-space-sm);
  margin-top: var(--apollo-space-lg);
  font-weight: 650;
}

.simple-field input,
.simple-field select {
  max-width: 34rem;
}

.simple-field small,
.simple-toggle-row span,
.card-note {
  color: var(--apollo-text-secondary);
  font-size: var(--apollo-text-caption);
  font-weight: 400;
}

.simple-toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--apollo-space-lg);
  padding-top: var(--apollo-space-lg);
  margin-top: var(--apollo-space-lg);
  border-top: 1px solid var(--apollo-border);
}

.simple-toggle-row.flush {
  padding-top: 0;
  margin-top: 0;
  border-top: 0;
}

.simple-toggle-row > div {
  display: grid;
  gap: var(--apollo-space-xs);
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
  padding: var(--apollo-space-md) var(--apollo-space-lg);
  border-radius: var(--apollo-radius-control);
  background: var(--apollo-surface-raised);
}

.status-summary > div {
  display: grid;
  gap: var(--apollo-space-xs);
}

.status-summary span {
  color: var(--apollo-text-secondary);
  font-size: var(--apollo-text-caption);
}

.status-summary i {
  color: var(--apollo-accent);
}

.card-note {
  margin: var(--apollo-space-md) 0;
}

.driver-note {
  margin-bottom: 0;
}

.quiet-link {
  display: inline-flex;
  align-items: center;
  gap: var(--apollo-space-sm);
  font-size: var(--apollo-text-caption);
  font-weight: 700;
  text-decoration: none;
}

.quiet-link i {
  font-size: var(--apollo-text-caption);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: var(--apollo-space-sm);
  margin-left: auto;
  padding: var(--apollo-space-xs) var(--apollo-space-sm);
  border-radius: var(--apollo-radius-pill);
  font-size: var(--apollo-text-caption);
  font-weight: 750;
  white-space: nowrap;
  background: var(--apollo-surface-raised);
}

.status-dot {
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: currentColor;
}

.status-success {
  color: var(--apollo-status-ok);
  background: color-mix(in srgb, var(--apollo-status-ok) 13%, transparent);
}

.status-warning {
  color: var(--apollo-status-warn);
  background: color-mix(in srgb, var(--apollo-status-warn) 15%, transparent);
}

.status-danger {
  color: var(--apollo-danger);
  background: color-mix(in srgb, var(--apollo-danger) 12%, transparent);
}

.tray-setting {
  padding-top: var(--apollo-space-lg);
  padding-bottom: var(--apollo-space-lg);
}

.tray-taskbar-repair {
  border-top: 1px solid var(--apollo-border);
  margin-top: var(--apollo-space-md);
  padding-top: var(--apollo-space-md);
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
