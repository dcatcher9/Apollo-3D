<script setup>
import { computed, inject } from 'vue'
import Host3dSubtitleApps from './Host3dSubtitleApps.vue'
import { virtualDisplayStatus } from '../../virtual-display-status'

const $t = inject('i18n').t

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  vdisplay: {
    type: [String, Number],
    default: '1',
  },
})

const resumeWindows = [0, 30000, 60000, 300000]

const driverState = computed(() => virtualDisplayStatus(props.vdisplay))

const webAccess = computed(() => {
  const access = props.config.origin_web_ui_allowed || 'lan'
  if (access === 'pc') return $t('config.essentials_web_access_pc')
  if (access === 'wan') return $t('config.essentials_web_access_wan')
  return $t('config.essentials_web_access_lan')
})

function isEnabled(key, defaultValue = false) {
  const value = props.config[key]
  if (value === undefined || value === null || value === '') return defaultValue
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
          <p class="settings-eyebrow">{{ $t('config.essentials_identity_eyebrow') }}</p>
          <h2>{{ $t('config.essentials_this_pc_title') }}</h2>
          <p>{{ $t('config.essentials_this_pc_desc') }}</p>
        </div>
      </div>

      <label class="simple-field" for="host-name">
        <span>{{ $t('config.essentials_computer_name') }}</span>
        <input id="host-name" v-model="config.sunshine_name" class="form-control" type="text" placeholder="Sunshine 3D" />
      </label>

      <div class="simple-toggle-row">
        <div>
          <strong>{{ $t('config.essentials_discovery') }}</strong>
          <span>{{ $t('config.essentials_discovery_desc') }}</span>
        </div>
        <label class="form-switch" :aria-label="$t('config.essentials_discovery')">
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
          <strong>{{ $t('config.essentials_pairing') }}</strong>
          <span>{{ $t('config.essentials_pairing_desc') }}</span>
        </div>
        <label class="form-switch" :aria-label="$t('config.essentials_pairing')">
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
          <p class="settings-eyebrow">{{ $t('config.essentials_streaming_eyebrow') }}</p>
          <h2>{{ $t('config.essentials_session_title') }}</h2>
        </div>
      </div>

      <label class="simple-field" for="resume-window">
        <span>{{ $t('config.essentials_reconnect_window') }}</span>
        <select id="resume-window" v-model.number="config.session_resume_grace" class="form-select">
          <option v-if="!resumeWindows.includes(Number(config.session_resume_grace))" :value="Number(config.session_resume_grace)">
            {{ $t('config.essentials_reconnect_custom', { seconds: Math.round(Number(config.session_resume_grace) / 1000) }) }}
          </option>
          <option :value="0">{{ $t('config.essentials_reconnect_none') }}</option>
          <option :value="30000">{{ $t('config.essentials_reconnect_30s') }}</option>
          <option :value="60000">{{ $t('config.essentials_reconnect_1min') }}</option>
          <option :value="300000">{{ $t('config.essentials_reconnect_5min') }}</option>
        </select>
        <small>{{ $t('config.essentials_reconnect_window_desc') }}</small>
      </label>


      <label class="simple-field" for="host-3d-strength">
        <span>{{ $t('config.sbs_3d_pop_strength') }}</span>
        <input
          id="host-3d-strength"
          v-model.number="config.sbs_3d_pop_strength"
          class="form-control"
          type="number"
          min="0.25"
          max="2"
          step="0.05"
        />
        <small>{{ $t('config.essentials_strength_desc') }}</small>
      </label>

      <Host3dSubtitleApps />
    </section>

    <section class="settings-card">
      <div class="settings-card-heading compact">
        <div class="settings-icon"><i class="fas fa-shield-alt"></i></div>
        <div>
          <p class="settings-eyebrow">{{ $t('config.essentials_access_eyebrow') }}</p>
          <h2>{{ $t('config.essentials_host_controls_title') }}</h2>
        </div>
      </div>

      <div class="status-summary">
        <div>
          <span>{{ $t('config.essentials_web_interface') }}</span>
          <strong>{{ webAccess }}</strong>
        </div>
        <i class="fas fa-wifi" aria-hidden="true"></i>
      </div>
      <p class="card-note">{{ $t('config.essentials_web_access_note') }}</p>
      <a class="quiet-link" href="#network">{{ $t('config.essentials_review_network') }} <i class="fas fa-arrow-right"></i></a>
    </section>

    <section class="settings-card settings-card-wide">
      <div class="settings-card-heading compact">
        <div class="settings-icon"><i class="fas fa-vr-cardboard"></i></div>
        <div>
          <p class="settings-eyebrow">{{ $t('config.essentials_3d_readiness_eyebrow') }}</p>
          <h2>{{ $t('config.essentials_virtual_display_title') }}</h2>
        </div>
        <span class="status-pill" :class="`status-${driverState.tone}`">
          <span class="status-dot"></span>{{ $t(driverState.labelKey) }}
        </span>
      </div>
      <p class="card-note driver-note">{{ $t(driverState.detailKey) }}</p>
      <a v-if="driverState.tone !== 'success'" class="quiet-link" href="./troubleshooting">
        {{ $t('config.essentials_open_diagnostics') }} <i class="fas fa-arrow-right"></i>
      </a>
    </section>

    <section class="settings-card settings-card-wide tray-setting">
      <div class="simple-toggle-row flush">
        <div>
          <strong>{{ $t('config.essentials_tray') }}</strong>
          <span>{{ $t('config.essentials_tray_desc') }}</span>
        </div>
        <label class="form-switch" :aria-label="$t('config.essentials_tray')">
          <input
            class="form-check-input"
            type="checkbox"
            :checked="isEnabled('system_tray')"
            @change="setEnabled('system_tray', $event)"
          />
        </label>
      </div>
      <div class="simple-toggle-row tray-taskbar-repair">
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
