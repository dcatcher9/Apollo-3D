<template>
  <section id="game3d-setup" class="apollo-section" aria-labelledby="game3d-setup-title">
    <div class="apollo-section-heading">
      <div>
        <p class="apollo-kicker">{{ $t('game3d_setup.kicker') }}</p>
        <h2 id="game3d-setup-title">{{ $t('game3d_setup.title') }}</h2>
      </div>
      <span class="apollo-state-pill" :class="`is-${status.tone}`" role="status">
        <span class="apollo-state-dot" aria-hidden="true"></span>
        {{ status.label }}
      </span>
    </div>

    <article class="apollo-readiness-card game3d-setup-card" :class="`is-${status.tone}`">
      <h3>{{ setup?.detected ? setup.game_name || $t('game3d_setup.detected_game') : $t('game3d_setup.launch_prompt') }}</h3>
      <p v-if="setup?.detected && setup.executable" class="game3d-executable">{{ setup.executable }}</p>
      <p v-if="gameDirectory && !isTextGame" class="game3d-executable">{{ $t('game3d_setup.setup_folder') }} <code>{{ gameDirectory }}</code></p>
      <p>{{ status.description }}</p>
      <p v-if="textProtection" class="game3d-text-protection" :class="`is-${textProtection}`" role="status">
        <span class="game3d-text-protection-label">{{ textProtection === 'game' ? $t('game3d_setup.text_protection_game') : $t('game3d_setup.text_protection_fallback') }}</span>
        {{ setup.text_protection_detail }}
      </p>

      <template v-if="canDecide && isTextGame">
        <p v-if="!textGameApproved && setup.chosen" class="game3d-hint">
          {{ usesTextHelper ? $t('game3d_setup.chosen_hint_helper') : $t('game3d_setup.chosen_hint_picture') }}
        </p>
        <p v-else-if="!textGameApproved && usesTextHelper" class="game3d-hint">{{ $t('game3d_setup.text_game_hint_helper') }}</p>
        <p v-else-if="!textGameApproved" class="game3d-hint">{{ $t('game3d_setup.text_game_hint_picture') }}</p>
        <div class="game3d-actions">
          <button v-if="!textGameApproved" class="btn btn-apollo-primary" type="button" :disabled="Boolean(busy)" @click="decide('install')">
            {{ busy === 'install' ? $t('game3d_setup.saving') : $t('game3d_setup.convert_game') }}
          </button>
          <button v-if="setup.state !== 'declined'" class="btn btn-apollo-quiet" type="button" :disabled="Boolean(busy)" @click="decide('never')">
            {{ busy === 'never' ? $t('game3d_setup.saving') : textGameApproved ? $t('game3d_setup.stop_for_game') : $t('game3d_setup.never_for_game') }}
          </button>
        </div>
        <p v-if="setup.state === 'current'" class="game3d-hint">
          {{ usesTextHelper ? $t('game3d_setup.stop_hint_helper') : $t('game3d_setup.stop_hint') }}
        </p>
        <p v-if="error" class="game3d-error" role="alert">{{ error }}</p>
      </template>

      <template v-else-if="canDecide">
        <i18n-t v-if="needsReShade" keypath="game3d_setup.install_reshade_hint" tag="p" class="game3d-hint">
          <template #dll><code>dxgi.dll</code></template>
          <template #ini><code>ReShade.ini</code></template>
        </i18n-t>
        <i18n-t v-else keypath="game3d_setup.install_addon_hint" tag="p" class="game3d-hint">
          <template #ini><code>ReShade.ini</code></template>
          <template #backups><code>Sunshine3D-backups</code></template>
        </i18n-t>
        <div class="game3d-actions">
          <button class="btn btn-apollo-primary" type="button" :disabled="Boolean(busy)" @click="decide('install')">
            {{ busy === 'install' ? $t('game3d_setup.saving') : setup.state === 'declined' ? $t('game3d_setup.set_up_game') : $t('game3d_setup.install_for_game') }}
          </button>
          <button v-if="setup.state === 'ready_to_install'" class="btn btn-apollo-quiet" type="button" :disabled="Boolean(busy)" @click="decide('never')">
            {{ busy === 'never' ? $t('game3d_setup.saving') : $t('game3d_setup.never_for_game') }}
          </button>
        </div>
        <p v-if="error" class="game3d-error" role="alert">{{ error }}</p>
      </template>

      <p v-if="setup?.detected && !isTextGame && !['anti_cheat', 'ready_to_install', 'declined'].includes(setup.state)" class="game3d-hint">
        {{ $t('game3d_setup.running_game_hint') }}
      </p>

      <details v-if="showManualSetup" class="game3d-instructions">
        <summary>{{ $t('game3d_setup.manual_setup') }}</summary>
        <p v-if="!needsReShade && !needsAddon">{{ $t('game3d_setup.manual_close_game') }}</p>
        <p v-if="showRuntimeFallback">{{ $t('game3d_setup.manual_runtime_intro') }}</p>
        <pre v-if="showRuntimeFallback"><code>{{ runtimeInstallCommand }}</code></pre>
        <i18n-t v-if="showRuntimeFallback" keypath="game3d_setup.manual_runtime_note" tag="p">
          <template #link>
            <a href="https://reshade.me/#download" target="_blank" rel="noopener noreferrer">{{ $t('game3d_setup.manual_reshade_setup_link') }}</a>
          </template>
        </i18n-t>
        <p>{{ $t('game3d_setup.manual_addon_intro') }}</p>
        <pre><code>{{ installCommand }}</code></pre>
        <p v-if="needsHostDirectory">{{ $t('game3d_setup.manual_replace_hostdir', { hostdir: hostDirPlaceholder }) }}</p>
        <p>{{ $t('game3d_setup.manual_launch_again', { menu: reshadeAddonMenu, option: reshadeEnableOption }) }}</p>
      </details>

      <p v-else-if="setup?.detected && !isTextGame && ['current', 'updated'].includes(setup.state)" class="game3d-hint">
        {{ $t('game3d_setup.still_2d_hint', { menu: reshadeAddonMenu, option: reshadeEnableOption }) }}
      </p>
    </article>

    <article v-if="choosable.length" class="apollo-readiness-card game3d-setup-card game3d-choosable">
      <h3>{{ $t('game3d_setup.choosable_title') }}</h3>
      <p>{{ $t('game3d_setup.choosable_desc', { count: choosable.length }, choosable.length) }}</p>
      <p class="game3d-hint">{{ $t('game3d_setup.choosable_hint') }}</p>
      <ul class="game3d-choosable-list">
        <li v-for="application in choosable" :key="application.executable">
          <div>
            <strong>{{ application.game_name || $t('game3d_setup.application') }}</strong>
            <p class="game3d-executable">{{ application.executable }}</p>
          </div>
          <button class="btn btn-apollo-quiet" type="button" :disabled="Boolean(busy)" @click="choose(application)">
            {{ busy === application.executable ? $t('game3d_setup.saving') : $t('game3d_setup.convert_application') }}
          </button>
        </li>
      </ul>
      <p v-if="chooseError" class="game3d-error" role="alert">{{ chooseError }}</p>
    </article>
  </section>
</template>

<script>
import { postJson } from './api'

export default {
  props: {
    setup: { type: Object, default: null },
  },
  emits: ['update'],
  data() {
    // ReShade's menu and option names as the add-on shows them, and the commands' folder
    // placeholder, stay untranslated inside the translated sentences.
    return { busy: '', error: '', chooseError: '', reshadeAddonMenu: 'Add-ons → Sunshine 3D', reshadeEnableOption: 'Enable Game 3D', hostDirPlaceholder: '<hostdir>' }
  },
  computed: {
    isTextGame() {
      return this.setup?.detected && this.setup.kind === 'text_hook'
    },
    choosable() {
      // Applications lately in front on the streamed display that no rule offers.
      return Array.isArray(this.setup?.choosable) ? this.setup.choosable : []
    },
    textProtection() {
      // Reported only while the game is in front on a stream and being converted.
      return this.isTextGame && this.setup.state === 'current' ? this.setup.text_protection || '' : ''
    },
    usesTextHelper() {
      return this.setup?.text_helper !== false
    },
    textGameApproved() {
      return this.isTextGame && ['current', 'installed'].includes(this.setup.state)
    },
    needsReShade() {
      return this.setup?.detected && ['ready_to_install', 'declined'].includes(this.setup.state) && !this.setup.reshade_present
    },
    needsAddon() {
      return this.setup?.detected && (this.setup.state === 'missing_addon' || (this.setup.state === 'ready_to_install' && this.setup.reshade_present))
    },
    canDecide() {
      // A text game's confirmation stands for every start, so its active state can still be withdrawn.
      const states = this.isTextGame ? ['ready_to_install', 'declined', 'current', 'installed'] : ['ready_to_install', 'declined']
      return this.setup?.detected && states.includes(this.setup.state) && Boolean(this.setup.identity)
    },
    showManualSetup() {
      return this.setup?.detected && !this.isTextGame && ['ready_to_install', 'declined', 'missing_addon', 'update_pending', 'unavailable'].includes(this.setup.state)
    },
    showRuntimeFallback() {
      return this.setup?.detected && (this.needsReShade || this.setup.state === 'unavailable')
    },
    gameDirectory() {
      return this.setup?.detected && this.setup.executable ? this.setup.executable.replace(/[\\/][^\\/]+$/, '') : ''
    },
    needsHostDirectory() {
      return !this.setup?.install_script || (this.showRuntimeFallback && !this.setup?.runtime_install_script)
    },
    runtimeInstallCommand() {
      const quote = value => `'${String(value).replace(/'/g, "''")}'`
      const script = this.setup?.runtime_install_script || '<hostdir>\\tools\\reshade\\install-runtime.ps1'
      return `& ${quote(script)} -GameExecutable ${quote(this.setup?.executable || '')}`
    },
    installCommand() {
      const quote = value => `'${String(value).replace(/'/g, "''")}'`
      const script = this.setup?.install_script || '<hostdir>\\tools\\reshade\\install.ps1'
      const allowRunning = this.needsReShade || this.needsAddon ? ' -AllowRunningGame' : ''
      return `& ${quote(script)} -GameExecutable ${quote(this.setup?.executable || '')}${allowRunning}`
    },
    status() {
      const $t = (...args) => this.$t(...args)
      if (!this.setup) return { label: $t('game3d_setup.status_checking'), tone: 'neutral', description: $t('game3d_setup.checking_desc') }
      if (this.setup.state === 'unavailable') {
        return { label: this.setup.detected ? $t('game3d_setup.status_setup_unavailable') : $t('game3d_setup.status_unavailable'), tone: this.setup.detected ? 'warning' : 'neutral', description: this.setup.detail || $t('game3d_setup.unavailable_desc') }
      }
      if (!this.setup.active && !this.setup.detected) {
        return { label: $t('game3d_setup.status_on_demand'), tone: 'neutral', description: $t('game3d_setup.on_demand_desc') }
      }
      if (!this.setup.detected) {
        return { label: $t('game3d_setup.status_waiting'), tone: 'neutral', description: this.setup.detail || $t('game3d_setup.waiting_desc') }
      }
      const states = {
        ready_to_install: { label: $t('game3d_setup.status_ready_to_install'), tone: 'warning', description: $t('game3d_setup.ready_to_install_desc') },
        declined: { label: $t('game3d_setup.status_declined'), tone: 'neutral', description: $t('game3d_setup.declined_desc') },
        anti_cheat: { label: $t('game3d_setup.status_anti_cheat'), tone: 'neutral', description: $t('game3d_setup.anti_cheat_desc') },
        missing_reshade: { label: $t('game3d_setup.status_missing_reshade'), tone: 'warning', description: $t('game3d_setup.missing_reshade_desc') },
        missing_addon: { label: $t('game3d_setup.status_missing_addon'), tone: 'warning', description: $t('game3d_setup.missing_addon_desc') },
        installing: { label: $t('game3d_setup.status_installing'), tone: 'active', description: $t('game3d_setup.installing_desc') },
        installed: { label: $t('game3d_setup.status_installed'), tone: 'success', description: $t('game3d_setup.installed_desc') },
        current: { label: $t('game3d_setup.status_current'), tone: 'success', description: $t('game3d_setup.current_desc') },
        update_available: { label: $t('game3d_setup.status_update_available'), tone: 'warning', description: $t('game3d_setup.update_available_desc') },
        update_pending: { label: $t('game3d_setup.status_update_pending'), tone: 'warning', description: $t('game3d_setup.update_pending_desc') },
        updated: { label: $t('game3d_setup.status_updated'), tone: 'success', description: $t('game3d_setup.updated_desc') },
      }
      if (this.isTextGame) {
        Object.assign(states, {
          ready_to_install: { label: $t('game3d_setup.status_ready_to_install'), tone: 'warning', description: $t('game3d_setup.text_ready_to_install_desc') },
          declined: { label: $t('game3d_setup.status_declined'), tone: 'neutral', description: $t('game3d_setup.text_declined_desc') },
          anti_cheat: { label: $t('game3d_setup.status_anti_cheat'), tone: 'neutral', description: $t('game3d_setup.text_anti_cheat_desc') },
          current: { label: $t('game3d_setup.status_converting'), tone: 'success', description: $t('game3d_setup.text_current_desc') },
          installed: { label: $t('game3d_setup.status_approved'), tone: 'success', description: $t('game3d_setup.text_installed_desc') },
        })
      }
      const status = states[this.setup.state] || { label: $t('game3d_setup.status_checking'), tone: 'neutral', description: $t('game3d_setup.checking_detected_desc') }
      return { ...status, description: this.setup.detail || status.description }
    },
  },
  watch: {
    'setup.identity'() {
      this.error = ''
    },
  },
  methods: {
    async decide(decision) {
      if (this.busy || !this.canDecide) return
      this.busy = decision
      this.error = ''
      try {
        const body = await postJson('./api/game3d/setup/decision', { executable: this.setup.executable, identity: this.setup.identity, decision })
        if (body.setup) this.$emit('update', body.setup)
      } catch (reason) {
        this.error = this.$t('game3d_setup.save_choice_failed', { error: String(reason.message || reason) })
      } finally {
        this.busy = ''
      }
    },
    async choose(application) {
      if (this.busy) return
      this.busy = application.executable
      this.chooseError = ''
      try {
        const body = await postJson('./api/game3d/setup/decision', { executable: application.executable, identity: application.identity, decision: 'convert' })
        if (body.setup) this.$emit('update', body.setup)
      } catch (reason) {
        this.chooseError = this.$t('game3d_setup.save_choice_failed', { error: String(reason.message || reason) })
      } finally {
        this.busy = ''
      }
    },
  },
}
</script>

<style scoped>
#game3d-setup {
  scroll-margin-top: 6rem;
}

.game3d-setup-card {
  min-width: 0;
}

.game3d-setup-card p + p,
.game3d-instructions,
.game3d-actions {
  margin-top: 0.8rem;
}

.game3d-executable,
.game3d-instructions pre {
  overflow-wrap: anywhere;
}

.game3d-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

.game3d-instructions summary {
  cursor: pointer;
}

.game3d-instructions pre {
  white-space: pre-wrap;
  margin: 0.8rem 0;
  padding: 0.9rem;
  border: 1px solid var(--bs-border-color);
  border-radius: var(--apollo-radius-control);
  background: var(--bs-tertiary-bg);
}

.game3d-hint {
  font-size: var(--apollo-text-caption);
}

.game3d-text-protection {
  font-size: var(--apollo-text-caption);
}

.game3d-text-protection-label {
  display: inline-block;
  margin-right: 0.4rem;
  padding: 0.05rem 0.5rem;
  border-radius: var(--apollo-radius-pill);
  font-weight: 600;
  color: var(--bs-success-text-emphasis);
  background: var(--bs-success-bg-subtle);
}

.game3d-text-protection.is-fallback .game3d-text-protection-label {
  color: var(--bs-warning-text-emphasis);
  background: var(--bs-warning-bg-subtle);
}

.game3d-choosable {
  margin-top: 1rem;
}

.game3d-choosable-list {
  margin: 0.8rem 0 0;
  padding: 0;
  list-style: none;
}

.game3d-choosable-list li {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  padding: 0.6rem 0;
  border-top: 1px solid var(--bs-border-color);
}

.game3d-choosable-list li > div {
  min-width: 0;
}

.game3d-choosable-list .game3d-executable {
  margin: 0;
  font-size: var(--apollo-text-caption);
}

.game3d-error {
  margin-top: 0.6rem;
  color: var(--bs-danger-text-emphasis);
}
</style>
