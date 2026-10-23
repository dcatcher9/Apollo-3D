<template>
  <section id="game3d-setup" class="apollo-section" aria-labelledby="game3d-setup-title">
    <div class="apollo-section-heading">
      <div>
        <p class="apollo-kicker">Game depth stereo</p>
        <h2 id="game3d-setup-title">Game 3D setup</h2>
      </div>
      <span class="apollo-state-pill" :class="`is-${status.tone}`" role="status">
        <span class="apollo-state-dot" aria-hidden="true"></span>
        {{ status.label }}
      </span>
    </div>

    <article class="apollo-readiness-card game3d-setup-card" :class="`is-${status.tone}`">
      <h3>{{ setup?.detected ? setup.game_name || 'Detected game' : 'Launch a game in Game 3D mode' }}</h3>
      <p v-if="setup?.detected && setup.executable" class="game3d-executable">{{ setup.executable }}</p>
      <p v-if="gameDirectory" class="game3d-executable">Setup folder: <code>{{ gameDirectory }}</code></p>
      <p>{{ status.description }}</p>

      <template v-if="canDecide">
        <p v-if="needsReShade" class="game3d-hint">
          Installing writes ReShade 6.8.0 with full add-on support (<code>dxgi.dll</code>, <code>ReShade.ini</code>) and the
          Sunshine 3D add-on into the setup folder above, for this executable only. Choose it only for an application you
          recognize as a single-player game. It takes effect the next time you launch the game.
        </p>
        <p v-else class="game3d-hint">
          Setting up installs and updates the Sunshine 3D add-on beside this game's existing ReShade. It takes effect the
          next time you launch the game.
        </p>
        <div class="game3d-actions">
          <button class="btn btn-apollo-primary" type="button" :disabled="Boolean(busy)" @click="decide('install')">
            {{ busy === 'install' ? 'Saving…' : setup.state === 'declined' ? 'Set up this game' : 'Install for this game' }}
          </button>
          <button v-if="setup.state === 'ready_to_install'" class="btn btn-apollo-quiet" type="button" :disabled="Boolean(busy)" @click="decide('never')">
            {{ busy === 'never' ? 'Saving…' : 'Never for this game' }}
          </button>
        </div>
        <p v-if="error" class="game3d-error" role="alert">{{ error }}</p>
      </template>

      <p v-if="setup?.detected && !['anti_cheat', 'ready_to_install', 'declined'].includes(setup.state)" class="game3d-hint">
        Setup leaves your running game open. New or updated components take effect the next time you launch the game.
      </p>

      <details v-if="showManualSetup" class="game3d-instructions">
        <summary>Manual setup</summary>
        <p v-if="!needsReShade && !needsAddon">Close the game before manually repairing or updating an existing add-on.</p>
        <p v-if="showRuntimeFallback">
          For a fresh game folder without ReShade or another proxy DLL, install the ReShade runtime first from PowerShell:
        </p>
        <pre v-if="showRuntimeFallback"><code>{{ runtimeInstallCommand }}</code></pre>
        <p v-if="showRuntimeFallback">
          This runtime installer preserves existing files and stops on conflicts or anti-cheat. For an existing ReShade installation, use
          <a href="https://reshade.me/#download" target="_blank" rel="noopener noreferrer">ReShade setup with full add-on support</a>.
          Resolve any reported proxy conflict while preserving the existing proxy DLLs.
        </p>
        <p>Once ReShade with full add-on support is installed, install the bundled Sunshine 3D add-on from PowerShell:</p>
        <pre><code>{{ installCommand }}</code></pre>
        <p v-if="needsHostDirectory">Replace &lt;hostdir&gt; with your Sunshine 3D installation folder.</p>
        <p>Launch the game again after setup. In ReShade, open Add-ons → Sunshine 3D and check Enable Game 3D if stereo remains unavailable.</p>
      </details>

      <p v-else-if="setup?.detected && ['current', 'updated'].includes(setup.state)" class="game3d-hint">
        If the game remains 2D after its next launch, confirm ReShade has full add-on support and Enable Game 3D is selected under Add-ons → Sunshine 3D.
      </p>
    </article>
  </section>
</template>

<script>
export default {
  props: {
    setup: { type: Object, default: null },
  },
  emits: ['update'],
  data() {
    return { busy: '', error: '' }
  },
  computed: {
    needsReShade() {
      return this.setup?.detected && (this.setup.state === 'ready_to_install' || (this.setup.state === 'declined' && !this.setup.reshade_present))
    },
    needsAddon() {
      return this.setup?.detected && this.setup.state === 'missing_addon'
    },
    canDecide() {
      return this.setup?.detected && ['ready_to_install', 'declined'].includes(this.setup.state) && Boolean(this.setup.identity)
    },
    showManualSetup() {
      return this.setup?.detected && ['ready_to_install', 'declined', 'missing_addon', 'update_pending', 'unavailable'].includes(this.setup.state)
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
      if (!this.setup) return { label: 'Checking', tone: 'neutral', description: 'Checking Game 3D setup status.' }
      if (this.setup.state === 'unavailable') {
        return { label: this.setup.detected ? 'Setup unavailable' : 'Unavailable', tone: this.setup.detected ? 'warning' : 'neutral', description: this.setup.detail || 'Game detection or setup is unavailable. Refresh to try again.' }
      }
      if (!this.setup.active && !this.setup.detected) {
        return { label: 'On demand', tone: 'neutral', description: 'Choose Game 3D in Moonlight 3D, then launch your game fullscreen. Sunshine 3D offers to install ReShade and its Game 3D add-on for that game; nothing is written into a game folder until you confirm it here. Folders that already contain ReShade receive the matching add-on automatically.' }
      }
      if (!this.setup.detected) {
        return { label: 'Waiting for a game', tone: 'neutral', description: this.setup.detail || 'Game 3D is active. Launch your game fullscreen on the streamed display to check its setup.' }
      }
      const states = {
        ready_to_install: { label: 'Ready to set up', tone: 'warning', description: 'Sunshine 3D can install ReShade with full add-on support and its Game 3D add-on for this executable. Nothing is written until you confirm.' },
        declined: { label: 'Not set up', tone: 'neutral', description: 'You chose not to set up this game. Sunshine 3D does not change its folder.' },
        anti_cheat: { label: 'Not offered', tone: 'neutral', description: 'Anti-cheat detected, so Sunshine 3D does not install ReShade for this game.' },
        missing_addon: { label: 'Preparing add-on', tone: 'warning', description: 'ReShade is already installed; Sunshine 3D will install its bundled add-on for this game when possible.' },
        installing: { label: 'Installing', tone: 'active', description: 'Installing the confirmed components beside this game.' },
        installed: { label: 'Installed for next launch', tone: 'success', description: 'ReShade and the Sunshine 3D add-on are installed. They will activate the next time you launch the game.' },
        blocked: { label: 'Setup needs attention', tone: 'warning', description: 'Setup could not proceed for this game. Review the detected installation before trying again.' },
        current: { label: 'Add-on up to date', tone: 'success', description: 'The installed Sunshine 3D add-on matches this host.' },
        update_available: { label: 'Update available', tone: 'warning', description: 'A matching bundled add-on update is available for this game.' },
        update_pending: { label: 'Update pending', tone: 'warning', description: 'The matching bundled add-on update will be applied when its files are available. It will activate the next time you launch the game.' },
        updated: { label: 'Add-on updated', tone: 'success', description: 'The Sunshine 3D add-on was updated from this host’s bundled copy. It will activate the next time you launch the game.' },
      }
      const status = states[this.setup.state] || { label: 'Checking', tone: 'neutral', description: 'Checking setup for the detected game.' }
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
        const response = await fetch('./api/game3d/setup/decision', {
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          method: 'POST',
          body: JSON.stringify({ executable: this.setup.executable, identity: this.setup.identity, decision }),
        })
        const body = await response.json().catch(() => ({}))
        if (!response.ok || !body.status) throw new Error(body.error || `Sunshine 3D returned HTTP ${response.status}`)
        if (body.setup) this.$emit('update', body.setup)
      } catch (reason) {
        this.error = `Could not save this choice: ${reason.message || reason}`
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

.game3d-error {
  margin-top: 0.6rem;
  color: var(--bs-danger-text-emphasis);
}
</style>
