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

      <p v-if="needsReShade || needsAddon" class="game3d-hint">
        Sunshine 3D will install the missing components automatically when possible. Manual setup is available below if it cannot finish.
      </p>
      <p v-if="setup?.detected" class="game3d-hint">
        Automatic setup leaves your running game open. New or updated components take effect the next time you launch the game.
      </p>

      <details v-if="showManualSetup" class="game3d-instructions">
        <summary>Manual setup if automatic setup cannot finish</summary>
        <p v-if="!needsReShade && !needsAddon">Close the game before manually repairing or updating an existing add-on.</p>
        <p v-if="showRuntimeFallback">
          For a fresh game folder without ReShade or another proxy DLL, install the ReShade runtime first from PowerShell:
        </p>
        <pre v-if="showRuntimeFallback"><code>{{ runtimeInstallCommand }}</code></pre>
        <p v-if="showRuntimeFallback">
          This runtime installer preserves existing files and stops on conflicts. For an existing ReShade installation, use
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
  computed: {
    needsReShade() {
      return this.setup?.detected && this.setup.state === 'missing_reshade'
    },
    needsAddon() {
      return this.setup?.detected && this.setup.state === 'missing_addon'
    },
    showManualSetup() {
      return this.setup?.detected && ['missing_reshade', 'missing_addon', 'update_pending', 'blocked', 'unavailable'].includes(this.setup.state)
    },
    showRuntimeFallback() {
      return this.setup?.detected && ['missing_reshade', 'blocked', 'unavailable'].includes(this.setup.state)
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
        return { label: this.setup.detected ? 'Setup unavailable' : 'Unavailable', tone: this.setup.detected ? 'warning' : 'neutral', description: this.setup.detail || 'Game detection or automatic setup is unavailable. Refresh to try again.' }
      }
      if (!this.setup.active && !this.setup.detected) {
        return { label: 'On demand', tone: 'neutral', description: 'Choose Game 3D in Moonlight 3D, then launch your game. Sunshine 3D will automatically install ReShade and the Sunshine 3D add-on when possible, and keep the add-on matched to this host.' }
      }
      if (!this.setup.detected) {
        return { label: 'Waiting for a game', tone: 'neutral', description: this.setup.detail || 'Game 3D is active. Launch your game on the streamed display to check its setup.' }
      }
      const states = {
        missing_reshade: { label: 'Preparing setup', tone: 'warning', description: 'Sunshine 3D will automatically install ReShade with full add-on support and its bundled add-on beside this game when possible.' },
        missing_addon: { label: 'Preparing add-on', tone: 'warning', description: 'Sunshine 3D will automatically install its bundled add-on for this game when possible.' },
        installing: { label: 'Installing', tone: 'active', description: 'Installing ReShade with full add-on support and the Sunshine 3D add-on beside this game.' },
        installed: { label: 'Installed for next launch', tone: 'success', description: 'ReShade and the Sunshine 3D add-on are installed. They will activate the next time you launch the game.' },
        blocked: { label: 'Setup needs attention', tone: 'warning', description: 'Automatic setup could not proceed for this game. Review the detected installation before trying again.' },
        current: { label: 'Add-on up to date', tone: 'success', description: 'The installed Sunshine 3D add-on matches this host.' },
        update_available: { label: 'Update available', tone: 'warning', description: 'A matching bundled add-on update is available for this game.' },
        update_pending: { label: 'Update pending', tone: 'warning', description: 'The matching bundled add-on update will be applied when its files are available. It will activate the next time you launch the game.' },
        updated: { label: 'Add-on updated', tone: 'success', description: 'The Sunshine 3D add-on was updated from this host’s bundled copy. It will activate the next time you launch the game.' },
      }
      const status = states[this.setup.state] || { label: 'Checking', tone: 'neutral', description: 'Checking setup for the detected game.' }
      return { ...status, description: this.setup.detail || status.description }
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
.game3d-instructions {
  margin-top: 0.8rem;
}

.game3d-executable,
.game3d-instructions pre {
  overflow-wrap: anywhere;
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
</style>
