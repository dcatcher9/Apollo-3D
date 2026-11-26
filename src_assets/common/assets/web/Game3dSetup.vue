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
      <p v-if="gameDirectory && !isTextGame" class="game3d-executable">Setup folder: <code>{{ gameDirectory }}</code></p>
      <p>{{ status.description }}</p>
      <p v-if="textProtection" class="game3d-text-protection" :class="`is-${textProtection}`" role="status">
        <span class="game3d-text-protection-label">{{ textProtection === 'game' ? 'Game text' : 'Fallback' }}</span>
        {{ setup.text_protection_detail }}
      </p>

      <template v-if="canDecide && isTextGame">
        <p v-if="!textGameApproved && setup.chosen" class="game3d-hint">
          You chose this application for conversion. Sunshine 3D converts its picture with its own depth and keeps its text
          readable{{ usesTextHelper ? ', with a small text helper loaded into it only while it runs' : ' from the picture alone' }}.
          Nothing is written into its folder. It takes effect at once and again each time the application starts.
        </p>
        <p v-else-if="!textGameApproved && usesTextHelper" class="game3d-hint">
          This looks like a 2D text game, which has no depth of its own. Sunshine 3D converts its picture with its own depth and
          uses a small text helper, loaded into the game only while it runs, to keep the game's text readable. Nothing is written into the
          game's folder. Choose it only for an application you recognize as a single-player game. It takes effect at once and
          again each time the game starts.
        </p>
        <p v-else-if="!textGameApproved" class="game3d-hint">
          This looks like a 2D text game, which has no depth of its own and composes its text itself. Sunshine 3D converts its
          picture with its own depth and keeps the text readable from the picture alone. Nothing is loaded into the game or
          written into its folder. It takes effect at once and again each time the game starts.
        </p>
        <div class="game3d-actions">
          <button v-if="!textGameApproved" class="btn btn-apollo-primary" type="button" :disabled="Boolean(busy)" @click="decide('install')">
            {{ busy === 'install' ? 'Saving…' : 'Convert this game' }}
          </button>
          <button v-if="setup.state !== 'declined'" class="btn btn-apollo-quiet" type="button" :disabled="Boolean(busy)" @click="decide('never')">
            {{ busy === 'never' ? 'Saving…' : textGameApproved ? 'Stop for this game' : 'Never for this game' }}
          </button>
        </div>
        <p v-if="setup.state === 'current'" class="game3d-hint">
          {{ usesTextHelper ? 'Stopping ends the conversion at once; a helper already loaded stays in the game until it closes.' : 'Stopping ends the conversion at once.' }}
        </p>
        <p v-if="error" class="game3d-error" role="alert">{{ error }}</p>
      </template>

      <template v-else-if="canDecide">
        <p v-if="needsReShade" class="game3d-hint">
          Installing writes ReShade 6.8.0 with full add-on support (<code>dxgi.dll</code>, <code>ReShade.ini</code>) and the
          Sunshine 3D add-on into the setup folder above, for this executable only. Choose it only for an application you
          recognize as a single-player game. It takes effect the next time you launch the game.
        </p>
        <p v-else class="game3d-hint">
          ReShade is already installed here. Installing the missing Sunshine 3D add-on copies it and its license into the
          add-on folder that this game's <code>ReShade.ini</code> names, disables ReShade's Generic Depth add-on and enables
          Game 3D there, and keeps the files it changes in <code>Sunshine3D-backups</code> in the setup folder. Once
          installed, the add-on is updated automatically after the game exits. It takes effect the next time you launch the
          game.
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

      <p v-if="setup?.detected && !isTextGame && !['anti_cheat', 'ready_to_install', 'declined'].includes(setup.state)" class="game3d-hint">
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

      <p v-else-if="setup?.detected && !isTextGame && ['current', 'updated'].includes(setup.state)" class="game3d-hint">
        If the game remains 2D after its next launch, confirm ReShade has full add-on support and Enable Game 3D is selected under Add-ons → Sunshine 3D.
      </p>
    </article>

    <article v-if="choosable.length" class="apollo-readiness-card game3d-setup-card game3d-choosable">
      <h3>Convert a 2D game yourself</h3>
      <p>
        {{ choosable.length === 1 ? 'This application was' : 'These applications were' }} in front on the streamed display, and
        Sunshine 3D does not take {{ choosable.length === 1 ? 'it' : 'them' }} for a 2D game by itself. If you are playing a 2D game
        there, such as a visual novel, you can have it converted all the same.
      </p>
      <p class="game3d-hint">
        Sunshine 3D then converts the application's picture with its own depth while it is in front and keeps its text readable.
        Where the text helper can follow the text, it is loaded into the application only while it runs. Nothing is written into
        its folder. Choose only an application you recognize as a single-player game. It takes effect at once and again each
        time the application starts, and you can stop it here at any time.
      </p>
      <ul class="game3d-choosable-list">
        <li v-for="application in choosable" :key="application.executable">
          <div>
            <strong>{{ application.game_name || 'Application' }}</strong>
            <p class="game3d-executable">{{ application.executable }}</p>
          </div>
          <button class="btn btn-apollo-quiet" type="button" :disabled="Boolean(busy)" @click="choose(application)">
            {{ busy === application.executable ? 'Saving…' : 'Convert this application' }}
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
    return { busy: '', error: '', chooseError: '' }
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
      if (!this.setup) return { label: 'Checking', tone: 'neutral', description: 'Checking Game 3D setup status.' }
      if (this.setup.state === 'unavailable') {
        return { label: this.setup.detected ? 'Setup unavailable' : 'Unavailable', tone: this.setup.detected ? 'warning' : 'neutral', description: this.setup.detail || 'Game detection or setup is unavailable. Refresh to try again.' }
      }
      if (!this.setup.active && !this.setup.detected) {
        return { label: 'On demand', tone: 'neutral', description: 'Choose Game 3D in Moonlight 3D, then launch your game fullscreen. Sunshine 3D offers to install ReShade and its Game 3D add-on for that game, or only the add-on where ReShade is already installed; nothing is written into a game folder until you confirm it here. An installed add-on is kept matched to this host automatically.' }
      }
      if (!this.setup.detected) {
        return { label: 'Waiting for a game', tone: 'neutral', description: this.setup.detail || 'Game 3D is active. Launch your game fullscreen on the streamed display to check its setup.' }
      }
      const states = {
        ready_to_install: { label: 'Ready to set up', tone: 'warning', description: 'Sunshine 3D can set up Game 3D for this executable. Nothing is written until you confirm.' },
        declined: { label: 'Not set up', tone: 'neutral', description: 'You chose not to set up this game. Sunshine 3D does not change its folder.' },
        anti_cheat: { label: 'Not offered', tone: 'neutral', description: 'Anti-cheat detected, so Sunshine 3D does not install ReShade for this game.' },
        missing_addon: { label: 'Add-on missing', tone: 'warning', description: 'ReShade is installed without the Sunshine 3D add-on.' },
        installing: { label: 'Installing', tone: 'active', description: 'Installing the confirmed components beside this game.' },
        installed: { label: 'Installed for next launch', tone: 'success', description: 'ReShade and the Sunshine 3D add-on are installed. They will activate the next time you launch the game.' },
        blocked: { label: 'Setup needs attention', tone: 'warning', description: 'Setup could not proceed for this game. Review the detected installation before trying again.' },
        current: { label: 'Add-on up to date', tone: 'success', description: 'The installed Sunshine 3D add-on matches this host.' },
        update_available: { label: 'Update available', tone: 'warning', description: 'A matching bundled add-on update is available for this game.' },
        update_pending: { label: 'Update pending', tone: 'warning', description: 'The matching bundled add-on update will be applied when its files are available. It will activate the next time you launch the game.' },
        updated: { label: 'Add-on updated', tone: 'success', description: 'The Sunshine 3D add-on was updated from this host’s bundled copy. It will activate the next time you launch the game.' },
      }
      if (this.isTextGame) {
        Object.assign(states, {
          ready_to_install: { label: 'Ready to set up', tone: 'warning', description: 'Sunshine 3D can convert this game. Nothing is loaded or converted until you confirm.' },
          declined: { label: 'Not set up', tone: 'neutral', description: 'You chose not to set up this game. Sunshine 3D loads nothing into it.' },
          anti_cheat: { label: 'Not offered', tone: 'neutral', description: 'Anti-cheat detected, so Sunshine 3D loads nothing into this game.' },
          current: { label: 'Converting', tone: 'success', description: 'Sunshine 3D converts this game while it is in front.' },
          installed: { label: 'Approved', tone: 'success', description: 'The game is converted the next time it starts.' },
        })
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
        const body = await postJson('./api/game3d/setup/decision', { executable: this.setup.executable, identity: this.setup.identity, decision })
        if (body.setup) this.$emit('update', body.setup)
      } catch (reason) {
        this.error = `Could not save this choice: ${reason.message || reason}`
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
        this.chooseError = `Could not save this choice: ${reason.message || reason}`
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
