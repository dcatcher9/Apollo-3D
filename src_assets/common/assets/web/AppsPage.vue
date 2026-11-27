<template>
  <Navbar></Navbar>
  <main id="content" class="container-xl library-page">
    <template v-if="!showEditForm">
      <header class="library-header">
        <div>
          <div class="library-eyebrow">{{ $t('apps.library_eyebrow') }}</div>
          <h1 class="library-title">{{ $t('apps.applications_title') }}</h1>
          <p class="library-subtitle">
            {{ $t('apps.library_subtitle') }}
          </p>
        </div>
        <button type="button" class="btn btn-primary btn-lg" @click="newApp" :disabled="actionDisabled">
          <i class="fas fa-plus me-2"></i>{{ $t('apps.add_new') }}
        </button>
      </header>

      <section class="library-toolbar" :aria-label="$t('apps.library_controls')">
        <label class="library-search">
          <span class="visually-hidden">{{ $t('apps.search_label') }}</span>
          <i class="fas fa-magnifying-glass"></i>
          <input
            v-model.trim="searchQuery"
            class="form-control"
            type="search"
            :placeholder="$t('apps.search_placeholder')"
            autocomplete="off"
          />
        </label>
        <span class="library-count">{{ $t('apps.shown_count', { shown: filteredApps.length, total: apps.length }) }}</span>
        <button
          type="button"
          class="btn btn-outline-secondary"
          @click="alphabetizeApps"
          :disabled="actionDisabled || !!searchQuery || reorderableAppCount < 2"
          v-if="!listReordered"
        >
          <i class="fas fa-arrow-down-a-z me-1"></i>
          <span class="d-none d-sm-inline">{{ $t('apps.alphabetize') }}</span>
        </button>
        <button
          type="button"
          class="btn btn-warning"
          @click="saveOrder"
          :disabled="actionDisabled"
          v-else
        >
          <i class="fas fa-floppy-disk me-1"></i>{{ $t('apps.save_order') }}
        </button>
      </section>

      <p class="small text-body-secondary mb-3" v-if="!searchQuery && reorderableAppCount > 1">
        <i class="fas fa-grip-vertical me-1"></i>{{ $t('apps.drag_hint') }}
      </p>

      <section class="library-grid" aria-live="polite">
        <article
          v-for="app in filteredApps"
          :key="app.uuid"
          class="app-card"
          :class="{ 'is-active': currentApp === app.uuid, dragover: app.dragover }"
          :style="{ '--app-accent': appAccent(app) }"
          :draggable="!app.builtin && !searchQuery && !actionDisabled"
          @dragstart="onDragStart($event, app)"
          @dragenter="onDragEnter($event, app)"
          @dragover="onDragOver($event)"
          @dragleave="onDragLeave(app)"
          @dragend="onDragEnd"
          @drop="onDrop($event, app)"
        >
          <div class="app-cover" @dblclick="exportLauncherFile(app)">
            <img
              v-if="coverPreviewUrl(app)"
              class="app-cover-image"
              :src="coverPreviewUrl(app)"
              :alt="app.name"
            />
            <span class="app-cover-letter" v-else>{{ appInitial(app) }}</span>
            <div class="app-card-badges">
              <span class="app-status-badge running" v-if="currentApp === app.uuid">
                <i class="fas fa-circle fa-2xs"></i>{{ $t('apps.running') }}
              </span>
              <span class="app-status-badge" v-if="app.builtin">
                <i class="fas fa-display"></i>{{ $t('apps.built_in') }}
              </span>
            </div>
            <span class="app-card-grip" v-if="!app.builtin && !searchQuery" :title="$t('apps.applications_reorder_desc')">
              <i class="fas fa-grip-vertical"></i>
            </span>
          </div>
          <div class="app-card-body">
            <div>
              <h2 class="app-card-title" :title="app.name">{{ app.name || $t('apps.untitled') }}</h2>
              <div class="app-card-meta" :title="app.cmd">{{ appCommandSummary(app) }}</div>
            </div>
            <div class="app-actions" @dragstart.stop.prevent>
              <button
                type="button"
                class="btn btn-warning"
                :disabled="actionDisabled"
                @click="closeApp"
                v-if="currentApp === app.uuid"
              >
                <i class="fas fa-stop me-1"></i>{{ $t('apps.close') }}
              </button>
              <button
                type="button"
                class="btn btn-outline-secondary"
                :disabled="actionDisabled"
                @click="editApp(app)"
                :aria-label="$t('apps.edit')"
                :title="$t('apps.edit')"
                v-if="!app.builtin"
              >
                <i class="fas fa-pen"></i>
              </button>
              <div class="dropdown">
                <button
                  type="button"
                  class="btn btn-outline-secondary"
                  data-bs-toggle="dropdown"
                  aria-expanded="false"
                  :disabled="actionDisabled"
                  :aria-label="$t('apps.more_actions')"
                >
                  <i class="fas fa-ellipsis"></i>
                </button>
                <ul class="dropdown-menu dropdown-menu-end">
                  <li>
                    <button type="button" class="dropdown-item" @click="exportLauncherFile(app)">
                      <i class="fas fa-arrow-up-from-bracket fa-fw me-2"></i>{{ $t('apps.export_launcher_file') }}
                    </button>
                  </li>
                  <li v-if="!app.builtin"><hr class="dropdown-divider"></li>
                  <li v-if="!app.builtin">
                    <button type="button" class="dropdown-item text-danger" @click="showDeleteForm(app)">
                      <i class="fas fa-trash fa-fw me-2"></i>{{ $t('apps.delete') }}
                    </button>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </article>

        <div
          class="library-drop-tail"
          v-if="draggingApp >= 0"
          @dragover="onDragOver"
          @drop="onDropAtEnd"
        >
          <span><i class="fas fa-arrow-down me-2"></i>{{ $t('apps.move_to_end') }}</span>
        </div>

        <button
          type="button"
          class="add-app-card"
          @click="newApp"
          :disabled="actionDisabled"
          v-if="!searchQuery"
        >
          <span class="add-app-icon"><i class="fas fa-plus"></i></span>
          <strong>{{ $t('apps.add_new') }}</strong>
          <span class="small mt-1">{{ $t('apps.add_new_desc') }}</span>
        </button>

        <div class="library-empty" v-if="filteredApps.length === 0 && searchQuery">
          <i class="fas fa-magnifying-glass fa-2x mb-3"></i>
          <strong>{{ $t('apps.no_results') }}</strong>
          <span class="small mt-1">{{ $t('apps.no_results_desc') }}</span>
          <button type="button" class="btn btn-link mt-2" @click="searchQuery = ''">{{ $t('apps.clear_search') }}</button>
        </div>
      </section>
    </template>

    <template v-else>
      <header class="editor-header">
        <div class="editor-title-wrap">
          <button
            type="button"
            class="btn btn-outline-secondary rounded-circle"
            @click="closeEditor"
            :aria-label="$t('_common.cancel')"
          >
            <i class="fas fa-arrow-left"></i>
          </button>
          <div class="min-w-0">
            <div class="library-eyebrow mb-1">{{ editForm.uuid ? $t('apps.edit_title') : $t('apps.new_title') }}</div>
            <h1 class="editor-title">{{ editForm.name || $t('apps.untitled') }}</h1>
          </div>
        </div>
        <button
          type="button"
          class="btn btn-outline-secondary"
          @click="exportLauncherFile(editForm)"
          :disabled="!editForm.uuid"
        >
          <i class="fas fa-arrow-up-from-bracket me-1"></i>
          <span class="d-none d-md-inline">{{ $t('apps.export_launcher_file') }}</span>
        </button>
      </header>

      <form @submit.prevent="save">
        <section class="editor-card">
          <h2 class="editor-section-title">{{ $t('apps.details_title') }}</h2>
          <div class="row g-4">
            <div class="col-8 col-sm-5 col-md-4 col-lg-3 mx-auto mx-md-0">
              <div class="editor-cover-preview" :style="{ '--edit-accent': appAccent(editForm) }">
                <img v-if="editCoverPreview" :src="editCoverPreview" :alt="editForm.name" />
                <span class="app-cover-letter" v-else>{{ appInitial(editForm) }}</span>
              </div>
              <p class="form-text text-center mb-0 mt-2">{{ $t('apps.artwork_desc') }}</p>
            </div>
            <div class="col-md-8 col-lg-9">
              <div class="mb-3">
                <label for="appName" class="form-label">{{ $t('apps.app_name') }}</label>
                <input
                  type="text"
                  class="form-control form-control-lg"
                  id="appName"
                  aria-describedby="appNameHelp"
                  v-model="editForm.name"
                  autofocus
                />
                <div id="appNameHelp" class="form-text">{{ $t('apps.app_name_desc') }}</div>
              </div>

              <div class="mb-3 dropdown">
                <label for="appImagePath" class="form-label">{{ $t('apps.artwork') }}</label>
                <div class="input-group">
                  <input
                    type="text"
                    class="form-control monospace"
                    id="appImagePath"
                    aria-describedby="appImagePathHelp"
                    v-model="editForm['image-path']"
                    @input="editCoverPreview = ''"
                  />
                  <button
                    class="btn btn-outline-secondary dropdown-toggle"
                    type="button"
                    id="findCoverToggle"
                    aria-expanded="false"
                    @click="showCoverFinder"
                    ref="coverFinderDropdown"
                  >
                    {{ $t('apps.find_cover') }}
                  </button>
                </div>
                <div id="appImagePathHelp" class="form-text">{{ $t('apps.image_desc') }}</div>
                <div class="dropdown-menu dropdown-menu-end cover-finder overflow-hidden" aria-labelledby="findCoverToggle">
                  <div class="modal-header px-3 py-2">
                    <h3 class="h5 modal-title">{{ $t('apps.covers_found') }}</h3>
                    <button type="button" class="btn-close" :aria-label="$t('apps.close_dialog')" @click="closeCoverFinder"></button>
                  </div>
                  <div class="modal-body cover-results px-3 pt-3" :class="{ busy: coverFinderBusy }">
                    <div class="row">
                      <div v-if="coverSearching" class="col-6 col-lg-4 mb-3">
                        <div class="cover-container">
                          <div class="spinner-border" role="status">
                            <span class="visually-hidden">{{ $t('apps.loading') }}</span>
                          </div>
                        </div>
                      </div>
                      <button
                        v-for="cover in coverCandidates"
                        :key="cover.key"
                        type="button"
                        class="col-6 col-lg-4 mb-3 border-0 bg-transparent text-body"
                        @click="useCover(cover)"
                      >
                        <span class="cover-container result d-block">
                          <img class="rounded" :src="cover.url" :alt="cover.name" />
                        </span>
                        <span class="d-block text-nowrap text-center text-truncate mt-1">{{ cover.name }}</span>
                      </button>
                      <div class="col-12 text-center text-body-secondary py-4" v-if="!coverSearching && coverCandidates.length === 0">
                        {{ $t('apps.no_covers') }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="mb-3">
                <label for="appCmd" class="form-label">{{ $t('apps.cmd') }}</label>
                <input
                  type="text"
                  class="form-control monospace"
                  id="appCmd"
                  aria-describedby="appCmdHelp"
                  v-model="editForm.cmd"
                />
                <div id="appCmdHelp" class="form-text">
                  {{ $t('apps.cmd_desc') }}
                  <span class="d-block"><b>{{ $t('_common.note') }}</b> {{ $t('apps.cmd_note') }}</span>
                </div>
              </div>

            </div>
          </div>
        </section>

        <details class="advanced-panel">
          <summary>
            <span class="advanced-summary-icon"><i class="fas fa-sliders"></i></span>
            <span>
              <strong class="d-block">{{ $t('apps.advanced_title') }}</strong>
              <small class="text-body-secondary">{{ $t('apps.advanced_desc') }}</small>
            </span>
          </summary>
          <div class="advanced-content">
            <Checkbox
              class="mb-4"
              id="appElevation"
              label="_common.run_as"
              desc="apps.run_as_desc"
              v-model="editForm.elevated"
              default="false"
            ></Checkbox>

            <div class="mb-4">
              <label class="form-label">{{ $t('apps.detached_cmds') }}</label>
              <div v-for="(c, i) in editForm.detached" :key="'detached-' + i" class="d-flex gap-2 my-2">
                <input type="text" v-model="editForm.detached[i]" class="form-control monospace" />
                <button type="button" class="btn btn-outline-danger" @click="editForm.detached.splice(i, 1)" :aria-label="$t('apps.delete')">
                  <i class="fas fa-trash"></i>
                </button>
                <button type="button" class="btn btn-outline-success" @click="editForm.detached.splice(i, 0, '')" :aria-label="$t('apps.add_new')">
                  <i class="fas fa-plus"></i>
                </button>
              </div>
              <button type="button" class="btn btn-outline-success mt-2" @click="editForm.detached.push('')">
                <i class="fas fa-plus me-1"></i>{{ $t('apps.detached_cmds_add') }}
              </button>
              <div class="form-text">
                {{ $t('apps.detached_cmds_desc') }}<br />
                <b>{{ $t('_common.note') }}</b> {{ $t('apps.detached_cmds_note') }}
              </div>
            </div>

            <Checkbox
              class="mb-3"
              id="excludeGlobalPrep"
              label="apps.global_prep_name"
              desc="apps.global_prep_desc"
              v-model="editForm['exclude-global-prep-cmd']"
              default="true"
              inverse-values
            ></Checkbox>
            <div class="mb-4">
              <label class="form-label">{{ $t('apps.cmd_prep_name') }}</label>
              <div class="form-text pre-wrap mb-2">{{ $t('apps.cmd_prep_desc') }}</div>
              <div class="table-responsive" v-if="editForm['prep-cmd'].length > 0">
                <table class="table align-middle">
                  <thead>
                    <tr>
                      <th scope="col"><i class="fas fa-play"></i> {{ $t('_common.do_cmd') }}</th>
                      <th scope="col"><i class="fas fa-undo"></i> {{ $t('_common.undo_cmd') }}</th>
                      <th scope="col"><i class="fas fa-shield-alt"></i> {{ $t('_common.run_as') }}</th>
                      <th scope="col"><span class="visually-hidden">{{ $t('apps.actions') }}</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(c, i) in editForm['prep-cmd']" :key="'prep-cmd-' + i">
                      <td><input type="text" class="form-control monospace" v-model="c.do" /></td>
                      <td><input type="text" class="form-control monospace" v-model="c.undo" /></td>
                      <td>
                        <Checkbox :id="'prep-cmd-admin-' + i" label="_common.elevated" desc="" v-model="c.elevated"></Checkbox>
                      </td>
                      <td>
                        <div class="d-flex gap-2 justify-content-end">
                          <button type="button" class="btn btn-outline-danger" @click="editForm['prep-cmd'].splice(i, 1)">
                            <i class="fas fa-trash"></i>
                          </button>
                          <button type="button" class="btn btn-outline-success" @click="addCmd(editForm['prep-cmd'], i)">
                            <i class="fas fa-plus"></i>
                          </button>
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <button type="button" class="btn btn-outline-success" @click="addCmd(editForm['prep-cmd'], -1)">
                <i class="fas fa-plus me-1"></i>{{ $t('apps.add_cmds') }}
              </button>
            </div>

            <div class="row g-3">
              <div class="col-md-6">
                <label for="appWorkingDir" class="form-label">{{ $t('apps.working_dir') }}</label>
                <input type="text" class="form-control monospace" id="appWorkingDir" aria-describedby="appWorkingDirHelp" v-model="editForm['working-dir']" />
                <div id="appWorkingDirHelp" class="form-text">{{ $t('apps.working_dir_desc') }}</div>
              </div>
              <div class="col-md-6">
                <label for="appOutput" class="form-label">{{ $t('apps.output_name') }}</label>
                <input type="text" class="form-control monospace" id="appOutput" aria-describedby="appOutputHelp" v-model="editForm.output" />
                <div id="appOutputHelp" class="form-text">{{ $t('apps.output_desc') }}</div>
              </div>
            </div>

            <hr class="my-4" />

            <div class="row g-3">
              <div class="col-lg-6">
                <Checkbox class="mb-3" id="autoDetach" label="apps.auto_detach" desc="apps.auto_detach_desc" v-model="editForm['auto-detach']" default="true"></Checkbox>
                <Checkbox class="mb-3" id="waitAll" label="apps.wait_all" desc="apps.wait_all_desc" v-model="editForm['wait-all']" default="true"></Checkbox>
              </div>
            </div>

            <div class="row g-3 mt-1">
              <div class="col-md-6">
                <label for="exitTimeout" class="form-label">{{ $t('apps.exit_timeout') }}</label>
                <input type="number" class="form-control monospace" id="exitTimeout" aria-describedby="exitTimeoutHelp" v-model="editForm['exit-timeout']" min="0" placeholder="5" />
                <div id="exitTimeoutHelp" class="form-text">{{ $t('apps.exit_timeout_desc') }}</div>
              </div>
            </div>

            <details class="nested-details">
              <summary>{{ $t('apps.env_vars_about') }}</summary>
              <div class="nested-details-content overflow-auto">
                <p class="form-text mt-0">{{ $t('apps.env_vars_desc') }}</p>
                <table class="env-table">
                  <tbody>
                    <tr><td class="monospace">APOLLO_APP_ID</td><td>{{ $t('apps.env_app_id') }}</td></tr>
                    <tr><td class="monospace">APOLLO_APP_NAME</td><td>{{ $t('apps.env_app_name') }}</td></tr>
                    <tr><td class="monospace">APOLLO_APP_UUID</td><td>{{ $t('apps.env_app_uuid') }}</td></tr>
                    <tr><td class="monospace">APOLLO_APP_STATUS</td><td>{{ $t('apps.env_app_status') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_UUID</td><td>{{ $t('apps.env_client_uuid') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_NAME</td><td>{{ $t('apps.env_client_name') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_WIDTH</td><td>{{ $t('apps.env_client_width') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_HEIGHT</td><td>{{ $t('apps.env_client_height') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_FPS</td><td>{{ $t('apps.env_client_fps') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_HDR</td><td>{{ $t('apps.env_client_hdr') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_HOST_AUDIO</td><td>{{ $t('apps.env_client_host_audio') }}</td></tr>
                    <tr><td class="monospace">APOLLO_CLIENT_AUDIO_CONFIGURATION</td><td>{{ $t('apps.env_client_audio_config') }}</td></tr>
                  </tbody>
                </table>

                <div class="form-text mt-3">
                  <b>{{ $t('apps.env_rtss_cli_example') }}</b>
                  <pre>cmd /C \path\to\rtss-cli.exe limit:set %APOLLO_CLIENT_FPS%</pre>
                </div>
              </div>
            </details>
          </div>
        </details>

        <div class="editor-actions">
          <button type="button" class="btn btn-outline-secondary" @click="closeEditor">
            {{ $t('_common.cancel') }}
          </button>
          <button type="submit" class="btn btn-primary" :disabled="actionDisabled || !editForm.name.trim()">
            <i class="fas fa-floppy-disk me-1"></i>{{ $t('_common.save') }}
          </button>
        </div>
      </form>
    </template>
  </main>
</template>

<script>
import Navbar from './Navbar.vue'
import Checkbox from './Checkbox.vue'
import { Dropdown } from 'bootstrap/dist/js/bootstrap'
import { getJson, postJson } from './api'
import desktopCover from '../desktop.png'
import desktopAltCover from '../desktop-alt.png'
import steamCover from '../steam.png'
import virtualDesktopCover from '../virtual_desktop.png'

const bundledCovers = {
  "desktop.png": desktopCover,
  "desktop-alt.png": desktopAltCover,
  "steam.png": steamCover,
  "virtual_desktop.png": virtualDesktopCover
}

// The page fills in the name in the page's language (appTemplate).
const newApp = {
  "output": "",
  "cmd": "",
  "exclude-global-prep-cmd": false,
  "elevated": false,
  "auto-detach": true,
  "wait-all": true,
  "exit-timeout": 5,
  "prep-cmd": [],
  "detached": [],
  "image-path": ""
}

export default {
  components: {
    Navbar,
    Checkbox
  },
  data() {
    return {
      apps: [],
      showEditForm: false,
      actionDisabled: false,
      editForm: null,
      coverSearching: false,
      coverFinderBusy: false,
      coverCandidates: [],
      currentApp: "",
      draggingApp: -1,
      hostName: "",
      hostUUID: "",
      listReordered: false,
      searchQuery: "",
      editCoverPreview: ""
    };
  },
  computed: {
    reorderableAppCount() {
      return this.apps.filter(app => !app.builtin).length;
    },
    filteredApps() {
      const query = this.searchQuery.toLocaleLowerCase();
      if (!query) {
        return this.apps;
      }

      return this.apps.filter(app => {
        const name = app.name?.toString().toLocaleLowerCase() || "";
        const command = app.cmd?.toString().toLocaleLowerCase() || "";
        return name.includes(query) || command.includes(query);
      });
    }
  },
  created() {
    this.loadApps();
  },
  methods: {
    onDragStart(e, app) {
      if (app.builtin || this.showEditForm || this.searchQuery || this.actionDisabled) {
        e.preventDefault();
        return;
      }

      this.draggingApp = this.apps.indexOf(app);
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", app.uuid);
    },
    onDragEnter(e, app) {
      if (this.draggingApp < 0 || this.apps[this.draggingApp] === app) {
        return;
      }
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      this.apps.forEach(item => item.dragover = false);
      app.dragover = true;
    },
    onDragOver(e) {
      if (this.draggingApp < 0) {
        return;
      }
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
    },
    onDragLeave(app) {
      app.dragover = false;
    },
    onDragEnd() {
      this.apps.forEach(app => app.dragover = false);
      this.draggingApp = -1;
    },
    onDrop(e, app) {
      app.dragover = false;
      if (this.draggingApp < 0) {
        return;
      }
      e.preventDefault();

      const sourceIndex = this.draggingApp;
      let targetIndex = this.apps.indexOf(app);
      if (targetIndex < 0 || targetIndex === sourceIndex || targetIndex - 1 === sourceIndex) {
        this.onDragEnd();
        return;
      }

      const [draggedApp] = this.apps.splice(sourceIndex, 1);

      if (targetIndex > sourceIndex) {
        targetIndex -= 1;
      }

      this.apps.splice(targetIndex, 0, draggedApp);
      this.listReordered = true;
      this.onDragEnd();
    },
    onDropAtEnd(e) {
      if (this.draggingApp < 0) {
        return;
      }
      e.preventDefault();

      const sourceIndex = this.draggingApp;
      const configurableApps = this.apps.filter(app => !app.builtin);
      if (configurableApps[configurableApps.length - 1] !== this.apps[sourceIndex]) {
        const [draggedApp] = this.apps.splice(sourceIndex, 1);
        const builtinIndex = this.apps.findIndex(app => app.builtin);
        this.apps.splice(builtinIndex < 0 ? this.apps.length : builtinIndex, 0, draggedApp);
        this.listReordered = true;
      }
      this.onDragEnd();
    },
    alphabetizeApps() {
      const originalOrder = this.apps.map(app => app.uuid).join("\n");
      const configurableApps = this.apps.filter(app => !app.builtin).sort((a, b) => a.name.localeCompare(b.name));
      const builtinApps = this.apps.filter(app => app.builtin);
      this.apps.splice(0, this.apps.length, ...configurableApps, ...builtinApps);
      this.listReordered = originalOrder !== this.apps.map(app => app.uuid).join("\n");
      if (!this.listReordered) {
        alert(this.$t('apps.already_ordered'));
      }
    },
    saveOrder() {
      this.actionDisabled = true;

      const reorderedUUIDs = this.apps.reduce((reordered, i) => {
        if (i.uuid && !i.builtin) {
          reordered.push(i.uuid)
        }
        return reordered
      }, [])

      postJson("./api/apps/reorder", {order: reorderedUUIDs})
      .catch((error) => {
        alert(this.$t("apps.reorder_failed") + error.message);
      })
      .finally(() => {
        return this.loadApps();
      })
      .finally(() => {
        this.actionDisabled = false;
      })

    },
    loadApps() {
      return getJson("./api/apps")
      .then(r => {
        const builtinApps = (Array.isArray(r.builtin_apps) ? r.builtin_apps : [])
          .filter(i => i.uuid)
          .map(i => ({...i, builtin: true, launching: false, dragover: false}));
        const builtinUUIDs = new Set(builtinApps.map(i => i.uuid));
        const configurableApps = (Array.isArray(r.apps) ? r.apps : [])
          .filter(i => i.uuid && !builtinUUIDs.has(i.uuid))
          .map(i => ({...i, builtin: false, launching: false, dragover: false}));
        this.apps = [...configurableApps, ...builtinApps];
        this.currentApp = r.current_app;
        this.hostName = r.host_name;
        this.hostUUID = r.host_uuid;
        this.listReordered = false;
      })
      .catch(error => console.error("Could not load the app list", error));
    },
    appTemplate() {
      return { name: this.$t('apps.new_app_name'), ...JSON.parse(JSON.stringify(newApp)) };
    },
    newApp() {
      this.editForm = this.appTemplate();
      this.editCoverPreview = "";
      this.showEditForm = true;
    },
    closeEditor() {
      this.closeCoverFinder();
      this.showEditForm = false;
      this.editForm = null;
      this.editCoverPreview = "";
    },
    closeApp() {
      if (confirm(this.$t('apps.close_warning'))) {
        this.actionDisabled = true;
        postJson("./api/apps/close")
        .catch(() => {
          alert(this.$t('apps.close_failed'))
        })
        .finally(() => {
          this.actionDisabled = false;
          this.loadApps()
        });
      }
    },
    editApp(app) {
      if (app.builtin) {
        return;
      }
      this.editForm = Object.assign(
        this.appTemplate(),
        JSON.parse(JSON.stringify(app))
      );
      this.editCoverPreview = this.coverPreviewUrl(app);
      this.showEditForm = true;
    },
    appInitial(app) {
      const name = app?.name?.toString().trim();
      return name ? name.charAt(0).toLocaleUpperCase() : "A";
    },
    appAccent(app) {
      const palette = ["#f8b400", "#ff7a45", "#8b7cf6", "#27b3a2", "#4c8dff", "#e45c92"];
      const name = app?.name?.toString() || "Sunshine 3D";
      let hash = 0;
      for (let i = 0; i < name.length; i += 1) {
        hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
      }
      return palette[Math.abs(hash) % palette.length];
    },
    coverPreviewUrl(app) {
      const path = app?.["image-path"]?.toString().trim() || "";
      if (/^(?:https:\/\/|data:image\/|blob:)/i.test(path)) {
        return path;
      }

      const filename = path.replaceAll("\\", "/").split("/").pop()?.toLowerCase();
      return bundledCovers[filename] || "";
    },
    appCommandSummary(app) {
      if (app?.builtin && app?.["virtual-display"]) {
        return this.$t('apps.private_desktop_summary');
      }

      const command = app?.cmd?.toString().trim();
      if (!command) {
        return this.$t('apps.no_command_summary');
      }

      const unquoted = command.replace(/^"([^"]+)".*$/, "$1");
      const parts = unquoted.split(/[\\/]/);
      return parts[parts.length - 1] || command;
    },
    exportLauncherFile(app) {
      const link = document.createElement('a');
      const fileContent = `# Moonlight 3D app entry
# Exported by Sunshine 3D
# https://github.com/ClassicOldSong/Apollo

[host_uuid] ${this.hostUUID}
[host_name] ${this.hostName}
[app_uuid] ${app.uuid}
[app_name] ${app.name}
`;
      link.href = `data:text/plain;charset=utf-8,${encodeURIComponent(fileContent)}`;
      link.download = `${app.name}.art`;
      link.click();
    },
    showDeleteForm(app) {
      if (app.builtin) {
        return;
      }

      const resp = confirm(
        this.$t('apps.delete_confirm', { name: app.name })
      );
      if (resp) {
        this.actionDisabled = true;
        postJson("./api/apps/delete", { uuid: app.uuid })
        .catch((error) => {
          alert(this.$t('apps.delete_failed', { error: error.message }));
        })
        .finally(() => {
          this.actionDisabled = false;
          this.loadApps();
        });
      }
    },
    addCmd(cmdArr, idx) {
      const template = {
        do: "",
        undo: ""
      };

      template.elevated = false;

      if (idx < 0) {
        cmdArr.push(template);
      } else {
        cmdArr.splice(idx, 0, template);
      }
    },
    showCoverFinder($event) {
      this.coverCandidates = [];
      this.coverSearching = true;
      const ref = this.$refs.coverFinderDropdown;
      if (!ref) {
        console.error("Ref not found!");
        return;
      }
      this.coverFinderDropdown = Dropdown.getInstance(ref);
      if (!this.coverFinderDropdown) {
        this.coverFinderDropdown = new Dropdown(ref);
        if (!this.coverFinderDropdown) {
          return;
        }
      }
      this.coverFinderDropdown.show();
      function getSearchBucket(name) {
        let bucket = name.substring(0, Math.min(name.length, 2)).toLowerCase().replaceAll(/[^a-z\d]/g, '');
        if (!bucket) {
          return '@';
        }
        return bucket;
      }

      function searchCovers(name) {
        if (!name) {
          return Promise.resolve([]);
        }
        let searchName = name.replaceAll(/\s+/g, '.').toLowerCase();

        // Use raw.githubusercontent.com to avoid CORS issues as we migrate the CNAME
        let dbUrl = "https://raw.githubusercontent.com/LizardByte/GameDB/gh-pages";
        let bucket = getSearchBucket(name);
        return fetch(`${dbUrl}/buckets/${bucket}.json`).then(function (r) {
          if (!r.ok) throw new Error("Failed to search covers");
          return r.json();
        }).then(maps => Promise.all(Object.keys(maps).map(id => {
          let item = maps[id];
          if (item.name.replaceAll(/\s+/g, '.').toLowerCase().startsWith(searchName)) {
            return fetch(`${dbUrl}/games/${id}.json`).then(function (r) {
              return r.json();
            }).catch(() => null);
          }
          return null;
        }).filter(item => item)))
          .then(results => results
            .filter(item => item && item.cover && item.cover.url)
            .map(game => {
              const thumb = game.cover.url;
              const dotIndex = thumb.lastIndexOf('.');
              const slashIndex = thumb.lastIndexOf('/');
              if (dotIndex < 0 || slashIndex < 0) {
                return null;
              }
              const slug = thumb.substring(slashIndex + 1, dotIndex);
              return {
                name: game.name,
                key: `igdb_${game.id}`,
                url: `https://images.igdb.com/igdb/image/upload/t_cover_big/${slug}.jpg`,
                saveUrl: `https://images.igdb.com/igdb/image/upload/t_cover_big_2x/${slug}.png`,
              }
            }).filter(item => item));
      }

      searchCovers(this.editForm["name"].toString().trim())
        .then(list => this.coverCandidates = list)
        .catch(() => this.coverCandidates = [])
        .finally(() => this.coverSearching = false);
    },
    closeCoverFinder() {
      const ref = this.$refs.coverFinderDropdown;
      if (!ref) {
        return;
      }
      const dropdown = this.coverFinderDropdown = Dropdown.getInstance(ref);
      if (!dropdown) {
        return;
      }
      dropdown.hide();
    },
    useCover(cover) {
      this.coverFinderBusy = true;
      postJson("./api/covers/upload", {
        key: cover.key,
        url: cover.saveUrl,
      }).then(body => {
        this.editForm["image-path"] = body.path;
        this.editCoverPreview = cover.url;
      })
        .then(() => this.closeCoverFinder())
        .catch(error => console.error("Failed to download covers", error))
        .finally(() => this.coverFinderBusy = false);
    },
    save() {
      this.editForm.name = this.editForm.name.trim();
      if (!this.editForm.name) {
        return;
      }
      // Zero is a valid choice (end the app at once); only a missing or unreadable value
      // falls back to the default.
      const exitTimeout = parseInt(this.editForm["exit-timeout"])
      this.editForm["exit-timeout"] = Number.isFinite(exitTimeout) ? exitTimeout : 5
      this.editForm["image-path"] = this.editForm["image-path"].toString().trim().replace(/"/g, '');
      delete this.editForm.id;
      delete this.editForm.builtin;
      delete this.editForm.launching;
      delete this.editForm.dragover;
      this.actionDisabled = true;
      postJson("./api/apps", this.editForm)
      .then(() => {
        this.showEditForm = false;
        this.editForm = null;
        this.editCoverPreview = "";
        return this.loadApps();
      })
      .catch((error) => {
        alert(this.$t('apps.save_failed') + error.message);
      })
      .finally(() => this.actionDisabled = false);
    },
  },
}
</script>
