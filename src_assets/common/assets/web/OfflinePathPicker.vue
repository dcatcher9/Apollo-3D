<template>
  <dialog
    ref="pathPickerDialog"
    class="offline-path-picker"
    :aria-labelledby="pickerMode === 'input' ? 'offlineInputPickerTitle' : 'offlineOutputPickerTitle'"
    @cancel.prevent="closePathPicker">
    <form class="offline-path-picker-shell" @submit.prevent="confirmPathPicker">
      <header class="offline-path-picker-header">
        <div>
          <p class="apollo-kicker">{{ $t('offline_conversion.this_pc') }}</p>
          <h2
            :id="pickerMode === 'input' ? 'offlineInputPickerTitle' : 'offlineOutputPickerTitle'">
            {{ pickerMode === 'input' ? $t('offline_conversion.choose_input_video') : $t('offline_conversion.choose_output_name') }}
          </h2>
        </div>
        <button
          class="offline-path-picker-close"
          type="button"
          :aria-label="$t('offline_conversion.close_picker')"
          @click="closePathPicker">
          <i class="fa-solid fa-xmark" aria-hidden="true"></i>
        </button>
      </header>

      <div class="offline-path-picker-body">
        <template v-if="pickerMode === 'input'">
          <label class="form-label" for="offlinePickerPath">{{ $t('offline_conversion.folder') }}</label>
          <div class="input-group">
            <input
              id="offlinePickerPath"
              v-model.trim="pickerTypedPath"
              class="form-control monospace"
              type="text"
              autocomplete="off"
              spellcheck="false"
              :placeholder="$t('offline_conversion.folder_placeholder')"
              @keydown.enter.prevent="navigatePicker(pickerTypedPath)">
            <button
              class="btn btn-outline-secondary"
              type="button"
              :disabled="pickerBusy"
              @click="navigatePicker(pickerTypedPath)">
              {{ $t('offline_conversion.go') }}
            </button>
          </div>

          <div class="offline-picker-toolbar">
            <button
              class="btn btn-sm btn-apollo-quiet"
              type="button"
              :disabled="pickerBusy || pickerParentPath === pickerCurrentPath"
              @click="navigatePicker(pickerParentPath)">
              <i class="fa-solid fa-arrow-up" aria-hidden="true"></i>
              {{ $t('offline_conversion.up') }}
            </button>
            <span class="monospace">{{ pickerCurrentPath || $t('offline_conversion.local_drives') }}</span>
          </div>

          <div
            ref="pickerEntries"
            class="offline-picker-entries"
            role="group"
            :aria-label="$t('offline_conversion.files_and_folders')"
            :aria-busy="pickerBusy">
            <div v-if="pickerBusy" class="offline-picker-message" role="status">
              <i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i>
              {{ $t('offline_conversion.loading_files') }}
            </div>
            <template v-else>
              <button
                v-for="(entry, index) in pickerEntries"
                :key="entry.path"
                class="offline-picker-entry"
                :class="{ selected: pickerSelectedPath === entry.path }"
                type="button"
                :tabindex="pickerFocusedIndex === index ? 0 : -1"
                :aria-pressed="entry.type === 'file' ? pickerSelectedPath === entry.path : undefined"
                @focus="pickerFocusedIndex = index"
                @keydown.down.prevent="movePickerFocus(index, 1)"
                @keydown.up.prevent="movePickerFocus(index, -1)"
                @keydown.home.prevent="movePickerFocus(index, -index)"
                @keydown.end.prevent="movePickerFocus(index, pickerEntries.length - index - 1)"
                @click="selectPickerEntry(entry, index)"
                @dblclick="activatePickerEntry(entry)">
                <i
                  class="fa-solid"
                  :class="entry.type === 'directory' ? 'fa-folder' : 'fa-file-video'"
                  aria-hidden="true"></i>
                <span>{{ entry.name }}</span>
                <i
                  v-if="entry.type === 'directory'"
                  class="fa-solid fa-chevron-right offline-picker-entry-arrow"
                  aria-hidden="true"></i>
              </button>
            </template>
            <div
              v-if="!pickerBusy && !pickerEntries.length"
              class="offline-picker-message">
              {{ $t('offline_conversion.no_entries') }}
            </div>
          </div>
          <p v-if="pickerTruncated" class="form-text">
            {{ $t('offline_conversion.truncated') }}
          </p>
        </template>

        <template v-else>
          <div class="offline-output-location">
            <span>{{ $t('offline_conversion.output_folder') }}</span>
            <code>{{ pickerCurrentPath || inputDirectory }}</code>
          </div>
          <div>
            <label class="form-label" for="offlinePickerOutputName">{{ $t('offline_conversion.new_video_file') }}</label>
            <input
              id="offlinePickerOutputName"
              ref="pickerOutputName"
              v-model.trim="pickerOutputName"
              class="form-control monospace"
              type="text"
              autocomplete="off"
              spellcheck="false"
              placeholder="movie-3d.mkv"
              pattern="[^\x2F\x5C]+\.(?:[mM][kK][vV]|[mM][pP]4)"
              required>
            <i18n-t keypath="offline_conversion.picker_output_name_desc" tag="div" class="form-text">
              <template v-slot:mkv><span class="monospace">.mkv</span></template>
              <template v-slot:mp4><span class="monospace">.mp4</span></template>
            </i18n-t>
          </div>
          <div v-if="existingOutputNames.length" class="offline-existing-outputs">
            <span>{{ $t('offline_conversion.existing_outputs') }}</span>
            <div>
              <code v-for="name in existingOutputNames" :key="name">{{ name }}</code>
            </div>
          </div>
        </template>

        <div v-if="pickerError" class="apollo-notice is-danger" role="alert">
          <span class="apollo-notice-icon" aria-hidden="true">
            <i class="fa-solid fa-circle-exclamation"></i>
          </span>
          <div>
            <strong>{{ $t('offline_conversion.path_not_selected') }}</strong>
            <p>{{ pickerError }}</p>
          </div>
        </div>
      </div>

      <footer class="offline-path-picker-actions">
        <span class="monospace">
          {{ pickerMode === 'input' ? pickerSelectedPath : pickerOutputPreview }}
        </span>
        <div>
          <button class="btn btn-apollo-quiet" type="button" @click="closePathPicker">
            {{ $t('_common.cancel') }}
          </button>
          <button
            class="btn btn-apollo-primary"
            type="submit"
            :disabled="pickerBusy || (pickerMode === 'input' && !pickerCanConfirm)">
            {{ pickerMode === 'input' ? $t('offline_conversion.choose_video') : $t('offline_conversion.use_path') }}
          </button>
        </div>
      </footer>
    </form>
  </dialog>
</template>

<script>
import { browseOfflineSbsFiles } from './offline-sbs-api'
import { joinHostPath } from './offline-host-path'

const OUTPUT_NAME_PATTERN = /^[^\\/]+\.(?:mkv|mp4)$/i
let pickerAbortController = null

// The host's own file picker for the offline conversion page. The page opens it for one input
// video or for a new output name in the input video's folder, and receives the chosen value.
export default {
  props: {
    inputDirectory: { type: String, default: '' },
  },
  emits: ['input-chosen', 'output-chosen', 'unsupported'],
  data() {
    return {
      pickerMode: 'input',
      pickerCurrentPath: '',
      pickerParentPath: '',
      pickerTypedPath: '',
      pickerEntries: [],
      pickerSelectedPath: '',
      pickerOutputName: '',
      pickerBusy: false,
      pickerError: '',
      pickerTruncated: false,
      pickerRequestSerial: 0,
      pickerFocusedIndex: -1,
    }
  },
  computed: {
    pickerOutputPreview() {
      return joinHostPath(
        this.pickerCurrentPath || this.inputDirectory,
        this.pickerOutputName.trim(),
      )
    },
    existingOutputNames() {
      return this.pickerEntries
        .filter(entry => entry?.type === 'file')
        .map(entry => String(entry.name || ''))
        .filter(Boolean)
    },
    pickerCanConfirm() {
      if (this.pickerMode === 'input') {
        const selected = this.pickerEntries.find(
          entry => entry?.path === this.pickerSelectedPath,
        )
        return selected?.type === 'file'
      }
      const name = this.pickerOutputName.trim()
      return (
        OUTPUT_NAME_PATTERN.test(name) &&
        !this.existingOutputNames.some(existing =>
          existing.localeCompare(name, undefined, { sensitivity: 'accent' }) === 0
        )
      )
    },
  },
  beforeUnmount() {
    pickerAbortController?.abort()
    this.pickerRequestSerial += 1
  },
  methods: {
    showPathPicker() {
      const dialog = this.$refs.pathPickerDialog
      if (!dialog?.showModal) {
        this.$emit('unsupported')
        return false
      }
      if (!dialog.open) dialog.showModal()
      return true
    },
    resetPathPicker(mode, outputName = '') {
      pickerAbortController?.abort()
      this.pickerRequestSerial += 1
      this.pickerMode = mode
      this.pickerCurrentPath = ''
      this.pickerParentPath = ''
      this.pickerTypedPath = ''
      this.pickerEntries = []
      this.pickerSelectedPath = ''
      this.pickerOutputName = mode === 'output' ? outputName.trim() : ''
      this.pickerBusy = false
      this.pickerError = ''
      this.pickerTruncated = false
      this.pickerFocusedIndex = -1
    },
    async openInputPicker(inputPath) {
      this.resetPathPicker('input')
      if (!this.showPathPicker()) return
      await this.navigatePicker(inputPath.trim())
    },
    async openOutputPicker(outputName) {
      this.resetPathPicker('output', outputName)
      if (!this.showPathPicker()) return
      await this.navigatePicker(this.inputDirectory)
      await this.$nextTick()
      this.$refs.pickerOutputName?.focus()
      this.$refs.pickerOutputName?.select()
    },
    closePathPicker() {
      pickerAbortController?.abort()
      this.pickerRequestSerial += 1
      this.pickerBusy = false
      const dialog = this.$refs.pathPickerDialog
      if (dialog?.open) dialog.close()
    },
    async navigatePicker(path) {
      pickerAbortController?.abort()
      const controller = new AbortController()
      pickerAbortController = controller
      const requestSerial = ++this.pickerRequestSerial
      this.pickerBusy = true
      this.pickerError = ''
      this.pickerSelectedPath = ''
      try {
        const payload = await browseOfflineSbsFiles({
          path: String(path || '').trim(),
          type: 'file',
          signal: controller.signal,
        })
        if (requestSerial !== this.pickerRequestSerial) return
        const listing = payload?.browse && typeof payload.browse === 'object' ?
          payload.browse :
          payload
        this.pickerCurrentPath = String(listing?.path || '')
        this.pickerParentPath = String(listing?.parent || '')
        this.pickerTypedPath = this.pickerCurrentPath
        this.pickerEntries = Array.isArray(listing?.entries) ? listing.entries : []
        this.pickerTruncated = listing?.truncated === true
        this.pickerFocusedIndex = this.pickerEntries.length ? 0 : -1
        this.pickerBusy = false
        if (this.pickerMode === 'input' && this.pickerFocusedIndex >= 0) {
          await this.$nextTick()
          if (requestSerial === this.pickerRequestSerial) {
            this.focusPickerEntry(this.pickerFocusedIndex)
          }
        }
      } catch (error) {
        if (error?.name === 'AbortError') return
        if (requestSerial === this.pickerRequestSerial) {
          this.pickerEntries = []
          this.pickerTruncated = false
          if (error?.status === 403) {
            this.pickerError = this.pickerMode === 'input' ?
              this.$t('offline_conversion.browse_remote_input') :
              this.$t('offline_conversion.browse_remote_output')
          } else if (error?.status === 404) {
            this.pickerError = this.pickerMode === 'input' ?
              this.$t('offline_conversion.browse_missing_input') :
              this.$t('offline_conversion.browse_missing_output')
          } else {
            this.pickerError = error?.message || this.$t('offline_conversion.manager_unreachable')
          }
        }
      } finally {
        if (requestSerial === this.pickerRequestSerial) {
          this.pickerBusy = false
        }
      }
    },
    focusPickerEntry(index) {
      const elements = this.$refs.pickerEntries?.querySelectorAll('.offline-picker-entry')
      if (!elements?.length) return
      const nextIndex = Math.max(0, Math.min(index, elements.length - 1))
      this.pickerFocusedIndex = nextIndex
      elements[nextIndex]?.focus()
    },
    movePickerFocus(index, offset) {
      this.focusPickerEntry(index + offset)
    },
    selectPickerEntry(entry, index = this.pickerFocusedIndex) {
      this.pickerFocusedIndex = index
      if (entry?.type === 'directory') {
        this.navigatePicker(entry.path)
        return
      }
      this.pickerSelectedPath = String(entry?.path || '')
      this.pickerError = ''
    },
    activatePickerEntry(entry) {
      if (entry?.type === 'directory') {
        this.navigatePicker(entry.path)
        return
      }
      this.pickerSelectedPath = String(entry?.path || '')
      this.confirmPathPicker()
    },
    confirmPathPicker() {
      if (this.pickerMode === 'input') {
        if (!this.pickerCanConfirm) return
        this.$emit('input-chosen', this.pickerSelectedPath)
        this.closePathPicker()
        return
      }
      if (!this.$refs.pickerOutputName?.reportValidity() || !this.pickerCanConfirm) {
        if (
          this.existingOutputNames.some(existing =>
            existing.localeCompare(
              this.pickerOutputName.trim(),
              undefined,
              { sensitivity: 'accent' },
            ) === 0
          )
        ) {
          this.pickerError = this.$t('offline_conversion.output_exists')
        }
        return
      }
      this.$emit('output-chosen', this.pickerOutputName.trim())
      this.closePathPicker()
    },
  },
}
</script>
